#!/usr/bin/env python3
"""Verify 17-step pre/post-checkpoint replay evidence against an uninterrupted run.

Reads *producer-declared* state digests and a sidecar describing a real checkpoint
file. Does not execute TPU training or independently attest checkpoint content.
Never put private checkpoint bytes or training data into the public repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

DIGEST_FIELDS = (
    "batch_sha256", "model_sha256", "optimizer_sha256",
    "scheduler_sha256", "rng_sha256", "loader_sha256",
    "dataset_manifest_sha256", "tokenizer_sha256", "config_sha256",
)
STATE_FIELDS = (
    "model_sha256", "optimizer_sha256", "scheduler_sha256",
    "rng_sha256", "loader_sha256",
)
IDENTITY_FIELDS = ("dataset_manifest_sha256", "tokenizer_sha256", "config_sha256")


def valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value.lower()
    )


def sha_file(path: Path) -> tuple[str, int]:
    hasher, length = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            hasher.update(chunk)
            length += len(chunk)
    return hasher.hexdigest(), length


def load_trace(path: Path, expected_steps: list[int]) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if len(rows) >= len(expected_steps):
                raise ValueError("trace contains more rows than expected")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in trace line {number}") from exc
            if not isinstance(row, dict) or type(row.get("step")) is not int:
                raise ValueError(f"malformed trace line {number}")
            if row["step"] != expected_steps[len(rows)]:
                raise ValueError(f"unexpected/duplicate step in trace line {number}")
            if any(not valid_sha(row.get(field)) for field in DIGEST_FIELDS):
                raise ValueError(f"missing or malformed state hash in trace line {number}")
            if type(row.get("data_cursor")) is not int or row["data_cursor"] < 1:
                raise ValueError(f"invalid loader cursor in trace line {number}")
            if (type(row.get("loss")) not in (int, float)
                    or not math.isfinite(row["loss"])):
                raise ValueError(f"non-finite or missing loss in trace line {number}")
            if rows and row["data_cursor"] <= rows[-1]["data_cursor"]:
                raise ValueError("loader cursor did not advance")
            rows.append(row)
    if len(rows) != len(expected_steps):
        raise ValueError("trace has missing steps")
    return rows


def verify_replay(
    *,
    baseline: Path,
    before: Path,
    after: Path,
    checkpoint: Path,
    checkpoint_sidecar: Path,
    expected_steps: int = 17,
    split_step: int = 8,
    expected_dataset_sha256: str,
    expected_tokenizer_sha256: str,
    expected_config_sha256: str,
    loss_atol: float = 0.0,
) -> dict:
    if (type(expected_steps) is not int or type(split_step) is not int
            or not 1 <= split_step < expected_steps):
        raise ValueError("expected_steps and split_step must be positive with split < total")
    if (type(loss_atol) not in (int, float) or not math.isfinite(loss_atol)
            or loss_atol < 0):
        raise ValueError("loss_atol must be finite and nonnegative")
    pins = (expected_dataset_sha256, expected_tokenizer_sha256, expected_config_sha256)
    if any(not valid_sha(x) for x in pins):
        raise ValueError("all three externally pinned digests must be valid SHA-256")
    expected_id = dict(zip(IDENTITY_FIELDS, (x.lower() for x in pins)))

    reference = load_trace(baseline, list(range(1, expected_steps + 1)))
    prefix = load_trace(before, list(range(1, split_step + 1)))
    suffix = load_trace(after, list(range(split_step + 1, expected_steps + 1)))
    combined = prefix + suffix
    result_errors = []

    def issue(code: str) -> None:
        if code not in result_errors:
            result_errors.append(code)

    for row in reference + combined:
        for key, pin in expected_id.items():
            if row[key].lower() != pin:
                issue("RUN_IDENTITY_MISMATCH")
    for index, (ref, replay) in enumerate(zip(reference, combined), 1):
        if ref["data_cursor"] != replay["data_cursor"]:
            issue("LOADER_CURSOR_MISMATCH")
        if ref["batch_sha256"].lower() != replay["batch_sha256"].lower():
            issue("BATCH_MISMATCH")
        for key in STATE_FIELDS:
            if ref[key].lower() != replay[key].lower():
                issue("STATE_MISMATCH")
        if abs(ref["loss"] - replay["loss"]) > loss_atol:
            issue("LOSS_MISMATCH")
        if index > split_step:
            if replay.get("resumed_from_step") != split_step or (
                type(replay.get("resumed_from_step")) is not int
            ):
                issue("RESUME_MARKER_MISSING")
    if suffix[0]["data_cursor"] <= prefix[-1]["data_cursor"]:
        issue("CURSOR_DID_NOT_ADVANCE_AFTER_RESTORE")

    metadata = json.loads(checkpoint_sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint sidecar must be an object")
    expected_meta_keys = set(STATE_FIELDS) | set(IDENTITY_FIELDS) | {
        "schema_version", "checkpoint_step", "data_cursor",
        "checkpoint_file_sha256", "checkpoint_file_size",
    }
    if not expected_meta_keys.issubset(metadata):
        issue("INCOMPLETE_CHECKPOINT_METADATA")
    elif (metadata.get("schema_version") != 1 or
          type(metadata.get("checkpoint_step")) is not int or
          metadata["checkpoint_step"] != split_step or
          type(metadata.get("data_cursor")) is not int or
          metadata["data_cursor"] != prefix[-1]["data_cursor"]):
        issue("CHECKPOINT_POSITION_MISMATCH")
    if any(not valid_sha(metadata.get(k)) for k in (
        *STATE_FIELDS, *IDENTITY_FIELDS, "checkpoint_file_sha256"
    )):
        issue("INVALID_CHECKPOINT_DIGEST")
    else:
        for key in STATE_FIELDS + IDENTITY_FIELDS:
            if metadata[key].lower() != prefix[-1][key].lower():
                issue("CHECKPOINT_STATE_MISMATCH")
    if (not isinstance(metadata.get("checkpoint_file_size"), int)
            or type(metadata.get("checkpoint_file_size")) is not int
            or metadata.get("checkpoint_file_size", 0) <= 0):
        issue("INVALID_CHECKPOINT_SIZE")
    digest, size = sha_file(checkpoint)
    if metadata.get("checkpoint_file_sha256", "").lower() != digest or (
        metadata.get("checkpoint_file_size") != size
    ):
        issue("CHECKPOINT_BYTES_MISMATCH")

    return {
        "schema_version": 1,
        "status": "PASS_DECLARED_STRICT_REPLAY" if not result_errors else "FAIL",
        "dataset_manifest_sha256": expected_id["dataset_manifest_sha256"],
        "tokenizer_sha256": expected_id["tokenizer_sha256"],
        "config_sha256": expected_id["config_sha256"],
        "expected_steps": expected_steps,
        "split_step": split_step,
        "traces_compared": 2,
        "steps_compared": expected_steps,
        "checkpoint_bytes_verified": not any(
            code in result_errors for code in (
                "CHECKPOINT_BYTES_MISMATCH", "INVALID_CHECKPOINT_SIZE",
                "INVALID_CHECKPOINT_DIGEST",
            )
        ),
        "error_codes": sorted(result_errors),
        "training_started": False,
        "real_tpu_gate_completed": False,
        "limitations": [
            "Producer-supplied state hashes and resume markers are not cryptographic proof of execution.",
            "Bitwise hash equality is strict and may be inappropriate for nondeterministic TPU kernels.",
            "Checkpoint sidecar metadata is self-reported; independently inspect actual runtime checkpoint contents.",
            "Does not check model quality, full optimizer state semantics, training data licensing or durable backup restore.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "before", "after", "checkpoint", "checkpoint-sidecar"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("dataset", "tokenizer", "config"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    parser.add_argument("--expected-steps", type=int, default=17)
    parser.add_argument("--split-step", type=int, default=8)
    parser.add_argument("--loss-atol", type=float, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_replay(
            baseline=args.baseline, before=args.before, after=args.after,
            checkpoint=args.checkpoint, checkpoint_sidecar=args.checkpoint_sidecar,
            expected_steps=args.expected_steps, split_step=args.split_step,
            expected_dataset_sha256=args.expected_dataset_sha256,
            expected_tokenizer_sha256=args.expected_tokenizer_sha256,
            expected_config_sha256=args.expected_config_sha256,
            loss_atol=args.loss_atol,
        )
        output = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        return 0 if result["status"] == "PASS_DECLARED_STRICT_REPLAY" else 2
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"resume replay error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
