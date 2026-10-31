#!/usr/bin/env python3
"""Synthetic-only 17-step checkpoint/replay fixture and CLI smoke.

A tiny deterministic integer state machine exercises model, optimizer,
scheduler, RNG, and loader-position persistence. This is NOT ML/TPU training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from scripts.resume_replay import verify_replay, sha_file, STATE_FIELDS, IDENTITY_FIELDS


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


PINS = {
    "dataset_manifest_sha256": digest("toy-dataset-v1"),
    "tokenizer_sha256": digest("toy-tokenizer-v1"),
    "config_sha256": digest("toy-config-v1"),
}


def init_state() -> dict:
    return {"model": 77123, "momentum": 0, "scheduler": 0, "rng": 137, "cursor": 0}


def step(state: dict, resumed_from_step: int | None = None) -> dict:
    state["rng"] = (1664525 * state["rng"] + 1013904223) % (2**32)
    batch = (state["cursor"] * 271 + (state["rng"] % 997)) % 100000
    state["momentum"] = (state["momentum"] * 3 + batch + 17) % 999983
    state["model"] = (state["model"] * 5 + state["momentum"]) % 1000003
    state["scheduler"] += 1
    state["cursor"] += 1
    result = {
        "step": state["scheduler"],
        "data_cursor": state["cursor"],
        "batch_sha256": digest({"cursor": state["cursor"], "batch": batch}),
        "model_sha256": digest(state["model"]),
        "optimizer_sha256": digest(state["momentum"]),
        "scheduler_sha256": digest(state["scheduler"]),
        "rng_sha256": digest(state["rng"]),
        "loader_sha256": digest(state["cursor"]),
        "loss": (state["model"] % 10000) / 10000,
        **PINS,
    }
    if resumed_from_step is not None:
        result["resumed_from_step"] = resumed_from_step
    return result


def write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_fixture(directory: Path, total: int = 17, split: int = 8) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    baseline_path = directory / "baseline.jsonl"
    before_path = directory / "before.jsonl"
    after_path = directory / "after.jsonl"
    checkpoint_path = directory / "checkpoint.toy.json"
    sidecar_path = directory / "checkpoint.meta.json"

    uninterrupted = init_state()
    full_rows = [step(uninterrupted) for _ in range(total)]
    write_trace(baseline_path, full_rows)
    interrupted = init_state()
    before_rows = [step(interrupted) for _ in range(split)]
    write_trace(before_path, before_rows)

    # Explicit durable checkpoint file, then a *new* dict loaded from its bytes.
    # No in-memory state object is reused for the restored portion.
    checkpoint_path.write_text(json.dumps(interrupted, sort_keys=True), encoding="utf-8")
    file_sha, file_size = sha_file(checkpoint_path)
    sidecar = {
        "schema_version": 1,
        "checkpoint_step": split,
        "data_cursor": before_rows[-1]["data_cursor"],
        "checkpoint_file_sha256": file_sha,
        "checkpoint_file_size": file_size,
        **{k: before_rows[-1][k] for k in (*STATE_FIELDS, *IDENTITY_FIELDS)},
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    restored = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    after_rows = [step(restored, resumed_from_step=split) for _ in range(total - split)]
    write_trace(after_path, after_rows)

    return {
        "baseline": baseline_path,
        "before": before_path,
        "after": after_path,
        "checkpoint": checkpoint_path,
        "checkpoint_sidecar": sidecar_path,
        "expected_steps": total,
        "split_step": split,
        "expected_dataset_sha256": PINS["dataset_manifest_sha256"],
        "expected_tokenizer_sha256": PINS["tokenizer_sha256"],
        "expected_config_sha256": PINS["config_sha256"],
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lytra_synthetic_resume_") as tmp:
        files = make_fixture(Path(tmp))
        report = verify_replay(**files)
        assert report["status"] == "PASS_DECLARED_STRICT_REPLAY", report
        assert report["steps_compared"] == 17
        assert report["checkpoint_bytes_verified"]
        checkpoint = files["checkpoint"]
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
        report_bad = verify_replay(**files)
        assert report_bad["status"] == "FAIL"
        assert "CHECKPOINT_BYTES_MISMATCH" in report_bad["error_codes"]
    print("SYNTHETIC_RESUME_SMOKE=PASS (17-step strict replay, tamper detection)")


if __name__ == "__main__":
    main()
