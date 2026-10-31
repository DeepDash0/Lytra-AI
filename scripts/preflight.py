#!/usr/bin/env python3
"""Combine Lytra V1 audit evidence into fail-closed *technical* preflight gates.

This tool never starts training. Passing its checks is NOT production approval:
manual rights review, data completeness, TPU streaming/resume and durable restore
remain independent requirements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from scripts.data_manifest import verify_manifest
    from scripts.tokenizer_alignment import valid_sha256
except ModuleNotFoundError:
    from data_manifest import verify_manifest
    from tokenizer_alignment import valid_sha256


def _int(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def evaluate_preflight(
    *,
    dataset: dict | None = None,
    packing: dict | None = None,
    alignment: dict | None = None,
    manifest_check: dict | None = None,
    pinned_tokenizer_sha256: str | None = None,
    min_content_tokens: int = 1,
) -> dict:
    if not _int(min_content_tokens, 1):
        raise ValueError("min_content_tokens must be a positive integer")
    checks: dict[str, dict] = {}
    def mark(name: str, pass_condition: bool, detail: str) -> None:
        checks[name] = {"status": "PASS" if pass_condition else "BLOCKED",
                        "reason": "ok" if pass_condition else detail}

    pinned = pinned_tokenizer_sha256.lower() if valid_sha256(pinned_tokenizer_sha256) else None
    mark("pinned_tokenizer", pinned is not None, "Frozen tokenizer SHA-256 not pinned or malformed.")

    if not isinstance(dataset, dict):
        mark("dataset_audit", False, "Dataset audit report missing.")
    else:
        c = dataset.get("counts")
        c = c if isinstance(c, dict) else {}
        accepted = c.get("accepted_content_tokens")
        blockers = (
            "malformed_json", "invalid_schema", "invalid_origin", "heldout_overlap",
            "unapproved_usage", "unapproved_teacher_usage",
        )
        blocked = any(not _int(c.get(name, 0)) or c.get(name, 0) != 0 for name in blockers)
        mark("dataset_audit",
             dataset.get("schema_version") == 1 and
             dataset.get("mode") == "exact_content_token_count_no_packing" and
             bool(pinned) and str(dataset.get("tokenizer_sha256", "")).lower() == pinned and
             dataset.get("heldout_hashes_supplied") is True and
             _int(dataset.get("heldout_hashes_count"), 1) and
             _int(accepted, min_content_tokens) and not blocked,
             "Dataset report invalid, tokenizer differs, heldout checks missing, flagged records or token minimum unmet.")

    if not isinstance(packing, dict):
        mark("packed_batch", False, "Packed-batch audit report missing.")
    else:
        c = packing.get("counts")
        c = c if isinstance(c, dict) else {}
        same_accepted = isinstance(dataset, dict) and isinstance(dataset.get("counts"), dict) and (
            c.get("content_tokens") == dataset["counts"].get("accepted_content_tokens")
        )
        seq_length = packing.get("sequence_length")
        sequences = packing.get("sequences_per_update")
        slot_expectation = seq_length * sequences if _int(seq_length, 1) and _int(sequences, 1) else None
        slots = c.get("token_slots")
        token_parts = [c.get(k) for k in ("content_tokens", "special_tokens", "padding_tokens")]
        mark("packed_batch",
             packing.get("schema_version") == 1 and
             packing.get("status") == "PASS_DECLARED_TRACE" and
             bool(pinned) and str(packing.get("tokenizer_sha256_declared", "")).lower() == pinned and
             packing.get("tokenizer_sha256_matches_upstream_audit") is True and
             packing.get("content_tokens_match_upstream_audit") is True and same_accepted and
             seq_length == 4096 and sequences == 512 and packing.get("vocab_size") == 65536 and
             _int(c.get("complete_optimizer_updates"), 1) and
             _int(slots, slot_expectation or 1) and slots % (slot_expectation or 1) == 0 and
             all(_int(x) for x in token_parts) and slots == sum(token_parts) and
             not packing.get("errors"),
             "Packing report incomplete, not canonical 4096x512, or content/tokenizer totals disagree.")

    if not isinstance(alignment, dict):
        mark("independent_alignment", False, "Source-to-tokenizer sample comparison report missing.")
    else:
        c = alignment.get("counts")
        c = c if isinstance(c, dict) else {}
        mark("independent_alignment",
             alignment.get("schema_version") == 1 and
             alignment.get("status") == "PASS_SAMPLE_ALIGNMENT" and
             alignment.get("tokenizer_sha256_matches_expected") is True and
             bool(pinned) and str(alignment.get("tokenizer_sha256", "")).lower() == pinned and
             alignment.get("vocab_size") == 65536 and
             _int(c.get("matched_records"), 1) and
             c.get("matched_records") == c.get("valid_source_rows") and
             not alignment.get("error_codes"),
             "Sample alignment missing, mismatched or tokenizer identity differs.")

    mark("shard_integrity",
         isinstance(manifest_check, dict) and manifest_check.get("status") == "PASS" and
         _int(manifest_check.get("file_count"), 1),
         "Missing/failed file-content manifest verification.")

    successful = all(v["status"] == "PASS" for v in checks.values())
    return {
        "schema_version": 1,
        "status": "PASS_TECHNICAL_ONLY" if successful else "BLOCKED",
        "checks": checks,
        "manual_rights_review": "NOT_ATTESTED",
        "production_ready": False,
        "training_started": False,
        "remaining_production_gates": [
            "Independent permission and source-provenance review",
            "Complete immutable training-manifest and contamination review",
            "17-step real-data TPU streaming and checkpoint-resume validation",
            "Durable checkpoint restore and full two-stage production configuration",
        ],
    }


def _read(path: Path | None) -> dict | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("report must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-report", type=Path)
    parser.add_argument("--packing-report", type=Path)
    parser.add_argument("--alignment-report", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--shard-root", type=Path)
    parser.add_argument("--expected-tokenizer-sha256")
    parser.add_argument("--min-content-tokens", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if (args.manifest is None) != (args.shard_root is None):
            raise ValueError("manifest and shard-root must both be provided")
        manifest_check = (verify_manifest(args.shard_root, _read(args.manifest))
                          if args.manifest else None)
        report = evaluate_preflight(
            dataset=_read(args.dataset_report),
            packing=_read(args.packing_report),
            alignment=_read(args.alignment_report),
            manifest_check=manifest_check,
            pinned_tokenizer_sha256=args.expected_tokenizer_sha256,
            min_content_tokens=args.min_content_tokens,
        )
        content = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
        else:
            sys.stdout.write(content)
        return 0 if report["status"] == "PASS_TECHNICAL_ONLY" else 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"preflight error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
