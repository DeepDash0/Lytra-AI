#!/usr/bin/env python3
"""Audit emitted packed-token JSONL batches without exporting token IDs.

Each row contains:
  step, seq_index, input_ids, attention_mask, special_tokens_mask

The masks classify every slot into content (attention=1, special=0),
special (attention=1, special=1), or padding (attention=0, special=0).
This validates emitted traces, but not their upstream provenance or licenses.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable


def _integer(value: object, min_value: int = 0) -> bool:
    return type(value) is int and value >= min_value


def audit_packed(
    paths: Iterable[Path],
    *,
    sequence_length: int = 4096,
    sequences_per_update: int = 512,
    vocab_size: int = 65536,
    audit_report: dict | None = None,
    tokenizer_sha256: str | None = None,
    max_errors: int = 20,
) -> dict:
    if not all(_integer(n, 1) for n in (sequence_length, sequences_per_update, vocab_size, max_errors)):
        raise ValueError("sequence_length, sequences_per_update, vocab_size and max_errors must be positive integers")

    counts: Counter[str] = Counter()
    errors: list[str] = []
    current_step: int | None = None
    next_sequence = 0
    rows_read = 0

    def error(message: str) -> None:
        if len(errors) < max_errors:
            errors.append(message)
        counts["invalid_rows_or_groups"] += 1

    def finish_step() -> None:
        if current_step is None:
            return
        if next_sequence != sequences_per_update:
            error(f"step {current_step}: incomplete update ({next_sequence}/{sequences_per_update} sequences)")
        else:
            counts["complete_optimizer_updates"] += 1

    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                rows_read += 1
                location = f"{Path(path).name}:{number}"
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    error(f"{location}: invalid JSON")
                    continue
                if not isinstance(item, dict):
                    error(f"{location}: expected JSON object")
                    continue
                step = item.get("step")
                index = item.get("seq_index")
                if not _integer(step) or not _integer(index):
                    error(f"{location}: step and seq_index must be nonnegative integers")
                    continue

                if current_step is None:
                    current_step = step
                elif step != current_step:
                    finish_step()
                    if step != current_step + 1:
                        error(f"{location}: noncontiguous optimizer step ({current_step} -> {step})")
                    current_step = step
                    next_sequence = 0

                # Each step must arrive in canonical index order; this detects
                # missing, repeated or misordered sequences.
                if index != next_sequence:
                    error(f"{location}: expected seq_index {next_sequence}, got {index}")
                    next_sequence = index + 1
                else:
                    next_sequence += 1
                if index >= sequences_per_update:
                    error(f"{location}: seq_index {index} out of range")
                input_ids = item.get("input_ids")
                attention = item.get("attention_mask")
                special = item.get("special_tokens_mask")
                if not all(isinstance(v, list) and len(v) == sequence_length
                           for v in (input_ids, attention, special)):
                    error(f"{location}: arrays must each contain {sequence_length} slots")
                    continue
                if any(not _integer(token) or token >= vocab_size for token in input_ids):
                    error(f"{location}: invalid token ID for vocab {vocab_size}")
                    continue
                if any(type(a) is not int or a not in (0, 1) or
                       type(s) is not int or s not in (0, 1) or s > a
                       for a, s in zip(attention, special)):
                    error(f"{location}: mask contains invalid bit or special token in padding")
                    continue

                # Packing must use right-padding only; interior zeros hide missing tokens.
                if any(a == 0 and b == 1 for a, b in zip(attention, attention[1:])):
                    error(f"{location}: attention mask contains interior padding")
                    continue
                padding = sequence_length - sum(attention)
                specials = sum(special)
                content = sum(attention) - specials
                counts["content_tokens"] += content
                counts["special_tokens"] += specials
                counts["padding_tokens"] += padding
                counts["token_slots"] += sequence_length
                counts["validated_sequences"] += 1
    finish_step()

    if rows_read == 0:
        error("no packed sequences supplied")
    if counts["validated_sequences"] != rows_read:
        error(f"only {counts['validated_sequences']} of {rows_read} rows were valid")
    if counts["token_slots"] != (
        counts["content_tokens"] + counts["special_tokens"] + counts["padding_tokens"]
    ):
        error("token slot conservation failed")

    audit_match: bool | None = None
    tokenizer_match: bool | None = None
    if audit_report is not None:
        if not isinstance(audit_report, dict):
            raise ValueError("upstream audit report must be a JSON object")
        expected = audit_report.get("counts", {}).get("accepted_content_tokens")
        if not _integer(expected):
            error("upstream audit report has no valid accepted_content_tokens")
        else:
            audit_match = counts["content_tokens"] == expected
            if not audit_match:
                error(f"upstream content-token mismatch: expected {expected}, got {counts['content_tokens']}")
        expected_sha = audit_report.get("tokenizer_sha256")
        if (tokenizer_sha256 is None or not isinstance(expected_sha, str)
                or len(expected_sha) != 64 or len(tokenizer_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_sha.lower())
                or any(c not in "0123456789abcdef" for c in tokenizer_sha256.lower())):
            error("tokenizer SHA-256 must be supplied and present in upstream audit report")
        else:
            tokenizer_match = expected_sha.lower() == tokenizer_sha256.lower()
            if not tokenizer_match:
                error("tokenizer SHA-256 differs from upstream audit report")

    if errors:
        status = "FAIL"
    elif audit_match is True and tokenizer_match is True:
        status = "PASS_DECLARED_TRACE"
    else:
        status = "STRUCTURE_ONLY_PASS"

    return {
        "schema_version": 1,
        "status": status,
        "sequence_length": sequence_length,
        "sequences_per_update": sequences_per_update,
        "vocab_size": vocab_size,
        "tokenizer_sha256_declared": tokenizer_sha256,
        "input_rows_seen": rows_read,
        "counts": dict(sorted(counts.items())),
        "content_tokens_match_upstream_audit": audit_match,
        "tokenizer_sha256_matches_upstream_audit": tokenizer_match,
        "errors": errors,
        "limitations": [
            "Masks and token IDs are producer-provided; this is not independent text-to-ID validation.",
            "PASS_DECLARED_TRACE verifies consistency with upstream metadata, not data rights, model quality or training readiness.",
            "Missing upstream audit produces STRUCTURE_ONLY_PASS even if masks and batch geometry are valid.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="Packed JSONL trace files")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--sequences-per-update", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--audit-report", type=Path, help="Prior dataset_audit.py aggregate report")
    parser.add_argument("--tokenizer-sha256", help="SHA-256 supplied by the sequence producer")
    parser.add_argument("--output", type=Path, help="Write aggregate JSON here")
    args = parser.parse_args(argv)
    try:
        prior = json.loads(args.audit_report.read_text(encoding="utf-8")) if args.audit_report else None
        result = audit_packed(
            args.files,
            sequence_length=args.sequence_length,
            sequences_per_update=args.sequences_per_update,
            vocab_size=args.vocab_size,
            audit_report=prior,
            tokenizer_sha256=args.tokenizer_sha256,
        )
        output = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 2 if result["status"] == "FAIL" else 0
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"packing audit error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
