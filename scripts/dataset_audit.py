#!/usr/bin/env python3
"""Stream provenance-aware JSONL audit without exporting raw text.

Requires a frozen Hugging Face tokenizer JSON to count content tokens.
This is an exploratory sample gate, not a legal opinion or packing audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
from typing import Callable, Iterable

REQUIRED_FIELDS = ("text", "source", "license", "origin", "split", "approved_for_training")
ORIGINS = frozenset({"human", "synthetic", "teacher"})


def content_hash(text: str) -> str:
    """Deterministic NFC-normalized SHA-256 text fingerprint."""
    return hashlib.sha256(unicodedata.normalize("NFC", text).strip().encode("utf-8")).hexdigest()


def read_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    hashes = set()
    with path.open("r", encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            value = raw.strip().lower()
            if not value:
                continue
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"invalid held-out SHA-256 on line {number}")
            hashes.add(value)
    return hashes


def audit(
    files: Iterable[Path],
    count_tokens: Callable[[str], int],
    *,
    tokenizer_sha256: str,
    heldout_hashes: set[str] | None = None,
    target_tokens: int | None = None,
) -> dict:
    """Return aggregates only, never individual text or per-record hashes."""
    counts: Counter[str] = Counter()
    by_origin: dict[str, Counter[str]] = defaultdict(Counter)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()
    heldout_checked = heldout_hashes is not None
    holdout = heldout_hashes if heldout_hashes is not None else set()

    for path in files:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                counts["records_seen"] += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    counts["malformed_json"] += 1
                    continue
                if not isinstance(item, dict) or any(key not in item for key in REQUIRED_FIELDS):
                    counts["invalid_schema"] += 1
                    continue
                text = item["text"]
                if not isinstance(text, str) or not text.strip():
                    counts["empty_or_invalid_text"] += 1
                    continue
                origin = item["origin"]
                if not isinstance(origin, str) or origin not in ORIGINS:
                    counts["invalid_origin"] += 1
                    continue
                if item["split"] != "train":
                    counts["non_train_records"] += 1
                    continue
                if not isinstance(item["source"], str) or not item["source"].strip():
                    counts["invalid_source"] += 1
                    continue
                if not isinstance(item["license"], str) or not item["license"].strip():
                    counts["missing_license"] += 1
                    continue
                if item["approved_for_training"] is not True:
                    counts["unapproved_usage"] += 1
                    continue
                if origin == "teacher" and item.get("teacher_use_approved") is not True:
                    counts["unapproved_teacher_usage"] += 1
                    continue
                fingerprint = content_hash(text)
                if fingerprint in holdout:
                    counts["heldout_overlap"] += 1
                    continue
                if fingerprint in seen:
                    counts["duplicates"] += 1
                    continue
                tokens = count_tokens(text)
                if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                    raise ValueError("tokenizer returned an invalid token count")
                seen.add(fingerprint)
                counts["accepted_records"] += 1
                counts["accepted_content_tokens"] += tokens
                by_origin[origin]["records"] += 1
                by_origin[origin]["content_tokens"] += tokens
                source = item["source"].strip()
                by_source[source]["records"] += 1
                by_source[source]["content_tokens"] += tokens

    report = {
        "schema_version": 1,
        "mode": "exact_content_token_count_no_packing",
        "tokenizer_sha256": tokenizer_sha256,
        "heldout_hashes_supplied": heldout_checked,
        "heldout_hashes_count": len(holdout) if heldout_checked else None,
        "counts": dict(sorted(counts.items())),
        "by_origin": {k: dict(v) for k, v in sorted(by_origin.items())},
        "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
        "manual_review_required": True,
        "limitations": [
            "License and teacher-use flags are metadata, not independently verified permission.",
            "Only exact normalized-text duplicates are detected; near-duplicates are not detected.",
            "This counts tokenizer content IDs only, not EOS/BOS, padding, packing slots or optimizer updates.",
            "Held-out leakage is checked only against supplied SHA-256 fingerprints.",
        ],
    }
    if target_tokens is not None:
        report["target_content_tokens"] = target_tokens
        report["target_progress_fraction"] = counts["accepted_content_tokens"] / target_tokens
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="UTF-8 JSONL input files")
    parser.add_argument("--tokenizer-json", required=True, type=Path, help="Frozen HF tokenizer.json")
    parser.add_argument("--heldout-hashes", type=Path, help="Optional eval SHA-256 hashes (one per line)")
    parser.add_argument("--target-tokens", type=int, help="Content-token goal")
    parser.add_argument("--output", type=Path, help="Write aggregate JSON here instead of stdout")
    args = parser.parse_args(argv)
    if args.target_tokens is not None and args.target_tokens <= 0:
        parser.error("--target-tokens must be positive")
    try:
        from tokenizers import Tokenizer
    except ImportError:
        parser.error("install tokenizers: python -m pip install tokenizers")
    try:
        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        tokenizer_sha = hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest()
        hashes = read_hashes(args.heldout_hashes) if args.heldout_hashes else None
        result = audit(
            args.files,
            lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
            tokenizer_sha256=tokenizer_sha,
            heldout_hashes=hashes,
            target_tokens=args.target_tokens,
        )
        output = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"audit error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
