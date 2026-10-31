#!/usr/bin/env python3
"""Independently re-tokenize source samples and compare producer-emitted token IDs.

Inputs stay local. The aggregate report omits source text, token IDs, hashes,
and local file paths. This is a *sample alignment* gate, not a full packing audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Iterable

HEXDIGITS = frozenset("0123456789abcdef")


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in HEXDIGITS for c in value.lower())


def exact_source_hash(text: str) -> str:
    """Hash raw UTF-8 bytes, unlike the NFC/strip duplicate filter."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compare_samples(
    source_files: Iterable[Path],
    emitted_files: Iterable[Path],
    tokenize: Callable[[str], list[int]],
    *,
    tokenizer_sha256: str,
    expected_tokenizer_sha256: str,
    vocab_size: int = 65536,
    max_records: int = 10000,
    max_errors: int = 20,
) -> dict:
    if not valid_sha256(tokenizer_sha256) or not valid_sha256(expected_tokenizer_sha256):
        raise ValueError("tokenizer hashes must be 64-character hexadecimal SHA-256")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if type(max_records) is not int or max_records <= 0:
        raise ValueError("max_records must be positive")
    if type(max_errors) is not int or max_errors <= 0:
        raise ValueError("max_errors must be positive")

    counts: Counter[str] = Counter()
    errors: list[str] = []

    def fail(code: str) -> None:
        counts[code] += 1
        if len(errors) < max_errors:
            errors.append(code)

    # Source sample is deliberately bounded; never use this on an entire large corpus.
    sources: dict[str, str] = {}
    for path in source_files:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                counts["source_rows"] += 1
                if counts["source_rows"] > max_records:
                    raise ValueError("source sample exceeds max_records")
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    fail("invalid_source_json")
                    continue
                if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                    fail("invalid_source_schema")
                    continue
                text = item["text"]
                if not text:
                    fail("empty_source_text")
                    continue
                digest = exact_source_hash(text)
                if digest in sources:
                    fail("duplicate_source_hash")
                    continue
                sources[digest] = text
                counts["valid_source_rows"] += 1

    seen: set[str] = set()
    for path in emitted_files:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                counts["emitted_rows"] += 1
                if counts["emitted_rows"] > max_records:
                    raise ValueError("emitted sample exceeds max_records")
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    fail("invalid_emitted_json")
                    continue
                if not isinstance(item, dict):
                    fail("invalid_emitted_schema")
                    continue
                digest = item.get("source_sha256")
                ids = item.get("token_ids")
                if not valid_sha256(digest) or not isinstance(ids, list) or any(
                    type(token) is not int or token < 0 or token >= vocab_size for token in ids
                ):
                    fail("invalid_emitted_schema")
                    continue
                digest = digest.lower()
                if digest in seen:
                    fail("duplicate_emitted_source")
                    continue
                seen.add(digest)
                if digest not in sources:
                    fail("unexpected_emitted_source")
                    continue
                result = tokenize(sources[digest])
                if not isinstance(result, list) or any(
                    type(token) is not int or token < 0 or token >= vocab_size for token in result
                ):
                    raise ValueError("tokenizer returned invalid token IDs")
                counts["compared_records"] += 1
                if ids != result:
                    fail("token_id_mismatch")
                else:
                    counts["matched_records"] += 1
                    counts["matched_content_tokens"] += len(ids)

    missing = len(sources) - len(seen & sources.keys())
    if missing:
        counts["missing_emitted_sources"] = missing
        if len(errors) < max_errors:
            errors.append("missing_emitted_sources")
    if not sources:
        fail("no_valid_source_rows")
    if counts["emitted_rows"] == 0:
        fail("no_emitted_rows")

    tokenizer_matches = tokenizer_sha256.lower() == expected_tokenizer_sha256.lower()
    if not tokenizer_matches:
        fail("tokenizer_sha256_mismatch")

    status = "PASS_SAMPLE_ALIGNMENT" if not errors and not any(
        counts[key] for key in (
            "invalid_source_json", "invalid_source_schema", "empty_source_text",
            "duplicate_source_hash", "invalid_emitted_json", "invalid_emitted_schema",
            "duplicate_emitted_source", "unexpected_emitted_source", "token_id_mismatch",
            "missing_emitted_sources", "no_valid_source_rows", "no_emitted_rows",
            "tokenizer_sha256_mismatch",
        )
    ) else "FAIL"

    return {
        "schema_version": 1,
        "status": status,
        "tokenizer_sha256": tokenizer_sha256.lower(),
        "tokenizer_sha256_matches_expected": tokenizer_matches,
        "vocab_size": vocab_size,
        "max_records": max_records,
        "counts": dict(sorted(counts.items())),
        "error_codes": errors,
        "limitations": [
            "Only explicitly supplied sample documents are checked, not the complete corpus.",
            "Emitted token_ids must exclude EOS/BOS and packing padding.",
            "Does not validate actual positions inside packed training batches.",
            "Source and producer records must be independently collected from the same samples.",
            "Comparing content tokens does not establish permission, contamination safety or model quality.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True, type=Path)
    parser.add_argument("--emitted", nargs="+", required=True, type=Path)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--expected-tokenizer-sha256", required=True,
                        help="Independently pinned frozen tokenizer file SHA-256")
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--output", type=Path, help="Aggregate JSON report destination")
    args = parser.parse_args(argv)

    try:
        if not valid_sha256(args.expected_tokenizer_sha256):
            raise ValueError("expected tokenizer SHA-256 must be 64 hexadecimal characters")
        from tokenizers import Tokenizer
        artifact_sha = hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest()
        if artifact_sha.lower() != args.expected_tokenizer_sha256.lower():
            raise ValueError("tokenizer artifact does not match independently pinned SHA-256")
        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        if tokenizer.get_vocab_size(with_added_tokens=True) != args.vocab_size:
            raise ValueError("tokenizer vocabulary size does not match required config")
        result = compare_samples(
            args.sources,
            args.emitted,
            lambda txt: tokenizer.encode(txt, add_special_tokens=False).ids,
            tokenizer_sha256=artifact_sha,
            expected_tokenizer_sha256=args.expected_tokenizer_sha256,
            vocab_size=args.vocab_size,
            max_records=args.max_records,
        )
        output = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 0 if result["status"] == "PASS_SAMPLE_ALIGNMENT" else 2
    except ImportError:
        print("alignment error: install package 'tokenizers'", file=sys.stderr)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        print(f"alignment error: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
