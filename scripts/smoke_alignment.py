#!/usr/bin/env python3
"""Exercise tokenizer_alignment CLI with a disposable synthetic tokenizer only."""
import hashlib
import json
from pathlib import Path
import tempfile

from scripts.tokenizer_alignment import exact_source_hash, main as alignment_main


def main() -> None:
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers
    except ImportError as exc:
        raise SystemExit("Install 'tokenizers' in a local virtual environment.") from exc

    with tempfile.TemporaryDirectory(prefix="lytra_toy_alignment_") as temp:
        root = Path(temp)
        tokenizer = Tokenizer(models.WordLevel(
            vocab={"[UNK]": 0, "hello": 1, "world": 2, "example": 3},
            unk_token="[UNK]",
        ))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer_path = root / "toy_tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        trusted_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        source = root / "sources.jsonl"
        emitted = root / "emitted.jsonl"
        source.write_text('\n'.join(json.dumps({"text": text}) for text in (
            "hello world", "example"
        )) + '\n', encoding="utf-8")
        producer_rows = [
            {"source_sha256": exact_source_hash("hello world"), "token_ids": [1, 2]},
            {"source_sha256": exact_source_hash("example"), "token_ids": [3]},
        ]
        emitted.write_text('\n'.join(json.dumps(x) for x in producer_rows) + '\n', encoding="utf-8")
        report_path = root / "result.json"
        args = [
            "--sources", str(source), "--emitted", str(emitted),
            "--tokenizer-json", str(tokenizer_path),
            "--expected-tokenizer-sha256", trusted_sha,
            "--vocab-size", "4",
            "--output", str(report_path),
        ]
        if alignment_main(args) != 0:
            raise AssertionError("positive toy alignment CLI failed")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "PASS_SAMPLE_ALIGNMENT" or report["counts"]["matched_records"] != 2:
            raise AssertionError("positive toy result is invalid")
        producer_rows[1]["token_ids"] = [1]
        emitted.write_text('\n'.join(json.dumps(x) for x in producer_rows) + '\n', encoding="utf-8")
        if alignment_main(args) != 2:
            raise AssertionError("negative toy alignment test failed to reject mismatch")
        producer_rows[1]["token_ids"] = [3]
        emitted.write_text('\n'.join(json.dumps(x) for x in producer_rows) + '\n', encoding="utf-8")
        incorrect_pin = args.copy()
        incorrect_pin[incorrect_pin.index("--expected-tokenizer-sha256") + 1] = "0" * 64
        if alignment_main(incorrect_pin) != 1:
            raise AssertionError("tampered tokenizer identity was not rejected")

    print("SYNTHETIC_ALIGNMENT_SMOKE=PASS (positive, ID mismatch, pin mismatch)")


if __name__ == "__main__":
    main()
