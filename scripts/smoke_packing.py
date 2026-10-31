#!/usr/bin/env python3
"""Generate a synthetic 512x4096 one-update trace and validate accounting.

No real tokenizer, dataset, model or remote runtime is used.
"""
import json
from pathlib import Path
import tempfile

from scripts.packing_audit import audit_packed


def main() -> None:
    sequence = {
        "input_ids": [101] * 4000 + [0] * 96,
        "attention_mask": [1] * 4000 + [0] * 96,
        "special_tokens_mask": [0] * 3999 + [1] + [0] * 96,
    }
    with tempfile.TemporaryDirectory(prefix="lytra_packing_smoke_") as tmp:
        trace = Path(tmp) / "trace.jsonl"
        with trace.open("w", encoding="utf-8") as out:
            for seq_index in range(512):
                out.write(json.dumps({"step": 0, "seq_index": seq_index, **sequence}, separators=(",", ":")) + "\n")
        result = audit_packed([trace], sequence_length=4096,
                              sequences_per_update=512, vocab_size=65536)
    counts = result["counts"]
    assert result["status"] == "STRUCTURE_ONLY_PASS", result["errors"]
    assert counts["complete_optimizer_updates"] == 1
    assert counts["token_slots"] == 2097152
    assert counts["content_tokens"] == 3999 * 512
    assert counts["special_tokens"] == 512
    assert counts["padding_tokens"] == 96 * 512
    print(json.dumps({"synthetic_only": True, "status": result["status"], "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
