"""Regression tests for emitted packed batches; never use private training data."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.packing_audit import audit_packed


class PackingAuditTests(unittest.TestCase):
    def sequence(self, step: int, index: int, ids=None, attention=None, special=None):
        return {
            "step": step,
            "seq_index": index,
            "input_ids": [1] * 8 if ids is None else ids,
            "attention_mask": [1] * 6 + [0] * 2 if attention is None else attention,
            "special_tokens_mask": [0] * 5 + [1] + [0] * 2 if special is None else special,
        }

    def check(self, records, **options):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "trace.jsonl"
            file.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            return audit_packed(
                [file], sequence_length=8, sequences_per_update=2, vocab_size=32, **options
            )

    def test_valid_two_updates_and_slot_conservation(self):
        rows = [self.sequence(step, index) for step in range(2) for index in range(2)]
        result = self.check(rows)
        self.assertEqual(result["status"], "STRUCTURE_ONLY_PASS")
        self.assertEqual(result["counts"]["complete_optimizer_updates"], 2)
        self.assertEqual(result["counts"]["token_slots"], 32)
        self.assertEqual(result["counts"]["content_tokens"], 20)
        self.assertEqual(result["counts"]["special_tokens"], 4)
        self.assertEqual(result["counts"]["padding_tokens"], 8)

    def test_matching_upstream_trace(self):
        rows = [self.sequence(0, i) for i in range(2)]
        sha = "a" * 64
        result = self.check(rows, audit_report={
            "counts": {"accepted_content_tokens": 10}, "tokenizer_sha256": sha
        }, tokenizer_sha256=sha)
        self.assertEqual(result["status"], "PASS_DECLARED_TRACE")

    def test_upstream_token_mismatch(self):
        rows = [self.sequence(0, i) for i in range(2)]
        sha = "a" * 64
        result = self.check(rows, audit_report={
            "counts": {"accepted_content_tokens": 11}, "tokenizer_sha256": sha
        }, tokenizer_sha256=sha)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["content_tokens_match_upstream_audit"])

    def test_tokenizer_mismatch(self):
        rows = [self.sequence(0, i) for i in range(2)]
        result = self.check(rows, audit_report={
            "counts": {"accepted_content_tokens": 10}, "tokenizer_sha256": "a"*64
        }, tokenizer_sha256="b"*64)
        self.assertFalse(result["tokenizer_sha256_matches_upstream_audit"])
        self.assertEqual(result["status"], "FAIL")

    def test_incomplete_batch_fails(self):
        result = self.check([self.sequence(0, 0)])
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["counts"].get("complete_optimizer_updates", 0), 0)

    def test_invalid_special_mask(self):
        record = self.sequence(0, 0)
        record["special_tokens_mask"][-1] = 1
        result = self.check([record, self.sequence(0, 1)])
        self.assertEqual(result["status"], "FAIL")

    def test_interior_padding_rejected(self):
        record = self.sequence(0, 0)
        record["attention_mask"] = [1, 1, 0, 1, 1, 1, 0, 0]
        result = self.check([record, self.sequence(0, 1)])
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("interior padding" in e for e in result["errors"]))

    def test_vocab_overflow(self):
        record = self.sequence(0, 0)
        record["input_ids"][0] = 32
        result = self.check([record, self.sequence(0, 1)])
        self.assertEqual(result["status"], "FAIL")

    def test_duplicate_indices_fails(self):
        result = self.check([self.sequence(0, 0), self.sequence(0, 0)])
        self.assertEqual(result["status"], "FAIL")

    def test_bool_token_id_rejected(self):
        record = self.sequence(0, 0)
        record["input_ids"][0] = True
        result = self.check([record, self.sequence(0, 1)])
        self.assertEqual(result["status"], "FAIL")

    def test_empty_trace_fails(self):
        result = self.check([])
        self.assertEqual(result["status"], "FAIL")

    def test_noncontiguous_steps_fail(self):
        result = self.check([self.sequence(2, 0), self.sequence(2, 1),
                             self.sequence(4, 0), self.sequence(4, 1)])
        self.assertEqual(result["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
