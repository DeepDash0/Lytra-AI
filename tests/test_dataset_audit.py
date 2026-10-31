"""Run with python -m unittest discover -s tests -v."""
from pathlib import Path
import json
import tempfile
import unittest

from scripts.dataset_audit import audit, content_hash, read_hashes


class DatasetAuditTests(unittest.TestCase):
    def run_audit(self, rows, heldout=None):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "sample.jsonl"
            p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
            return audit([p], lambda text: len(text.split()), tokenizer_sha256="test", heldout_hashes=heldout)

    def row(self, text="alpha beta", **changes):
        item = dict(text=text, source="corpus-a", license="example-license",
                    origin="human", split="train", approved_for_training=True)
        item.update(changes)
        return item

    def test_accept_dedup_and_rights(self):
        result = self.run_audit([
            self.row(), self.row(), self.row("gamma", approved_for_training=False),
            self.row("delta", license=""), self.row("omega", origin="teacher"),
            self.row("eta theta", origin="teacher", teacher_use_approved=True),
        ])
        c = result["counts"]
        self.assertEqual(c["accepted_records"], 2)
        self.assertEqual(c["accepted_content_tokens"], 4)
        self.assertEqual(c["duplicates"], 1)
        self.assertEqual(c["unapproved_usage"], 1)
        self.assertEqual(c["missing_license"], 1)
        self.assertEqual(c["unapproved_teacher_usage"], 1)
        self.assertEqual(result["by_origin"]["teacher"]["records"], 1)

    def test_heldout_overlap_and_non_train(self):
        result = self.run_audit([
            self.row("eval content"), self.row("training data"),
            self.row("holdout", split="eval"),
        ], heldout={content_hash("eval content")})
        self.assertEqual(result["counts"]["heldout_overlap"], 1)
        self.assertEqual(result["counts"]["non_train_records"], 1)
        self.assertEqual(result["counts"]["accepted_content_tokens"], 2)
        self.assertTrue(result["heldout_hashes_supplied"])

    def test_malformed_schema_and_priority(self):
        result = self.run_audit([{}, self.row("a", approved_for_training=False), self.row("a")])
        self.assertEqual(result["counts"]["invalid_schema"], 1)
        self.assertEqual(result["counts"]["accepted_records"], 1)

    def test_unhashable_origin_is_rejected_without_crash(self):
        result = self.run_audit([self.row("safe", origin=["human"]), self.row("real")])
        self.assertEqual(result["counts"]["invalid_origin"], 1)
        self.assertEqual(result["counts"]["accepted_records"], 1)

    def test_unicode_fingerprint(self):
        self.assertEqual(content_hash("caf\u00e9"), content_hash("cafe\u0301"))

    def test_read_hashes_reject_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "bad.txt"
            p.write_text("not-a-sha\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_hashes(p)


if __name__ == "__main__":
    unittest.main()
