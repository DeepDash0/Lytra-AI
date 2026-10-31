"""Synthetic-only tests for independent tokenizer/source-output alignment."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.tokenizer_alignment import compare_samples, exact_source_hash


SHA = "a" * 64


def encode(text):
    return [ord(c) for c in text]


class TokenizerAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def compare(self, sources, emitted, **opts):
        s = self.directory / "sources.jsonl"
        e = self.directory / "emitted.jsonl"
        s.write_text("".join(json.dumps(x) + "\n" for x in sources), encoding="utf-8")
        e.write_text("".join(json.dumps(x) + "\n" for x in emitted), encoding="utf-8")
        return compare_samples(
            [s], [e], encode,
            tokenizer_sha256=opts.pop("tokenizer_sha256", SHA),
            expected_tokenizer_sha256=opts.pop("expected_tokenizer_sha256", SHA),
            vocab_size=128, **opts,
        )

    def emitted(self, text, ids=None):
        return {"source_sha256": exact_source_hash(text),
                "token_ids": encode(text) if ids is None else ids}

    def test_exact_matching_ids(self):
        a, b = "hello", "world"
        report = self.compare([{"text": a}, {"text": b}], [self.emitted(b), self.emitted(a)])
        self.assertEqual(report["status"], "PASS_SAMPLE_ALIGNMENT")
        self.assertEqual(report["counts"]["matched_records"], 2)
        self.assertEqual(report["counts"]["matched_content_tokens"], 10)
        self.assertNotIn("hello", json.dumps(report))
        self.assertNotIn(exact_source_hash(a), json.dumps(report))

    def test_token_mismatch_rejected(self):
        report = self.compare([{"text": "hello"}], [self.emitted("hello", [1, 2, 3])])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["counts"]["token_id_mismatch"], 1)

    def test_missing_emitted_source(self):
        report = self.compare([{"text": "a"}, {"text": "b"}], [self.emitted("a")])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["counts"]["missing_emitted_sources"], 1)

    def test_unexpected_source_and_missing(self):
        report = self.compare([{"text": "a"}], [self.emitted("b")])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["counts"]["unexpected_emitted_source"], 1)
        self.assertEqual(report["counts"]["missing_emitted_sources"], 1)

    def test_duplicate_source(self):
        report = self.compare([{"text": "a"}, {"text": "a"}], [self.emitted("a")])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["counts"]["duplicate_source_hash"], 1)

    def test_duplicate_emitted(self):
        report = self.compare([{"text": "a"}], [self.emitted("a"), self.emitted("a")])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["counts"]["duplicate_emitted_source"], 1)

    def test_boolean_token_invalid(self):
        report = self.compare([{"text": "a"}], [self.emitted("a", [True])])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["counts"]["invalid_emitted_schema"], 1)

    def test_tokenizer_sha_mismatch(self):
        report = self.compare([{"text": "a"}], [self.emitted("a")],
                              expected_tokenizer_sha256="b" * 64)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["tokenizer_sha256_matches_expected"])

    def test_max_records_fails_closed(self):
        with self.assertRaises(ValueError):
            self.compare([{"text": "a"}, {"text": "b"}], [self.emitted("a")], max_records=1)

    def test_empty_sources_fail(self):
        report = self.compare([], [])
        self.assertEqual(report["status"], "FAIL")

    def test_raw_whitespace_affects_digest(self):
        self.assertNotEqual(exact_source_hash("foo"), exact_source_hash(" foo "))
        self.assertNotEqual(exact_source_hash("caf\u00e9"), exact_source_hash("cafe\u0301"))

    def test_invalid_tokenizer_sha_rejected(self):
        with self.assertRaises(ValueError):
            self.compare([], [], expected_tokenizer_sha256="not-a-sha")

    def test_invalid_source_schema_is_failure(self):
        report = self.compare([{"text": ["a"]}, {"text": "ok"}], [self.emitted("ok")])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["counts"]["invalid_source_schema"], 1)


if __name__ == "__main__":
    unittest.main()
