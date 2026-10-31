"""Test the local file-manifest integrity checks with synthetic content."""
from pathlib import Path
import tempfile
import unittest

from scripts.data_manifest import create_manifest, verify_manifest


class FileManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "a.jsonl").write_text('{"text":"x"}\n', encoding="utf-8")
        (self.root / "b.jsonl").write_text('{"text":"y"}\n', encoding="utf-8")

    def test_create_and_verify(self):
        m = create_manifest(self.root, ["b.jsonl", "a.jsonl"])
        self.assertEqual([f["path"] for f in m["files"]], ["a.jsonl", "b.jsonl"])
        self.assertEqual(verify_manifest(self.root, m)["status"], "PASS")

    def test_change_is_detected(self):
        m = create_manifest(self.root, ["a.jsonl"])
        (self.root / "a.jsonl").write_text("modified", encoding="utf-8")
        self.assertEqual(verify_manifest(self.root, m)["status"], "FAIL")

    def test_manifest_tamper_is_detected(self):
        m = create_manifest(self.root, ["a.jsonl"])
        m["files"][0]["size_bytes"] = 123456
        self.assertEqual(verify_manifest(self.root, m)["status"], "FAIL")

    def test_empty_duplicate_and_escape_rejected(self):
        with self.assertRaises(ValueError):
            create_manifest(self.root, [])
        with self.assertRaises(ValueError):
            create_manifest(self.root, ["a.jsonl", "a.jsonl"])
        with self.assertRaises(ValueError):
            create_manifest(self.root, ["../outside.dat"])

    def test_missing_file_rejected(self):
        m = create_manifest(self.root, ["a.jsonl"])
        (self.root / "a.jsonl").unlink()
        self.assertEqual(verify_manifest(self.root, m)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
