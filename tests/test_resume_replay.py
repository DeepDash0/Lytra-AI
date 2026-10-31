"""Synthetic-only regression tests for Lytra checkpoint/replay verifier."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.resume_replay import verify_replay, main as resume_main
from scripts.smoke_resume import make_fixture


class ResumeReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = make_fixture(Path(self.temp.name))

    def run_gate(self, **overrides):
        return verify_replay(**{**self.fixture, **overrides})

    def change_trace(self, key, index, update):
        path = self.fixture[key]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        update(rows[index])
        path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")

    def change_checkpoint_meta(self, update):
        path = self.fixture["checkpoint_sidecar"]
        data = json.loads(path.read_text(encoding="utf-8"))
        update(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_cli_report_success(self):
        report_path = Path(self.temp.name) / "report.json"
        args = []
        for name in ("baseline", "before", "after", "checkpoint", "checkpoint_sidecar"):
            args += ["--" + name.replace("_", "-"), str(self.fixture[name])]
        for name in ("dataset", "tokenizer", "config"):
            args += ["--expected-" + name + "-sha256", self.fixture["expected_" + name + "_sha256"]]
        args += ["--output", str(report_path)]
        self.assertEqual(resume_main(args), 0)
        result = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "PASS_DECLARED_STRICT_REPLAY")

    def test_synthetic_17_step_strict_pass(self):
        result = self.run_gate()
        self.assertEqual(result["status"], "PASS_DECLARED_STRICT_REPLAY")
        self.assertTrue(result["checkpoint_bytes_verified"])
        self.assertFalse(result["real_tpu_gate_completed"])

    def test_recovery_optimizer_state_differs(self):
        self.change_trace("after", 0, lambda row: row.update(optimizer_sha256="1" * 64))
        self.assertIn("STATE_MISMATCH", self.run_gate()["error_codes"])

    def test_model_state_differs(self):
        self.change_trace("after", 1, lambda row: row.update(model_sha256="2" * 64))
        self.assertIn("STATE_MISMATCH", self.run_gate()["error_codes"])

    def test_batch_differs(self):
        self.change_trace("after", 0, lambda row: row.update(batch_sha256="3" * 64))
        self.assertIn("BATCH_MISMATCH", self.run_gate()["error_codes"])

    def test_rng_differs(self):
        self.change_trace("after", 0, lambda row: row.update(rng_sha256="4" * 64))
        self.assertIn("STATE_MISMATCH", self.run_gate()["error_codes"])

    def test_config_pin_differs(self):
        result = self.run_gate(expected_config_sha256="5" * 64)
        self.assertIn("RUN_IDENTITY_MISMATCH", result["error_codes"])

    def test_wrong_cursor(self):
        self.change_trace("after", 0, lambda row: row.update(data_cursor=999))
        with self.assertRaises(ValueError):
            self.run_gate()

    def test_checkpoint_corrupt_bytes(self):
        p = self.fixture["checkpoint"]
        p.write_bytes(p.read_bytes() + b"tamper")
        self.assertIn("CHECKPOINT_BYTES_MISMATCH", self.run_gate()["error_codes"])

    def test_checkpoint_sidecar_tampered(self):
        self.change_checkpoint_meta(lambda row: row.update(optimizer_sha256="6" * 64))
        self.assertIn("CHECKPOINT_STATE_MISMATCH", self.run_gate()["error_codes"])

    def test_checkpoint_step_wrong(self):
        self.change_checkpoint_meta(lambda row: row.update(checkpoint_step=7))
        self.assertIn("CHECKPOINT_POSITION_MISMATCH", self.run_gate()["error_codes"])

    def test_missing_checkpoint_key(self):
        self.change_checkpoint_meta(lambda row: row.pop("rng_sha256"))
        result = self.run_gate()
        self.assertIn("INCOMPLETE_CHECKPOINT_METADATA", result["error_codes"])

    def test_loss_mismatch(self):
        self.change_trace("after", 0, lambda row: row.update(loss=777.0))
        self.assertIn("LOSS_MISMATCH", self.run_gate()["error_codes"])

    def test_missing_resume_marker(self):
        self.change_trace("after", 0, lambda row: row.pop("resumed_from_step"))
        self.assertIn("RESUME_MARKER_MISSING", self.run_gate()["error_codes"])

    def test_step_gap_raises(self):
        self.change_trace("after", 0, lambda row: row.update(step=16))
        with self.assertRaises(ValueError):
            self.run_gate()

    def test_missing_row_raises(self):
        p = self.fixture["after"]
        p.write_text("".join(p.read_text(encoding="utf-8").splitlines(True)[:-1]), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_gate()

    def test_nonfinite_loss_raises(self):
        self.change_trace("after", 0, lambda row: row.update(loss=float("nan")))
        with self.assertRaises(ValueError):
            self.run_gate()

    def test_invalid_sha_raises(self):
        self.change_trace("before", 0, lambda row: row.update(model_sha256="bad"))
        with self.assertRaises(ValueError):
            self.run_gate()

    def test_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            self.run_gate(split_step=17)

    def test_invalid_tolerance_raises(self):
        with self.assertRaises(ValueError):
            self.run_gate(loss_atol=-1)

    def test_loss_tolerance_only_affects_loss(self):
        self.change_trace("after", 0, lambda row: row.update(loss=row["loss"] + 0.0001))
        self.assertEqual(self.run_gate(loss_atol=0.001)["status"], "PASS_DECLARED_STRICT_REPLAY")


if __name__ == "__main__":
    unittest.main()
