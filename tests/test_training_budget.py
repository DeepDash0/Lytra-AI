"""Training budget math and step-ledger accounting use synthetic data only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.training_budget import (
    audit_ledger, ceil_div, learning_rate_at, plan, ratio, read_config,
)


CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "pretraining_200b_proposal.json"
SLOTS = 512 * 4096


class TrainingBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.sha = read_config(CONFIG_FILE)

    def test_exact_geometry_and_theoretical_minimum(self):
        out = plan(self.config, self.sha)
        self.assertEqual(out["slots_per_optimizer_step"], SLOTS)
        self.assertEqual(out["nominal_minimum_optimizer_steps_at_perfect_utilization"], 95368)
        self.assertEqual(out["target_accepted_content_tokens"], 200_000_000_000)
        self.assertEqual(out["status"], "PROPOSAL_ONLY")
        self.assertFalse(out["production_ready"])

    def test_per_stage_rounding_and_utilization(self):
        out = plan(self.config, self.sha)
        for s in out["scenarios"]:
            self.assertEqual(sum(x["estimated_optimizer_updates"] for x in s["stage_estimates"]),
                             s["total_optimizer_updates"])
            self.assertEqual(s["total_allocated_slots"], SLOTS * s["total_optimizer_updates"])
            self.assertGreaterEqual(s["total_optimizer_updates"], 95368)
            self.assertEqual(s["learning_rate_samples"][-1]["learning_rate"], "0.000016")
        self.assertLess(out["scenarios"][0]["total_optimizer_updates"],
                        out["scenarios"][-1]["total_optimizer_updates"])

    def test_warmup_lr(self):
        from decimal import Decimal
        peak = Decimal("0.00016")
        final = Decimal("0.000016")
        self.assertEqual(learning_rate_at(954, 100000, 954, peak, final), "0.00016")
        self.assertEqual(learning_rate_at(100000, 100000, 954, peak, final), "0.000016")
        self.assertEqual(learning_rate_at(1, 100000, 954, peak, final), format(peak / 954, ".12g"))

    def test_invalid_utilization_and_fraction_rounding(self):
        for bad in (0, "0", "-0.1", "1.01", "nan", "inf", "not_a_number"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ratio(bad)
        self.assertEqual(ceil_div(10, 3), 4)

    def test_stage_targets_must_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = copy.deepcopy(self.config)
            config["targets"]["stages"][1]["accepted_content_tokens"] += 1
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_config(path)

    def test_microbatch_geometry_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = copy.deepcopy(self.config)
            config["batch"]["gradient_accumulation"] = 63
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_config(path)

    def test_approval_flags_must_remain_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = copy.deepcopy(self.config)
            config["approval"]["licensed_dataset_verified"] = True
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_config(path)


class SyntheticLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "trace.jsonl"
        self.config = copy.deepcopy(read_config(CONFIG_FILE)[0])
        self.config["targets"]["accepted_content_tokens"] = 7_000_000
        self.config["targets"]["stages"][0]["accepted_content_tokens"] = 4_000_000
        self.config["targets"]["stages"][1]["accepted_content_tokens"] = 3_000_000
        self.sha = "f" * 64

    def row(self, step, stage, content=2_000_000, special=64):
        return {
            "step": step, "stage": stage,
            "content_tokens": content,
            "special_tokens": special,
            "padding_tokens": SLOTS - content - special,
            "config_sha256": self.sha,
        }

    def check(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return audit_ledger(self.path, self.config, self.sha)

    def good_rows(self):
        return [self.row(1, "base"), self.row(2, "base"),
                self.row(3, "recency"), self.row(4, "recency")]

    def test_complete_accepted_content_goal(self):
        out = self.check(self.good_rows())
        self.assertEqual(out["status"], "CONTENT_GOAL_REACHED")
        self.assertEqual(out["counts"]["verified_updates"], 4)
        self.assertEqual(out["counts"]["token_slots"], 4 * SLOTS)
        self.assertEqual(out["counts"]["content_tokens"], 8_000_000)
        self.assertEqual(out["stage_progress"]["base"]["optimizer_updates"], 2)
        self.assertFalse(out["training_started"])

    def test_partial_ledger_is_not_goal_reached(self):
        out = self.check(self.good_rows()[:2])
        self.assertEqual(out["status"], "IN_PROGRESS")
        self.assertEqual(out["stage_progress"]["recency"]["optimizer_updates"], 0)

    def test_duplicate_or_gap_step_fails(self):
        rows = self.good_rows()
        rows[2]["step"] = 2
        self.assertEqual(self.check(rows)["error_codes"], ["STEP_ORDER_OR_REPLAY_MISMATCH"])

    def test_out_of_order_stage_fails(self):
        rows = self.good_rows()
        rows[1]["stage"] = "recency"
        self.assertEqual(self.check(rows)["error_codes"], ["STAGE_ORDER_MISMATCH"])

    def test_extra_update_after_both_stages_fails(self):
        rows = self.good_rows() + [self.row(5, "recency")]
        self.assertEqual(self.check(rows)["error_codes"], ["EXTRA_UPDATE_AFTER_TARGET"])

    def test_inflated_content_count_fails(self):
        rows = self.good_rows()
        rows[0]["content_tokens"] = SLOTS + 1
        self.assertEqual(self.check(rows)["error_codes"], ["TOKEN_SLOT_MISMATCH"])

    def test_declared_optimizer_configuration_mismatch_fails(self):
        rows = self.good_rows()
        rows[0]["config_sha256"] = "a" * 64
        self.assertEqual(self.check(rows)["error_codes"], ["CONFIG_HASH_MISMATCH"])

    def test_boolean_token_count_rejected(self):
        rows = self.good_rows()
        rows[0]["special_tokens"] = True
        self.assertEqual(self.check(rows)["error_codes"], ["INVALID_TOKEN_COUNTS"])

    def test_empty_ledger_fails(self):
        self.assertEqual(self.check([])["error_codes"], ["EMPTY_LEDGER"])

    def test_row_limit_fails_closed(self):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in self.good_rows()), encoding="utf-8")
        with self.assertRaises(ValueError):
            audit_ledger(self.path, self.config, self.sha, max_rows=2)

    def test_malformed_json_is_fail_not_success(self):
        self.path.write_text("{ this is not json\n", encoding="utf-8")
        self.assertEqual(audit_ledger(self.path, self.config, self.sha)["error_codes"], ["INVALID_JSON"])


if __name__ == "__main__":
    unittest.main()
