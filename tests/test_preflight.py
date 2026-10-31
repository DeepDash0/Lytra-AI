"""Fail-closed technical preflight regression tests (synthetic evidence only)."""
import copy
import unittest

from scripts.preflight import evaluate_preflight


PIN = "c" * 64


def evidence():
    dataset = {
        "schema_version": 1, "mode": "exact_content_token_count_no_packing",
        "tokenizer_sha256": PIN, "heldout_hashes_supplied": True,
        "heldout_hashes_count": 100,
        "counts": {"accepted_content_tokens": 2047488},
    }
    packing = {
        "schema_version": 1, "status": "PASS_DECLARED_TRACE",
        "sequence_length": 4096, "sequences_per_update": 512, "vocab_size": 65536,
        "tokenizer_sha256_declared": PIN,
        "tokenizer_sha256_matches_upstream_audit": True,
        "content_tokens_match_upstream_audit": True,
        "counts": {"complete_optimizer_updates": 1, "content_tokens": 2047488,
                   "special_tokens": 512, "padding_tokens": 49152,
                   "token_slots": 2097152},
        "errors": [],
    }
    alignment = {
        "schema_version": 1, "status": "PASS_SAMPLE_ALIGNMENT",
        "vocab_size": 65536, "tokenizer_sha256": PIN,
        "tokenizer_sha256_matches_expected": True,
        "counts": {"matched_records": 30, "valid_source_rows": 30},
        "error_codes": [],
    }
    manifest_sha = "d" * 64
    replay = {
        "schema_version": 1, "status": "PASS_DECLARED_STRICT_REPLAY",
        "expected_steps": 17, "split_step": 8, "steps_compared": 17,
        "checkpoint_bytes_verified": True, "real_tpu_gate_completed": False,
        "training_started": False, "error_codes": [],
        "tokenizer_sha256": PIN, "dataset_manifest_sha256": manifest_sha,
        "config_sha256": "e" * 64,
    }
    return dict(
        dataset=dataset, packing=packing, alignment=alignment, replay=replay,
        manifest_check={"status": "PASS", "file_count": 2, "entries_sha256": manifest_sha},
        pinned_tokenizer_sha256=PIN, min_content_tokens=100000,
    )


class PreflightTests(unittest.TestCase):
    def test_replay_evidence_missing_blocks(self):
        x = copy.deepcopy(evidence())
        x["replay"] = None
        self.assertEqual(evaluate_preflight(**x)["checks"]["declared_resume_replay"]["status"], "BLOCKED")

    def test_replay_manifest_identity_mismatch_blocks(self):
        x = copy.deepcopy(evidence())
        x["replay"]["dataset_manifest_sha256"] = "a" * 64
        self.assertEqual(evaluate_preflight(**x)["checks"]["declared_resume_replay"]["status"], "BLOCKED")

    def test_replay_missing_checkpoint_bytes_blocks(self):
        x = copy.deepcopy(evidence())
        x["replay"]["checkpoint_bytes_verified"] = False
        self.assertEqual(evaluate_preflight(**x)["checks"]["declared_resume_replay"]["status"], "BLOCKED")

    def test_no_artifacts_is_blocked(self):
        result = evaluate_preflight()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["production_ready"])
        self.assertFalse(result["training_started"])

    def test_synthetic_evidence_can_pass_technical_only(self):
        result = evaluate_preflight(**evidence())
        self.assertEqual(result["status"], "PASS_TECHNICAL_ONLY")
        self.assertFalse(result["production_ready"])
        self.assertEqual(result["manual_rights_review"], "NOT_ATTESTED")

    def test_tokenizer_hash_mismatch_blocked(self):
        x = copy.deepcopy(evidence())
        x["alignment"]["tokenizer_sha256"] = "b" * 64
        self.assertEqual(evaluate_preflight(**x)["checks"]["independent_alignment"]["status"], "BLOCKED")

    def test_no_heldout_evidence_blocked(self):
        x = copy.deepcopy(evidence())
        x["dataset"]["heldout_hashes_supplied"] = False
        self.assertEqual(evaluate_preflight(**x)["checks"]["dataset_audit"]["status"], "BLOCKED")

    def test_bad_manifest_blocked(self):
        x = copy.deepcopy(evidence())
        x["manifest_check"] = {"status": "FAIL", "file_count": 2}
        self.assertEqual(evaluate_preflight(**x)["checks"]["shard_integrity"]["status"], "BLOCKED")

    def test_unclean_content_counts_blocked(self):
        x = copy.deepcopy(evidence())
        x["packing"]["counts"]["content_tokens"] += 1
        self.assertEqual(evaluate_preflight(**x)["checks"]["packed_batch"]["status"], "BLOCKED")

    def test_noncanonical_sequence_length_blocked(self):
        x = copy.deepcopy(evidence())
        x["packing"]["sequence_length"] = 2048
        self.assertEqual(evaluate_preflight(**x)["checks"]["packed_batch"]["status"], "BLOCKED")

    def test_dataset_contamination_blocked(self):
        x = copy.deepcopy(evidence())
        x["dataset"]["counts"]["heldout_overlap"] = 1
        self.assertEqual(evaluate_preflight(**x)["checks"]["dataset_audit"]["status"], "BLOCKED")

    def test_small_token_count_blocked(self):
        x = copy.deepcopy(evidence())
        x["min_content_tokens"] = 3000000
        self.assertEqual(evaluate_preflight(**x)["checks"]["dataset_audit"]["status"], "BLOCKED")

    def test_malformed_positive_token_minimum(self):
        with self.assertRaises(ValueError):
            evaluate_preflight(min_content_tokens=0)

    def test_missing_alignment_blocked(self):
        x = copy.deepcopy(evidence())
        x["alignment"] = None
        self.assertEqual(evaluate_preflight(**x)["checks"]["independent_alignment"]["status"], "BLOCKED")

    def test_only_structure_pack_is_blocked(self):
        x = copy.deepcopy(evidence())
        x["packing"]["status"] = "STRUCTURE_ONLY_PASS"
        self.assertEqual(evaluate_preflight(**x)["checks"]["packed_batch"]["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
