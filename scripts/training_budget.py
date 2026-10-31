#!/usr/bin/env python3
"""Plan Lytra accepted-content token budgets and audit real optimizer-step ledgers.

This is a planning/accounting utility, NOT a training launcher or data approval.
All stage allocations and utilization percentages in the example config are
hypothetical until approved and measured against real packed datasets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys


def positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def ratio(value: object) -> Fraction:
    if not isinstance(value, str):
        raise ValueError("utilization must be an exact decimal string")
    try:
        number = Fraction(Decimal(value))
    except (InvalidOperation, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("utilization must be a finite decimal") from exc
    if not 0 < number <= 1:
        raise ValueError("packing utilization must be greater than zero and at most 1")
    return number


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def read_config(path: Path) -> tuple[dict, str]:
    content = path.read_bytes()
    config = json.loads(content)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported budget configuration")
    if config.get("status") != "PROPOSAL_NOT_APPROVED":
        raise ValueError("budget is an unapproved planning proposal, not a finalized training schedule")
    targets = config.get("targets")
    batch = config.get("batch")
    optimizer = config.get("optimizer")
    if not all(isinstance(v, dict) for v in (targets, batch, optimizer)):
        raise ValueError("missing targets, batch or optimizer")
    if not positive_int(targets.get("accepted_content_tokens")):
        raise ValueError("invalid total content-token target")
    stages = targets.get("stages")
    if not isinstance(stages, list) or len(stages) != 2:
        raise ValueError("exactly two proposed stages are required")
    if any(not isinstance(s, dict) or not isinstance(s.get("name"), str) or
           not s["name"] or not positive_int(s.get("accepted_content_tokens"))
           for s in stages):
        raise ValueError("invalid stage specification")
    if len({s["name"] for s in stages}) != 2:
        raise ValueError("stage names must be unique")
    if sum(s["accepted_content_tokens"] for s in stages) != targets["accepted_content_tokens"]:
        raise ValueError("stage content targets must equal total target")
    fields = ("devices", "microbatch_per_device", "gradient_accumulation",
              "sequences_per_optimizer_step", "sequence_length")
    if any(not positive_int(batch.get(f)) for f in fields):
        raise ValueError("invalid optimizer batch geometry")
    if batch["devices"] * batch["microbatch_per_device"] * batch["gradient_accumulation"] != batch["sequences_per_optimizer_step"]:
        raise ValueError("devices * microbatch * accumulation must equal global sequence count")
    if batch["sequence_length"] != 4096 or batch["sequences_per_optimizer_step"] != 512:
        raise ValueError("Lytra V1 canonical geometry must be 512 sequences x 4096 slots")
    if (optimizer.get("schedule") != "continuous_cosine_after_warmup" or
        optimizer.get("name") != "AdamW" or
        not positive_int(optimizer.get("warmup_optimizer_steps"))):
        raise ValueError("optimizer recipe not recognized")
    peak = ratio_lr(optimizer.get("peak_learning_rate"))
    final = ratio_lr(optimizer.get("final_learning_rate"))
    if final > peak:
        raise ValueError("final learning rate exceeds peak")
    utils = config.get("example_packing_utilizations")
    if not isinstance(utils, list) or not utils:
        raise ValueError("utilization scenarios required")
    seen: set[str] = set()
    for value in utils:
        _ = ratio(value)
        if value in seen:
            raise ValueError("duplicate utilization scenario")
        seen.add(value)
    bench = config.get("historical_benchmark_seconds_per_step")
    if not isinstance(bench, str):
        raise ValueError("historical benchmark must be a decimal string")
    if Decimal(bench) <= 0 or not Decimal(bench).is_finite():
        raise ValueError("invalid historical seconds per step")
    approval = config.get("approval")
    if not isinstance(approval, dict) or not approval or any(type(v) is not bool for v in approval.values()):
        raise ValueError("explicit unverified approval flags required")
    if any(approval.values()):
        raise ValueError("example proposal cannot assert verified training approvals")
    return config, hashlib.sha256(content).hexdigest()


def ratio_lr(value: object) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("learning rates must be decimal strings")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid learning rate") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError("learning rates must be positive and finite")
    return result


def learning_rate_at(step: int, total: int, warmup: int, peak: Decimal, end: Decimal) -> str:
    if not 1 <= step <= total or not 1 <= warmup < total:
        raise ValueError("learning rate schedule requires 0 < warmup < total steps")
    if step <= warmup:
        lr = peak * Decimal(step) / Decimal(warmup)
    else:
        fraction = (step - warmup) / (total - warmup)
        lr = end + (peak - end) * Decimal(str((1 + math.cos(math.pi * fraction)) / 2))
    return format(lr.normalize(), ".12g")


def plan(config: dict, sha: str) -> dict:
    batch = config["batch"]
    stages = config["targets"]["stages"]
    total_target = config["targets"]["accepted_content_tokens"]
    slots_per_step = batch["sequences_per_optimizer_step"] * batch["sequence_length"]
    opt = config["optimizer"]
    warmup = opt["warmup_optimizer_steps"]
    peak = ratio_lr(opt["peak_learning_rate"])
    final = ratio_lr(opt["final_learning_rate"])
    benchmark = Decimal(config["historical_benchmark_seconds_per_step"])
    scenarios = []
    for scenario in config["example_packing_utilizations"]:
        util = ratio(scenario)
        per_stage = []
        cumulative = 0
        for stage in stages:
            target = stage["accepted_content_tokens"]
            updates = ceil_div(target * util.denominator, slots_per_step * util.numerator)
            cumulative += updates
            per_stage.append({
                "name": stage["name"],
                "target_content_tokens": target,
                "estimated_optimizer_updates": updates,
                "cumulative_optimizer_updates": cumulative,
                "allocated_token_slots": updates * slots_per_step,
            })
        if cumulative <= warmup:
            raise ValueError("warmup must finish before end of schedule")
        sampling_steps = sorted(set((1, warmup, per_stage[0]["cumulative_optimizer_updates"], cumulative)))
        scenarios.append({
            "assumed_content_fraction": scenario,
            "stage_estimates": per_stage,
            "total_optimizer_updates": cumulative,
            "total_allocated_slots": cumulative * slots_per_step,
            "historical_runtime_hours_excluding_overhead": format(benchmark * Decimal(cumulative) / Decimal(3600), ".6f"),
            "learning_rate_samples": [
                {"optimizer_step": step, "learning_rate": learning_rate_at(step, cumulative, warmup, peak, final)}
                for step in sampling_steps
            ],
        })
    return {
        "schema_version": 1,
        "status": "PROPOSAL_ONLY",
        "config_sha256": sha,
        "target_accepted_content_tokens": total_target,
        "nominal_minimum_optimizer_steps_at_perfect_utilization": ceil_div(total_target, slots_per_step),
        "slots_per_optimizer_step": slots_per_step,
        "warmup_optimizer_steps": warmup,
        "scenarios": scenarios,
        "training_started": False,
        "production_ready": False,
        "limitations": [
            "Per-stage allocation and utilization scenarios are proposals; none are measured data.",
            "Optimizer step estimates are independently rounded up per stage; one batch may overshoot a target.",
            "Runtime estimate multiplies a historic TPU benchmark and excludes compilation, IO, evaluation, checkpointing, and outages.",
            "No final tokenizer, rights verification, real packed-data utilization, TPU resume gate, or durable checkpoint restore is inferred.",
        ],
    }


def audit_ledger(path: Path, config: dict, config_sha: str, *, max_rows: int = 1_000_000) -> dict:
    if not positive_int(max_rows):
        raise ValueError("max_rows must be positive")
    slots = config["batch"]["sequences_per_optimizer_step"] * config["batch"]["sequence_length"]
    stages = config["targets"]["stages"]
    totals = defaultdict(int)
    stage_counts: dict[str, dict] = {
        s["name"]: {"accepted_content_tokens": 0, "token_slots": 0, "optimizer_updates": 0}
        for s in stages
    }
    stage_idx = 0
    step_number = 0
    error_codes: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            step_number += 1
            if step_number > max_rows:
                raise ValueError("ledger exceeds maximum row count")
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                error_codes.append("INVALID_JSON")
                break
            if not isinstance(item, dict):
                error_codes.append("INVALID_ROW")
                break
            if stage_idx >= len(stages):
                error_codes.append("EXTRA_UPDATE_AFTER_TARGET")
                break
            expected_stage = stages[stage_idx]["name"]
            if item.get("config_sha256") != config_sha:
                error_codes.append("CONFIG_HASH_MISMATCH")
                break
            if type(item.get("step")) is not int or item["step"] != step_number:
                error_codes.append("STEP_ORDER_OR_REPLAY_MISMATCH")
                break
            if item.get("stage") != expected_stage:
                error_codes.append("STAGE_ORDER_MISMATCH")
                break
            fields = ("content_tokens", "special_tokens", "padding_tokens")
            if any(type(item.get(f)) is not int or item[f] < 0 for f in fields):
                error_codes.append("INVALID_TOKEN_COUNTS")
                break
            content = item["content_tokens"]
            if content == 0 or sum(item[f] for f in fields) != slots:
                error_codes.append("TOKEN_SLOT_MISMATCH")
                break
            stage_counts[expected_stage]["accepted_content_tokens"] += content
            stage_counts[expected_stage]["token_slots"] += slots
            stage_counts[expected_stage]["optimizer_updates"] += 1
            totals["content_tokens"] += content
            totals["special_tokens"] += item["special_tokens"]
            totals["padding_tokens"] += item["padding_tokens"]
            totals["token_slots"] += slots
            totals["verified_updates"] += 1
            if stage_counts[expected_stage]["accepted_content_tokens"] >= stages[stage_idx]["accepted_content_tokens"]:
                stage_idx += 1
    if step_number == 0:
        error_codes.append("EMPTY_LEDGER")
    done = stage_idx == len(stages)
    return {
        "schema_version": 1,
        "status": "FAIL" if error_codes else ("CONTENT_GOAL_REACHED" if done else "IN_PROGRESS"),
        "config_sha256": config_sha,
        "expected_content_tokens": config["targets"]["accepted_content_tokens"],
        "counts": dict(sorted(totals.items())),
        "stage_progress": stage_counts,
        "error_codes": error_codes,
        "training_started": False,
        "production_ready": False,
        "limitations": [
            "Ledger entries are producer declarations, not independent proof of tokenization, uniqueness or optimizer execution.",
            "An interrupted run must export the continuous full-step ledger without gaps or replayed updates.",
            "CONTENT_GOAL_REACHED indicates only declared content-token counts; it is not training or release approval.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pretraining_200b_proposal.json"))
    parser.add_argument("--ledger", type=Path, help="Optional training ledger JSONL to audit instead of planning")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        config, sha = read_config(args.config)
        result = audit_ledger(args.ledger, config, sha) if args.ledger else plan(config, sha)
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
        return 2 if result["status"] == "FAIL" else 0
    except (OSError, ValueError, TypeError, ZeroDivisionError) as exc:
        print(f"training budget error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
