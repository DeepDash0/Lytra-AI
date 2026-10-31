# Lytra V1 — Proposed 200B Content-Token Schedule (not approved)

## Why a slot budget alone is not enough

Lytra V1 targets at least **200,000,000,000 accepted content tokens**. The canonical global optimizer batch is 512 sequences x 4096 positions = **2,097,152 token slots/update**. Padding, EOS/BOS and other special tokens consume slots but must **not** be counted as accepted content tokens.

The old figure of **95,368 optimizer updates** is a theoretical minimum for a single continuous 200B content-token goal at perfect utilization: it provides 200,001,191,936 total slots. It is **not** a guarantee of 200B content tokens at any real-world utilization.

## Provisional two-stage split

`configs/pretraining_200b_proposal.json` proposes the following **illustrative** split. These are not measured or approved quotas:

| Stage | Proposed accepted-content target | Intended mix |
| --- | ---: | --- |
| Base | 190,000,000,000 | High-quality English, code, math, structured knowledge |
| Recency | 10,000,000,000 | Late recency-oriented continuation |

The actual source composition, rights, weightings, recency cutoffs, stage sizes, tokenizer and full dataset manifests are not frozen. This file is explicitly tagged `PROPOSAL_NOT_APPROVED`. The parser rejects an example config that pretends its approval gates have been completed.

## Exactly calculated scenario counts

For stage target `T`, slots/update `S`, and **assumed** content fraction `u`, compute planned optimizer updates as `ceil(T / (S × u))`. The tool uses exact decimal rational arithmetic. Each stage rounds independently to a whole optimizer update, so the sum can be one or more updates greater than `ceil(total/S/u)`.

| Assumed useful slot ratio | Stage 1 updates | Stage 2 updates | Total steps |
| ---: | ---: | ---: | ---: |
| 100% | 90,600 | 4,769 | 95,369 |
| 98% | 92,449 | 4,866 | 97,315 |
| 95% | 95,368 | 5,020 | 100,388 |

The 100% scenario above must not be mistaken for a measured packing result. Historical TPU benchmark **42.703 seconds/update** from earlier numerical performance tests is multiplied by these steps to give rough *benchmark-based arithmetic only*; no compilation, checkpointing, dataset I/O, evaluation, failure, TPU queue time or cost is covered. Actual production throughput may differ substantially.

## Usage

Calculate schedule and learning-rate sample points:

```bash
python -m scripts.training_budget --config configs/pretraining_200b_proposal.json --output artifacts/training_budget_plan.json
```

The peak learning rate is 0.00016, final learning rate 0.000016, with 954 optimizer warmup updates. The example computes one **continuous** cosine decay after warmup, without restarting at stage 2; this policy is provisional until approved.

Verify a producer-exported full optimizer-step ledger, not just a microbatch log:

```bash
python -m scripts.training_budget --config configs/pretraining_200b_proposal.json --ledger /private/training_ledger.jsonl --output artifacts/ledger_audit.json
```

Each JSONL record must have `step` (starting at 1, strictly consecutive even across resumed runs), `stage` (`base` then `recency` only when the prior accepted-content target was reached), `content_tokens`, `special_tokens`, `padding_tokens`, and `config_sha256` (SHA-256 of the exact config file bytes). The three token counters must sum to 2,097,152 for every optimizer update; all counters must be nonnegative integers, and content tokens must be positive. Stage targets are accepted-content totals; crossing a target may overshoot by one batch. A later update cannot claim it is still in a completed stage. Extra updates after both stages have reached their goals are rejected.

Ledger returns `IN_PROGRESS` for correctly structured but incomplete runs and `CONTENT_GOAL_REACHED` for claimed complete stage goals. Neither result proves real optimizer execution, correct tokenizer usage, legal data, or actual training readiness. `FAIL` exits 2, input errors exit 1.

Run local regression tests:

```bash
python -m unittest discover -s tests -q
```

**Remaining before production**: independently pinned real tokenizer, validated and licensed full data, actual packing-utilization measurements, frozen manifest/immutable stage composition, verified 17-step real-data TPU streaming/resume, durable remote checkpoint restore and formally approved training recipe. No training process is launched by these utilities.
