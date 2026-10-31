# Lytra V1 Pretraining Plan

## Target

The current base-pretraining target is **at least 200B accepted content tokens**.

The data pipeline is English-only for V1, with dedicated high-quality web, code, mathematics, structured knowledge, and recency components.

## Optimizer recipe

- AdamW
- peak LR: 1.6e-4
- beta1: 0.9
- beta2: 0.95
- epsilon: 1e-8
- weight decay: 0.1
- global gradient clipping: 1.0
- cosine schedule
- final LR: 1.6e-5
- warmup: 954 optimizer updates
- dropout: 0
- BF16 model weights and activations

## Canonical optimizer batch

- 512 packed sequences / update
- 4,096 token slots / sequence
- 2,097,152 token slots / update
- theoretical minimum 95,368 updates for a **single** 200B accepted-content target at 100% useful slot utilization; real packing efficiency and whole-step per-stage rounding increase this

Accepted content-token accounting and runtime token-slot capacity are tracked separately. The [provisional two-stage budget model](TRAINING_BUDGET.md) proposes a 190B base + 10B recency split strictly for planning. Stage quotas, packing efficiency, exact optimizer steps, and training dates are **not approved or frozen**.

## Data-quality principles

The production pipeline is designed to:

- remain English-only for V1
- use exact final-tokenizer accounting
- deduplicate before final training manifests are frozen
- protect held-out evaluation material from train contamination
- avoid lowering source quality thresholds simply to fill quotas
- keep a dedicated late recency stage

Raw datasets are not stored in this public repository.

## Checkpoints

Production checkpoints are full-state checkpoints: model parameters, optimizer state, step, and data-loader position must all be recoverable.

The current storage design uses rolling local retention plus verified durable archival.
