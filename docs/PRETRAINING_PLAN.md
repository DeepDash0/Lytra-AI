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
- nominal 95,368 updates for the 200B schedule

Accepted content-token accounting and runtime token-slot capacity are tracked separately.

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
