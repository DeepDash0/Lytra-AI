# Lytra

Lytra is an independent language-model research project focused on training a compact, efficient decoder-only model **from scratch**.

The first target is **Lytra V1**, an English + code focused dense Transformer with exactly **2,512,537,600 parameters** and a custom **65,536-token byte-level BPE tokenizer**.

> Status: active research / pretraining infrastructure validation. No pretrained Lytra V1 weights have been released yet.

## Lytra V1

- 36 decoder layers
- hidden size: 2,048
- 16 query heads / 2 KV heads (GQA)
- head dimension: 128
- SwiGLU intermediate size: 9,216
- RMSNorm, epsilon 1e-6
- RoPE theta: 1,000,000
- tied input/output embeddings
- 65,536-token custom byte-level BPE
- initial pretraining context: 4,096
- planned context extension: 8K → 16K → 32K → 64K → 128K
- training: random initialization / scratch

The exact parameter count can be independently reproduced with:

```bash
python scripts/parameter_count.py
```

## Pretraining target

The current V1 recipe targets a minimum of **200B accepted content tokens**.

Canonical optimizer geometry:

- 512 packed sequences / optimizer update
- 4,096 token slots / sequence
- 2,097,152 token slots / update
- nominal 95,368 optimizer updates
- AdamW
- peak LR: 1.6e-4
- warmup: 954 updates
- cosine decay
- BF16 model/activations

See [docs/PRETRAINING_PLAN.md](docs/PRETRAINING_PLAN.md).

## Current validation status

The exact 2.512B architecture and frozen tokenizer have passed numerical and full-state checkpoint/resume testing on an **8-device TPU** runtime.

A production-batch topology benchmark using the canonical 512-sequence optimizer batch also passed. The best measured candidate was:

- per-device batch: 1
- gradient accumulation: 64
- 8 TPU devices / 1 process
- ~42.703 s / optimizer update after compilation
- ~49.1k token slots / second

These are **infrastructure/numerical results, not capability scores**.

See [docs/NUMERICAL_GATES.md](docs/NUMERICAL_GATES.md).

## Release direction

The repository is public for project transparency and reproducibility work.

The current plan is to release future trained checkpoints as **open weights**. The final weight license has not been selected yet, and no claim is made that the complete training dataset will be redistributable.

No production-pretrained Lytra V1 checkpoint is currently published.

## Repository scope

This public repository intentionally excludes:

- private credentials and account data
- raw training datasets
- large local artifacts
- checkpoints
- provider-specific private deployment files
- local machine paths

Additional training tooling will be published as it is cleaned and made reproducible.

## Project status

Lytra is independently developed and is not affiliated with a company, university, AMD, Google, or OpenAI.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the current engineering milestones.

## Local dataset validation tooling

The repository includes local, auditable validation steps:

- [Dataset sample audit](docs/DATASET_AUDIT.md): provenance flags, exact normalized-text duplicates, held-out SHA-256 overlaps and content-token counts with the frozen tokenizer.
- [Packed-token audit](docs/PACKING_AUDIT.md): verifies emitted token-ID ranges, 4096-slot masks, 512-sequence optimizer updates and consistency with upstream accepted-token counts.
- [Data shard manifest](docs/DATA_MANIFEST.md): deterministic SHA-256 integrity verification for explicitly selected shard files.
- [Independent tokenizer alignment](docs/TOKENIZER_PREFLIGHT.md): re-tokenizes exact raw source samples using a pinned frozen tokenizer and compares producer-side token IDs.
- [Fail-closed preflight](docs/TOKENIZER_PREFLIGHT.md): combines independent technical evidence but never declares production readiness or launches a training run.

Create a private local Python environment and install the pinned CLI dependency with `python -m pip install -r requirements-audit.txt`. Run local tests with `python -m unittest discover -s tests -q`. Exercise synthetic test paths via `python -m scripts.smoke_packing` and `python -m scripts.smoke_alignment`. GitHub Actions reruns the unit tests and synthetic smokes on branch pushes and pull requests.

**Passing the local tests is not a real-data training readiness result.** Tokenizer identity, provider/rights verification, production dataset manifests, real TPU checkpoint restore and streaming/resume validation are still separate requirements.
