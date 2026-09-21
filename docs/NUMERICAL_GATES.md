# Numerical / Runtime Gates

These results validate numerical stability and infrastructure. They are **not model-quality or capability benchmarks**.

## Final-tokenizer numerical gate

Status: **PASS**

Exact Lytra V1:
- parameters: 2,512,537,600
- sequence length: 4,096
- TPU topology: 8 devices, 1 process
- full-state checkpoint save/restore: PASS
- contiguous training metrics after resume: PASS

Observed 2.5B gate:
- train loss: 11.5735 → 10.6286
- train minimum: 10.5990
- gradient norm: approximately 0.992–1.0

The gate used a short numerical run from random initialization; it is not a pretrained checkpoint.

## Canonical optimizer-batch benchmark

All candidates preserved 512 global sequences / optimizer update.

| Microbatch/device | Gradient accumulation | Seconds/update |
|---:|---:|---:|
| 1 | 64 | ~42.703 |
| 2 | 32 | ~42.742 |
| 4 | 16 | ~43.746 |
| 8 | 8 | ~45.851 |

The best measured topology was batch/device 1 + GA64, approximately **49.1k token slots/s**.

A separate real-data streaming/resume pilot is the next runtime gate before final production configuration is frozen.
