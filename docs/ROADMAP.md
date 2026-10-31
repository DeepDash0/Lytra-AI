# Roadmap

## Completed

- [x] Lock Lytra V1 2.512B architecture
- [x] Build and freeze custom 65,536-token byte-level BPE
- [x] Validate exact model parameter count
- [x] 126M final-tokenizer numerical smoke
- [x] exact 2.512B / 4K TPU numerical gate
- [x] full-state checkpoint/resume validation
- [x] 8-device canonical 512-sequence batch benchmark
- [x] real-data packing and contamination pilot

## In progress

- [ ] 17-step real-data TPU streaming + resume gate
- [ ] durable checkpoint-backend restore smoke
- [ ] immutable full production data manifest
- [ ] final exact-token accounting / packing audit
- [ ] final two-stage 200B production config

## Later

- [ ] base pretraining
- [ ] evaluation suite
- [ ] progressive context extension
- [ ] open-weight checkpoint release

## Local support tooling (not production gate completion)

- [x] Add provenance-aware sample dataset audit and unit tests
- [x] Add packed-token trace auditing and synthetic canonical 512x4096 batch smoke
- [x] Add SHA-256 shard integrity manifest create/verify helper
- [x] Build independent sample text-to-token alignment and fail-closed technical preflight
- [x] Add Python regression tests and synthetic tokenizer CLI smoke
- [x] Build 17-step declared-state replay comparator and synthetic checkpoint/resume smoke
- [x] Add replay identity check to local technical preflight (not real TPU completion)
- [x] Add provisional two-stage 200B content-token schedule calculator and tokenizer-slot accounting ledger
- [ ] Approve final two-stage source quotas and measure actual packing utilization before freezing production config
- [ ] Run audits on verified real Lytra tokenizer and approved dataset shards
- [ ] Cross-check actual Lytra-produced sample token IDs against independently re-tokenized source texts
