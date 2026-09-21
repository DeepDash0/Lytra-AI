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
