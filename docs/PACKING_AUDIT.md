# Lytra V1 packed-sequence audit (engineering gate v0)

This tool validates emitted packed-token traces, not raw text. Canonical Lytra V1 geometry is 4096 token slots per sequence and 512 sequences per optimizer update. The tool streams line-by-line and emits only aggregate counters and structural errors.

## Trace contract

One UTF-8 JSON object per line, containing: step, seq_index, input_ids, attention_mask, special_tokens_mask.

- step is a nonnegative optimizer-update number. Rows must arrive in consecutive optimizer steps with sequence indices 0..511 for each update.
- input_ids must be integer token IDs in the range 0..65535. Booleans are rejected.
- attention_mask=1 means non-padding; special_tokens_mask=1 denotes BOS/EOS or another masked special. A special token in padding is invalid.
- Content = attention 1 and special 0; special = attention 1 and special 1; padding = attention 0 and special 0. Every position belongs to exactly one category.
- Both masks must represent the actual packed trace, including specials and final padding. Never fabricate them after the fact.
- Sharded traces must be passed in logical production order.

## Usage

Run regression checks:

    py -3 -m unittest discover -s tests -q

Inspect locally:

    py -3 scripts/packing_audit.py packed_trace.jsonl --sequence-length 4096 --sequences-per-update 512 --vocab-size 65536 --output packing_report.json

Cross-check against upstream audit for exactly the same content-token subset:

    py -3 scripts/packing_audit.py packed_trace.jsonl --audit-report audit_report.json --tokenizer-sha256 YOUR_VERIFIED_SHA256 --output packing_report.json

If --audit-report is supplied, every accepted upstream content token must occur exactly once in the packed trace (no truncation, duplication or partial epoch). For a sample, produce an upstream audit report for precisely that sample.

STRUCTURE_ONLY_PASS means consistent shape, IDs, masks and batch geometry. PASS_DECLARED_TRACE additionally means content-token totals and declared tokenizer SHA match the upstream audit report. FAIL exits with code 2; input/config errors exit with code 1.

## Important limitations

- This is NOT an independent text-to-token verification. IDs and masks are producer-provided. Independently re-tokenize sample source records using the frozen tokenizer before production.
- Matching tokenizer hashes proves matching declarations, not that the packer actually loaded the corresponding tokenizer file.
- Upstream metadata does not independently prove rights, compliance, near-duplicate removal or source quality.
- Traces must represent complete optimizer updates, not standalone microbatches. Gradient accumulation happens within the training runtime.
- Do not commit real data, token ID streams, credentials or checkpoints to this public repository.
- Real-data TPU 17-step resume, durable checkpoint restoration, full dataset manifests, and production training readiness remain open gates.
