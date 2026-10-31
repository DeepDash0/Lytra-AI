# Lytra dataset sample audit (v0)

Before large training runs, inspect 1–10M candidate content tokens for provenance, source rights declarations, duplicated text, held-out contamination and exact frozen-tokenizer content-token counts. Raw text is never exported by the audit.

## Setup

Install the optional tokenizer runtime: python -m pip install tokenizers
Run tests: python -m unittest discover -s tests -v

Input is UTF-8 JSONL; one record per line. Required keys are text, source, license, origin, split and approved_for_training. The origin must be human, synthetic or teacher. Teacher-generated records also need teacher_use_approved=true. Sample record:

{"text":"Example training sample.","source":"corpus-A","license":"permission reference","origin":"human","split":"train","approved_for_training":true}

## Run

python scripts/dataset_audit.py sample.jsonl --tokenizer-json path/to/frozen/tokenizer.json --heldout-hashes path/to/eval_hashes.txt --target-tokens 10000000 --output audit_report.json

The optional heldout hashes file must contain one lowercase SHA-256 per line, computed from NFC-normalized, stripped evaluation text. Missing heldout hashes means contamination has NOT been assessed. The report includes SHA-256 of the tokenizer file to make token counts reproducible.

Rights and teacher-use approval flags are declarations and must be independently verified against contracts, provider terms and applicable licenses. Exact normalized-text deduplication is not near-duplicate detection. No BOS/EOS, padding, truncation, packing efficiency or optimizer token-slot capacity is included in accepted_content_tokens. Final training manifest and actual packing validation remain separate gates. Keep raw data and private tokenizer outside this public repository.
