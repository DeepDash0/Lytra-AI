# Dataset file integrity manifests

Use scripts/data_manifest.py to record the SHA-256 and byte size for each *explicitly selected* data shard. Relative paths are kept; raw file content is never added to the manifest.

Create a manifest in a private, ignored artifact directory:

    py -3 scripts/data_manifest.py create --root path/to/private/shards --file shard01.jsonl --file shard02.jsonl --output artifacts/private_manifest.json

Verify later with the same root:

    py -3 scripts/data_manifest.py verify --root path/to/private/shards --manifest artifacts/private_manifest.json

An altered, missing or truncated file fails verification. A manifest is not a cryptographic signature: if an attacker can alter both data and manifest, the hash can be recomputed. Keep a trusted copy of the manifest digest outside the mutable dataset storage. No data provenance, license, tokenizer, content quality or completeness claims are made here.

Never commit private manifest paths, raw shard contents, tokens or user metadata. For a full immutable production manifest, include independent storage version IDs, source provenance, verified rights records and externally pinned digests.
