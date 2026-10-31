# Lytra V1: checkpoint replay evidence gate (v0)

## Scope

`scripts/resume_replay.py` is a **strict evidence comparator** for a 17-update experiment, interrupted after update 8. It compares an uninterrupted reference trace (steps 1–17), a separate pre-checkpoint trace (steps 1–8), and a separate restored continuation (steps 9–17). It verifies the SHA-256 and byte length of the checkpoint artifact against its sidecar, confirms cursor continuity and compares per-step declared digests and loss.

The tool **does not perform TPU training**. A PASS proves internal consistency of the supplied logs, not that a TPU really ran or that the checkpoint serializer restored each state correctly. Production-ready remains false.

## Trace schema (private JSONL)

Each step must have these fields:

```json
{
  "step": 1,
  "batch_sha256": "<64 lowercase hex>",
  "model_sha256": "<64 lowercase hex>",
  "optimizer_sha256": "<64 lowercase hex>",
  "scheduler_sha256": "<64 lowercase hex>",
  "rng_sha256": "<64 lowercase hex>",
  "loader_sha256": "<64 lowercase hex>",
  "dataset_manifest_sha256": "<64 lowercase hex>",
  "tokenizer_sha256": "<64 lowercase hex>",
  "config_sha256": "<64 lowercase hex>",
  "data_cursor": 1,
  "loss": 12.0
}
```

The actual cursor should represent a monotonic data-stream position; the toy example increments it per optimizer update. The restored continuation must also include `"resumed_from_step": 8` on each row. State digests must be computed over canonical, deterministic serialized state representations by the training producer, and should cover all ranks/devices. The batch fingerprint should uniquely identify the entire global optimizer batch in stable sequence order. Loss must be finite.

## Checkpoint sidecar (private JSON)

Sidecar fields: `schema_version` (1), `checkpoint_step` (8), `data_cursor`, `checkpoint_file_sha256`, `checkpoint_file_size`, all five state digests (`model_sha256`, `optimizer_sha256`, `scheduler_sha256`, `rng_sha256`, `loader_sha256`) and the three identity pins (`dataset_manifest_sha256`, `tokenizer_sha256`, `config_sha256`). It should be captured at the saved optimizer step, not reconstructed from the reference run. This checks **file bytes against the sidecar**, but does not authenticate the sidecar or inspect serialized tensors inside an opaque checkpoint.

## Run a 17-step comparison

```bash
python -m scripts.resume_replay \
  --baseline /private/baseline.jsonl \
  --before /private/before.jsonl \
  --after /private/after.jsonl \
  --checkpoint /private/checkpoint.bin \
  --checkpoint-sidecar /private/checkpoint.meta.json \
  --expected-dataset-sha256 <trusted_manifest_entries_sha256> \
  --expected-tokenizer-sha256 <trusted_tokenizer_sha256> \
  --expected-config-sha256 <trusted_training_config_sha256> \
  --output artifacts/replay_report.json
```

Windows PowerShell users can put the command on one line instead of using bash backslashes.

Exit code 0 = `PASS_DECLARED_STRICT_REPLAY`; 2 = evidence mismatch; 1 = malformed/missing inputs. Default model/optimizer/scheduler/RNG/loader/batch digests must match **exactly**; `--loss-atol` affects only loss values, not state hashes. A non-bitwise-deterministic TPU backend may require a different, separately reviewed numerical validation method, rather than weakening this strict gate.

## Synthetic regression

```bash
python -m unittest discover -s tests -q
python -m scripts.smoke_resume
```

The smoke test uses a tiny deterministic integer state machine, writes a JSON checkpoint, restores into a new state object, and verifies all 17 steps and checkpoint tamper detection. No actual neural network or TPU is involved.

## Technical preflight integration

Preflight now expects `--replay-report` plus the existing dataset, packing, alignment and integrity reports. It requires a matching tokenizer pin and the replay dataset SHA-256 equal to the shard manifest's canonical `entries_sha256`. Synthetic tests may produce `PASS_TECHNICAL_ONLY`, but the output *always* reports `production_ready=false` and explicitly lists independent real-world gates.

Real production still needs an authenticated full-state checkpoint (including tensor state and all devices), actual independent reloading by the trainer, verified continuous data-loader cursor/stream provenance, remote durable-storage restoration, approved data rights, and the real 17-step TPU training comparison.
