"""Reproduce the locked Lytra V1 parameter count."""

VOCAB = 65_536
HIDDEN = 2_048
LAYERS = 36
Q_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 128
INTERMEDIATE = 9_216

embedding = VOCAB * HIDDEN  # tied input/output embeddings

q_proj = HIDDEN * (Q_HEADS * HEAD_DIM)
k_proj = HIDDEN * (KV_HEADS * HEAD_DIM)
v_proj = HIDDEN * (KV_HEADS * HEAD_DIM)
o_proj = (Q_HEADS * HEAD_DIM) * HIDDEN
attention = q_proj + k_proj + v_proj + o_proj

# SwiGLU: gate projection + up projection + down projection
mlp = HIDDEN * INTERMEDIATE * 2 + INTERMEDIATE * HIDDEN

# Two RMSNorm vectors per transformer layer.
layer_norms = 2 * HIDDEN
per_layer = attention + mlp + layer_norms

final_norm = HIDDEN
total = embedding + LAYERS * per_layer + final_norm

print(f"Embedding:   {embedding:,}")
print(f"Per layer:   {per_layer:,}")
print(f"36 layers:   {LAYERS * per_layer:,}")
print(f"Final norm:  {final_norm:,}")
print(f"Total:       {total:,}")

assert total == 2_512_537_600
