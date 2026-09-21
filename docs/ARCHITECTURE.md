# Lytra V1 Architecture

Lytra V1 is a dense decoder-only Transformer trained from random initialization.

| Component | Value |
|---|---:|
| Parameters | 2,512,537,600 |
| Layers | 36 |
| Hidden size | 2,048 |
| Query heads | 16 |
| KV heads | 2 |
| Head dimension | 128 |
| SwiGLU intermediate | 9,216 |
| Vocabulary | 65,536 |
| Norm | RMSNorm |
| RMSNorm epsilon | 1e-6 |
| RoPE theta | 1,000,000 |
| Initial context | 4,096 |

Input and output embeddings are tied. GQA with two KV heads is used to reduce KV-cache cost.

The initial 4K pretraining path is intentionally conservative. Longer context is planned as a progressive extension rather than being mixed into the first training phase.
