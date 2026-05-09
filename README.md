# Mini GPT-2 From Scratch

This project implements a small GPT-2 style autoregressive language model from scratch using PyTorch.

The goal is not to reproduce the full OpenAI GPT-2 model, but to build a clear, modular, and engineering-oriented Mini GPT-2 implementation with standard Transformer components.

---

## Features

- Character-level tokenizer
- Config-driven model and training setup
- GPT-style token embedding and positional embedding
- Standard 4D multi-head causal self-attention
- Pre-norm Transformer block
- MLP feed-forward network with GELU
- Autoregressive text generation
- Temperature sampling
- Top-k sampling
- Train / validation loss logging
- Checkpoint saving
- Loss curve visualization
- Basic KV cache implementation for autoregressive inference

---

## Project Structure

```text
mini_gpt2/
├── data/
│   └── input.txt
├── checkpoints/
│   ├── best_model.pt
│   └── latest_model.pt
├── logs/
│   └── train_log.csv
├── outputs/
│   └── loss_curve.png
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── utils.py
│   └── model/
│       ├── attention.py
│       ├── mlp.py
│       ├── block.py
│       └── gpt.py
├── scripts/
│   ├── test_dataset.py
│   ├── test_model_forward.py
│   ├── test_attention.py
│   ├── test_block.py
│   ├── test_full_gpt.py
│   ├── test_kv_cache.py
│   ├── train.py
│   ├── infer.py
│   └── visualize.py
├── requirements.txt
└── README.md
```

---

## Model Architecture

The model follows a simplified GPT-2 style architecture.

```text
input token ids: [B, T]
↓
token embedding: [B, T, C]
↓
position embedding: [T, C]
↓
embedding sum + dropout: [B, T, C]
↓
Transformer Block × n_layer
↓
final LayerNorm
↓
LM head
↓
logits: [B, T, vocab_size]
```

Each Transformer block uses a pre-norm structure:

```python
x = x + attention(layer_norm_1(x))
x = x + mlp(layer_norm_2(x))
```

---

## Attention Shape

The attention module uses standard 4D multi-head attention.

```text
x: [B, T, C]

q, k, v: [B, T, C]

reshape:
[B, T, C] → [B, T, n_head, head_size]

transpose:
[B, T, n_head, head_size] → [B, n_head, T, head_size]

attention score:
[B, n_head, T, head_size] @ [B, n_head, head_size, T]
→ [B, n_head, T, T]

attention output:
[B, n_head, T, T] @ [B, n_head, T, head_size]
→ [B, n_head, T, head_size]

merge heads:
[B, n_head, T, head_size] → [B, T, C]
```

---

## Dataset

The project currently uses a character-level tokenizer.

The input text is encoded into a one-dimensional token sequence:

```text
data: [N]
```

Training batches are sampled as:

```text
x: [B, T]
y: [B, T]
```

where `y` is `x` shifted by one token:

```text
x = [t0, t1, t2, ..., tN]
y = [t1, t2, t3, ..., tN+1]
```

This implements next-token prediction.

---

## Training

Run:

```bash
python scripts/train.py
```

The training script will:

- Load `data/input.txt`
- Build the dataset
- Initialize the Mini GPT-2 model
- Train the model
- Evaluate train / validation loss periodically
- Save checkpoints
- Write logs to `logs/train_log.csv`

Checkpoints are saved to:

```text
checkpoints/latest_model.pt
checkpoints/best_model.pt
```

---

## Inference

Generate text without KV cache:

```bash
python scripts/infer.py --prompt "To be" --max_new_tokens 100 --temperature 0.8 --top_k 10
```

Generate text with KV cache:

```bash
python scripts/infer.py --prompt "To be" --max_new_tokens 20 --temperature 0.8 --top_k 10 --use_cache
```

Current KV cache implementation is a minimal educational version and does not yet support rolling cache beyond `block_size`.

---

## Visualization

Run:

```bash
python scripts/visualize.py
```

This reads:

```text
logs/train_log.csv
```

and saves:

```text
outputs/loss_curve.png
```

---

## KV Cache

The project implements a basic KV cache for autoregressive inference.

### Prefill Stage

```text
idx: [B, T_prompt]

per layer:
k: [B, n_head, T_prompt, head_size]
v: [B, n_head, T_prompt, head_size]
```

### Decode Stage

```text
idx_next: [B, 1]

new k/v:
[B, n_head, 1, head_size]

past k/v:
[B, n_head, T_past, head_size]

updated k/v:
[B, n_head, T_past + 1, head_size]
```

This avoids recomputing historical K/V at every generation step.

---

## Current Limitations

- Character-level tokenizer only
- No BPE tokenizer
- Small dataset may lead to overfitting
- No learning rate scheduler
- No mixed precision training
- No rolling KV cache
- No FlashAttention
- No HuggingFace checkpoint compatibility
- No distributed training

---

## Suggested Future Improvements

- Save and load tokenizer metadata
- Add BPE tokenizer
- Add learning rate warmup and cosine decay
- Add mixed precision training
- Add rolling KV cache
- Add larger text dataset
- Add model comparison experiments
- Add command-line training arguments
- Add experiment config files

---

## Example Result

With a very small training corpus, the model can generate character-level text, but the output quality is limited by data size.

Example:

```text
Prompt:
To be

Generated:
To beers oumendon bes ano
```

The result shows that the generation pipeline works, but better text quality requires a larger dataset and more stable training.

---

## Project Summary

This project implements a Mini GPT-2 style autoregressive language model from scratch in PyTorch.

The model includes:

- Token embedding
- Positional embedding
- Multi-head causal self-attention
- Pre-norm Transformer blocks
- GELU MLP
- Residual connections
- LM head
- Training loop
- Checkpointing
- Inference generation
- Temperature and top-k sampling
- Loss visualization
- Basic KV cache inference

The implementation is designed for learning and engineering practice rather than production deployment.