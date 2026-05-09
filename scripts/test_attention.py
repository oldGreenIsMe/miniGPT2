import os
import sys

sys.path.append(os.path.abspath("."))

import torch

from src.config import GPTConfig
from src.model.attention import CausalSelfAttention


def main():
    config = GPTConfig(
        vocab_size=47,
        block_size=16,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=True,
    )

    attn = CausalSelfAttention(config)

    B = 4
    T = 16
    C = config.n_embd

    x = torch.randn(B, T, C)

    y = attn(x)

    print("===== Attention Test =====")
    print(f"input x shape  : {x.shape}")
    print(f"output y shape : {y.shape}")

    print("\n===== Attention Config =====")
    print(f"n_embd    : {config.n_embd}")
    print(f"n_head    : {config.n_head}")
    print(f"head_size : {config.n_embd // config.n_head}")
    print(f"block_size: {config.block_size}")


if __name__ == "__main__":
    main()