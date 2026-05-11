import os
import sys

sys.path.append(os.path.abspath("."))

import torch

from src.config import GPTConfig
from src.model.gpt import GPT


def main():
    config = GPTConfig(
        vocab_size=47,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.1,
        bias=True,
    )

    model = GPT(config)
    model.eval()

    B = 1
    prompt_len = 5
    steps = 10

    idx = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(B, prompt_len),
        dtype=torch.long,
    )

    logits, _, past_key_values, _ = model(
        idx,
        past_key_values=None,
        use_cache=True,
    )

    print("=" * 60)
    print("Rolling KV Cache Test")
    print("=" * 60)
    print(f"block_size : {config.block_size}")
    print(f"prompt len : {prompt_len}")
    print(f"steps      : {steps}")
    print("=" * 60)

    print("\nAfter prefill:")
    for layer_idx, (k, v) in enumerate(past_key_values):
        print(
            f"layer {layer_idx}: "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}"
        )

    for step in range(1, steps + 1):
        idx_next = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(B, 1),
            dtype=torch.long,
        )

        logits, _, past_key_values, _ = model(
            idx_next,
            past_key_values=past_key_values,
            use_cache=True,
        )

        cache_len = past_key_values[0][0].size(2)

        print(f"\nAfter decode step {step}: cache_len={cache_len}")

        for layer_idx, (k, v) in enumerate(past_key_values):
            print(
                f"layer {layer_idx}: "
                f"k={tuple(k.shape)}, v={tuple(v.shape)}"
            )

        assert cache_len <= config.block_size, (
            f"cache_len={cache_len} exceeds block_size={config.block_size}"
        )

    print("\nPASS: rolling KV cache length never exceeds block_size.")
    print("=" * 60)


if __name__ == "__main__":
    main()