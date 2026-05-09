import os
import sys

sys.path.append(os.path.abspath("."))

import torch

from src.config import GPTConfig
from src.model.gpt import GPT


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

    model = GPT(config)
    model.eval()

    B = 1
    T_prompt = 5

    idx = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(B, T_prompt),
        dtype=torch.long,
    )

    logits, loss, past_key_values = model(
        idx,
        past_key_values=None,
        use_cache=True,
    )

    print("===== KV Cache Prefill Test =====")
    print(f"idx shape    : {idx.shape}")
    print(f"logits shape : {logits.shape}")
    print(f"loss         : {loss}")

    print(f"num layers in cache: {len(past_key_values)}")

    for layer_idx, (k, v) in enumerate(past_key_values):
        print(
            f"layer {layer_idx}: "
            f"k shape = {k.shape}, "
            f"v shape = {v.shape}"
        )

    idx_next = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(B, 1),
        dtype=torch.long,
    )

    logits2, loss2, past_key_values2 = model(
        idx_next,
        past_key_values=past_key_values,
        use_cache=True,
    )

    print("\n===== KV Cache Decode One Step Test =====")
    print(f"idx_next shape : {idx_next.shape}")
    print(f"logits2 shape  : {logits2.shape}")
    print(f"loss2          : {loss2}")

    for layer_idx, (k, v) in enumerate(past_key_values2):
        print(
            f"layer {layer_idx}: "
            f"k shape = {k.shape}, "
            f"v shape = {v.shape}"
        )


if __name__ == "__main__":
    main()