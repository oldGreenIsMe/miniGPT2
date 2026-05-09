import os
import sys

sys.path.append(os.path.abspath("."))

import torch

from src.config import DataConfig, GPTConfig, TrainerConfig
from src.dataset import CharDataset
from src.model.gpt import GPT
from src.utils import print_model_parameters, set_seed


def main():
    data_cfg = DataConfig()
    trainer_cfg = TrainerConfig(
        batch_size=4,
        device="cpu",
    )

    set_seed(data_cfg.seed)

    with open(data_cfg.data_path, "r", encoding=data_cfg.encoding) as f:
        text = f.read()

    block_size = 16

    dataset = CharDataset(
        text=text,
        block_size=block_size,
        train_ratio=data_cfg.train_ratio,
        device=trainer_cfg.device,
    )

    gpt_cfg = GPTConfig(
        vocab_size=dataset.get_vocab_size(),
        block_size=block_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=True,
    )

    model = GPT(gpt_cfg)
    model = model.to(trainer_cfg.device)

    x, y = dataset.get_batch(
        split="train",
        batch_size=trainer_cfg.batch_size,
    )

    logits, loss = model(x, y)

    print("===== Full GPT Forward Test =====")
    print(f"x shape      : {x.shape}")
    print(f"y shape      : {y.shape}")
    print(f"logits shape : {logits.shape}")
    print(f"loss         : {loss.item():.4f}")

    print("\n===== Config =====")
    print(f"vocab_size : {gpt_cfg.vocab_size}")
    print(f"block_size : {gpt_cfg.block_size}")
    print(f"n_layer    : {gpt_cfg.n_layer}")
    print(f"n_head     : {gpt_cfg.n_head}")
    print(f"n_embd     : {gpt_cfg.n_embd}")

    print()
    print_model_parameters(model)


if __name__ == "__main__":
    main()