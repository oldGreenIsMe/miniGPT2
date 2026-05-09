import os
import sys

sys.path.append(os.path.abspath("."))

import torch

from src.config import DataConfig, GPTConfig, TrainerConfig
from src.dataset import CharDataset
from src.model.gpt import GPT
from src.trainer import Trainer
from src.utils import print_model_parameters, set_seed


def main():
    data_cfg = DataConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer_cfg = TrainerConfig(
        batch_size=32,
        max_iters=1000,
        eval_interval=100,
        eval_iters=20,
        learning_rate=3e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        device=device,
        log_interval=20,
        latest_ckpt_path="checkpoints/latest_model.pt",
        best_ckpt_path="checkpoints/best_model.pt",
        log_path="logs/train_log.csv",
    )

    set_seed(data_cfg.seed)

    with open(data_cfg.data_path, "r", encoding=data_cfg.encoding) as f:
        text = f.read()

    block_size = 64

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

    print("=" * 60)
    print("Dataset Info")
    print("=" * 60)
    print(f"Vocab size   : {dataset.info.vocab_size}")
    print(f"Total tokens : {dataset.info.total_tokens}")
    print(f"Train tokens : {dataset.info.train_tokens}")
    print(f"Val tokens   : {dataset.info.val_tokens}")
    print("=" * 60)
    print()

    print_model_parameters(model)
    print()

    trainer = Trainer(
        model=model,
        dataset=dataset,
        gpt_config=gpt_cfg,
        trainer_config=trainer_cfg,
    )

    trainer.train()


if __name__ == "__main__":
    main()