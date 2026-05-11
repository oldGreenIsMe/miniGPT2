import os
import sys

sys.path.append(os.path.abspath("."))

import argparse
import torch

from src.config import DataConfig, GPTConfig, TrainerConfig
from src.dataset import CharDataset
from src.model.gpt import GPT
from src.trainer import Trainer
from src.utils import count_parameters, print_model_parameters, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Mini GPT-2 model from scratch."
    )

    # experiment
    parser.add_argument(
        "--run_name",
        type=str,
        default="default",
        help="Experiment run name. Results will be saved under runs/<run_name>/.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs",
        help="Root directory for experiment outputs.",
    )

    # data
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/input.txt",
        help="Path to input text file.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Text file encoding.",
    )

    # model
    parser.add_argument(
        "--block_size",
        type=int,
        default=64,
        help="Maximum context length.",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=4,
        help="Number of Transformer blocks.",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=128,
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--no_bias",
        action="store_true",
        help="Disable bias in Linear layers.",
    )

    # training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=1000,
        help="Maximum training iterations.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=100,
        help="Evaluate every N iterations.",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=20,
        help="Number of batches used for loss estimation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping threshold. Use 0 to disable.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Print training loss every N iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # optional explicit paths
    parser.add_argument(
        "--latest_ckpt_path",
        type=str,
        default=None,
        help="Optional path to save latest checkpoint.",
    )
    parser.add_argument(
        "--best_ckpt_path",
        type=str,
        default=None,
        help="Optional path to save best checkpoint.",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="Optional path to save tokenizer metadata.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Optional path to save training log CSV.",
    )

    # device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def validate_args(args):
    if args.n_embd % args.n_head != 0:
        raise ValueError(
            f"n_embd must be divisible by n_head, "
            f"but got n_embd={args.n_embd}, n_head={args.n_head}"
        )

    if args.block_size <= 0:
        raise ValueError("block_size must be positive.")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if args.max_iters <= 0:
        raise ValueError("max_iters must be positive.")

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1.")

    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1).")


def build_run_paths(args):
    run_dir = os.path.join(args.out_dir, args.run_name)

    default_latest_ckpt_path = os.path.join(
        run_dir, "checkpoints", "latest_model.pt"
    )
    default_best_ckpt_path = os.path.join(
        run_dir, "checkpoints", "best_model.pt"
    )
    default_meta_path = os.path.join(
        run_dir, "checkpoints", "meta.pkl"
    )
    default_log_path = os.path.join(
        run_dir, "logs", "train_log.csv"
    )
    default_config_path = os.path.join(
        run_dir, "config.json"
    )

    latest_ckpt_path = args.latest_ckpt_path or default_latest_ckpt_path
    best_ckpt_path = args.best_ckpt_path or default_best_ckpt_path
    meta_path = args.meta_path or default_meta_path
    log_path = args.log_path or default_log_path
    config_path = default_config_path

    return (
        run_dir,
        latest_ckpt_path,
        best_ckpt_path,
        meta_path,
        log_path,
        config_path,
    )


def main():
    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)

    (
        run_dir,
        latest_ckpt_path,
        best_ckpt_path,
        meta_path,
        log_path,
        config_path,
    ) = build_run_paths(args)

    data_cfg = DataConfig(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        encoding=args.encoding,
        seed=args.seed,
    )

    trainer_cfg = TrainerConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=device,
        log_interval=args.log_interval,
        latest_ckpt_path=latest_ckpt_path,
        best_ckpt_path=best_ckpt_path,
        meta_path=meta_path,
        log_path=log_path,
    )

    set_seed(data_cfg.seed)

    with open(data_cfg.data_path, "r", encoding=data_cfg.encoding) as f:
        text = f.read()

    dataset = CharDataset(
        text=text,
        block_size=args.block_size,
        train_ratio=data_cfg.train_ratio,
        device=trainer_cfg.device,
    )

    gpt_cfg = GPTConfig(
        vocab_size=dataset.get_vocab_size(),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=not args.no_bias,
    )

    model = GPT(gpt_cfg)

    total_params, trainable_params = count_parameters(model)

    experiment_config = {
        "run_name": args.run_name,
        "run_dir": run_dir,
        "data_config": {
            "data_path": data_cfg.data_path,
            "train_ratio": data_cfg.train_ratio,
            "encoding": data_cfg.encoding,
            "seed": data_cfg.seed,
        },
        "gpt_config": {
            "vocab_size": gpt_cfg.vocab_size,
            "block_size": gpt_cfg.block_size,
            "n_layer": gpt_cfg.n_layer,
            "n_head": gpt_cfg.n_head,
            "n_embd": gpt_cfg.n_embd,
            "head_size": gpt_cfg.n_embd // gpt_cfg.n_head,
            "dropout": gpt_cfg.dropout,
            "bias": gpt_cfg.bias,
        },
        "trainer_config": {
            "batch_size": trainer_cfg.batch_size,
            "max_iters": trainer_cfg.max_iters,
            "eval_interval": trainer_cfg.eval_interval,
            "eval_iters": trainer_cfg.eval_iters,
            "learning_rate": trainer_cfg.learning_rate,
            "weight_decay": trainer_cfg.weight_decay,
            "betas": trainer_cfg.betas,
            "grad_clip": trainer_cfg.grad_clip,
            "device": trainer_cfg.device,
            "log_interval": trainer_cfg.log_interval,
            "latest_ckpt_path": trainer_cfg.latest_ckpt_path,
            "best_ckpt_path": trainer_cfg.best_ckpt_path,
            "meta_path": trainer_cfg.meta_path,
            "log_path": trainer_cfg.log_path,
        },
        "model_info": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_m": total_params / 1e6,
        },
    }

    save_json(experiment_config, config_path)

    print("=" * 60)
    print("Dataset Info")
    print("=" * 60)
    print(f"Data path    : {data_cfg.data_path}")
    print(f"Vocab size   : {dataset.info.vocab_size}")
    print(f"Total tokens : {dataset.info.total_tokens}")
    print(f"Train tokens : {dataset.info.train_tokens}")
    print(f"Val tokens   : {dataset.info.val_tokens}")
    print("=" * 60)
    print()

    print("=" * 60)
    print("Experiment Config")
    print("=" * 60)
    print(f"Run name     : {args.run_name}")
    print(f"Run dir      : {run_dir}")
    print(f"Device       : {trainer_cfg.device}")
    print(f"Block size   : {gpt_cfg.block_size}")
    print(f"n_layer      : {gpt_cfg.n_layer}")
    print(f"n_head       : {gpt_cfg.n_head}")
    print(f"n_embd       : {gpt_cfg.n_embd}")
    print(f"head_size    : {gpt_cfg.n_embd // gpt_cfg.n_head}")
    print(f"dropout      : {gpt_cfg.dropout}")
    print(f"bias         : {gpt_cfg.bias}")
    print(f"batch_size   : {trainer_cfg.batch_size}")
    print(f"max_iters    : {trainer_cfg.max_iters}")
    print(f"lr           : {trainer_cfg.learning_rate}")
    print(f"weight_decay : {trainer_cfg.weight_decay}")
    print(f"latest ckpt  : {trainer_cfg.latest_ckpt_path}")
    print(f"best ckpt    : {trainer_cfg.best_ckpt_path}")
    print(f"meta path    : {trainer_cfg.meta_path}")
    print(f"log path     : {trainer_cfg.log_path}")
    print(f"config path  : {config_path}")
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