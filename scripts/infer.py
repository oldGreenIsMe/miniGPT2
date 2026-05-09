import os
import sys

sys.path.append(os.path.abspath("."))

import argparse

import torch

from src.config import DataConfig, GPTConfig
from src.dataset import CharDataset
from src.model.gpt import GPT


def load_dataset(data_path: str, encoding: str, block_size: int, device: str):
    with open(data_path, "r", encoding=encoding) as f:
        text = f.read()

    dataset = CharDataset(
        text=text,
        block_size=block_size,
        train_ratio=0.9,
        device=device,
    )

    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to checkpoint",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="To be",
        help="Input prompt text",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Number of new tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k sampling. Use 0 to disable.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        args.ckpt,
        map_location=device,
        weights_only=False,
    )

    gpt_config_dict = checkpoint["gpt_config"]
    gpt_cfg = GPTConfig(**gpt_config_dict)

    data_cfg = DataConfig()

    dataset = load_dataset(
        data_path=data_cfg.data_path,
        encoding=data_cfg.encoding,
        block_size=gpt_cfg.block_size,
        device=device,
    )

    model = GPT(gpt_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    prompt_ids = dataset.encode(args.prompt)
    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    top_k = args.top_k if args.top_k > 0 else None

    out = model.generate(
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
    )

    generated_ids = out[0].tolist()
    generated_text = dataset.decode(generated_ids)

    print("=" * 60)
    print("Generation Result")
    print("=" * 60)
    print(f"Checkpoint       : {args.ckpt}")
    print(f"Prompt           : {args.prompt}")
    print(f"Max new tokens   : {args.max_new_tokens}")
    print(f"Temperature      : {args.temperature}")
    print(f"Top-k            : {top_k}")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()