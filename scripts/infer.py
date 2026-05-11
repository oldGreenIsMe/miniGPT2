import os
import sys

sys.path.append(os.path.abspath("."))

import argparse

import torch

from src.config import GPTConfig
from src.dataset import CharDataset
from src.model.gpt import GPT


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Experiment run name under runs/.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs",
        help="Root directory of experiment runs.",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint.",
    )

    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to tokenizer meta file.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="To be",
        help="Input prompt text.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k sampling. Use 0 to disable.",
    )

    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use KV cache generation.",
    )

    return parser.parse_args()


def resolve_paths(args):
    if args.run_name is not None:
        run_dir = os.path.join(args.out_dir, args.run_name)

        ckpt = args.ckpt or os.path.join(
            run_dir, "checkpoints", "best_model.pt"
        )
        meta = args.meta or os.path.join(
            run_dir, "checkpoints", "meta.pkl"
        )
    else:
        ckpt = args.ckpt or "checkpoints/best_model.pt"
        meta = args.meta or "checkpoints/meta.pkl"

    return ckpt, meta


def main():
    args = parse_args()
    ckpt_path, meta_path = resolve_paths(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    gpt_config_dict = checkpoint["gpt_config"]
    gpt_cfg = GPTConfig(**gpt_config_dict)

    tokenizer = CharDataset.load_tokenizer(meta_path)

    model = GPT(gpt_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)

    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    top_k = args.top_k if args.top_k > 0 else None

    if args.use_cache:
        out = model.generate_with_cache(
            idx=idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )
    else:
        out = model.generate(
            idx=idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )

    generated_ids = out[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    print("=" * 60)
    print("Generation Result")
    print("=" * 60)
    print(f"Run name         : {args.run_name}")
    print(f"Checkpoint       : {ckpt_path}")
    print(f"Tokenizer meta   : {meta_path}")
    print(f"Prompt           : {args.prompt}")
    print(f"Max new tokens   : {args.max_new_tokens}")
    print(f"Temperature      : {args.temperature}")
    print(f"Top-k            : {top_k}")
    print(f"Use KV cache     : {args.use_cache}")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()