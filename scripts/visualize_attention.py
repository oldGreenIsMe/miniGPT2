import os
import sys

sys.path.append(os.path.abspath("."))

import argparse

import torch
import matplotlib.pyplot as plt

from src.config import GPTConfig
from src.dataset import CharDataset
from src.model.gpt import GPT


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_name",
        type=str,
        default="gpt2_small",
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
        help="Path to checkpoint. If not set, use runs/<run_name>/checkpoints/best_model.pt.",
    )

    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to tokenizer meta. If not set, use runs/<run_name>/checkpoints/meta.pkl.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="To be",
        help="Prompt text to visualize.",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Layer index to visualize.",
    )

    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Attention head index to visualize.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save attention heatmap.",
    )

    return parser.parse_args()


def resolve_paths(args):
    run_dir = os.path.join(args.out_dir, args.run_name)

    ckpt_path = args.ckpt or os.path.join(
        run_dir,
        "checkpoints",
        "best_model.pt",
    )

    meta_path = args.meta or os.path.join(
        run_dir,
        "checkpoints",
        "meta.pkl",
    )

    save_path = args.save_path or os.path.join(
        run_dir,
        "outputs",
        f"attention_layer{args.layer}_head{args.head}.png",
    )

    return ckpt_path, meta_path, save_path


def main():
    args = parse_args()
    ckpt_path, meta_path, save_path = resolve_paths(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    gpt_cfg = GPTConfig(**checkpoint["gpt_config"])

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

    if idx.size(1) > gpt_cfg.block_size:
        idx = idx[:, -gpt_cfg.block_size:]
        prompt_ids = idx[0].tolist()

    with torch.no_grad():
        logits, _, _, attn_maps = model(
            idx,
            return_attn=True,
        )

    if attn_maps is None:
        raise RuntimeError("No attention maps were returned.")

    if args.layer < 0 or args.layer >= len(attn_maps):
        raise ValueError(
            f"layer index out of range: {args.layer}, "
            f"num layers = {len(attn_maps)}"
        )

    att = attn_maps[args.layer]

    # att: [B, n_head, T, T]
    if args.head < 0 or args.head >= att.size(1):
        raise ValueError(
            f"head index out of range: {args.head}, "
            f"num heads = {att.size(1)}"
        )

    att_head = att[0, args.head].numpy()

    tokens = [tokenizer.decode([i]) for i in prompt_ids]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.imshow(att_head, aspect="auto")
    plt.colorbar(label="Attention Weight")

    plt.xticks(
        ticks=range(len(tokens)),
        labels=tokens,
        rotation=90,
    )

    plt.yticks(
        ticks=range(len(tokens)),
        labels=tokens,
    )

    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.title(
        f"Attention Map | layer={args.layer}, head={args.head}\n"
        f"Prompt: {args.prompt}"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    print("=" * 60)
    print("Attention Visualization")
    print("=" * 60)
    print(f"Run name    : {args.run_name}")
    print(f"Checkpoint  : {ckpt_path}")
    print(f"Meta        : {meta_path}")
    print(f"Prompt      : {args.prompt}")
    print(f"Layer       : {args.layer}")
    print(f"Head        : {args.head}")
    print(f"Attn shape  : {att.shape}")
    print(f"Saved to    : {save_path}")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()