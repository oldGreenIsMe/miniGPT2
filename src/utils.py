import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def count_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_params, trainable_params


def print_model_parameters(model: torch.nn.Module):
    total_params, trainable_params = count_parameters(model)

    print("=" * 60)
    print("Model Parameter Count")
    print("=" * 60)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print(f"Total params (M) : {total_params / 1e6:.3f} M")
    print("=" * 60)

    print("\nParameters by top-level module:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20}: {params:,}")