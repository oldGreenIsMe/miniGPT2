import os
import sys

sys.path.append(os.path.abspath("."))

import argparse

import pandas as pd
import matplotlib.pyplot as plt


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
        "--log_path",
        type=str,
        default=None,
        help="Path to training log CSV file.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save loss curve figure.",
    )

    return parser.parse_args()


def resolve_paths(args):
    if args.run_name is not None:
        run_dir = os.path.join(args.out_dir, args.run_name)
        log_path = args.log_path or os.path.join(
            run_dir, "logs", "train_log.csv"
        )
        save_path = args.save_path or os.path.join(
            run_dir, "outputs", "loss_curve.png"
        )
    else:
        log_path = args.log_path or "logs/train_log.csv"
        save_path = args.save_path or "outputs/loss_curve.png"

    return log_path, save_path


def main():
    args = parse_args()
    log_path, save_path = resolve_paths(args)

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(log_path)

    required_cols = {"iter", "train_loss", "val_loss"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_cols}, "
            f"but got: {set(df.columns)}"
        )

    best_idx = df["val_loss"].idxmin()
    best_iter = int(df.loc[best_idx, "iter"])
    best_val_loss = float(df.loc[best_idx, "val_loss"])
    best_train_loss = float(df.loc[best_idx, "train_loss"])

    print("=" * 60)
    print("Training Log Summary")
    print("=" * 60)
    print(f"Log path        : {log_path}")
    print(f"Save path       : {save_path}")
    print(f"Num eval points : {len(df)}")
    print(f"Best iter       : {best_iter}")
    print(f"Best train loss : {best_train_loss:.4f}")
    print(f"Best val loss   : {best_val_loss:.4f}")
    print("=" * 60)

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["iter"],
        df["train_loss"],
        marker="o",
        label="Train Loss",
    )

    plt.plot(
        df["iter"],
        df["val_loss"],
        marker="o",
        label="Val Loss",
    )

    plt.scatter(
        [best_iter],
        [best_val_loss],
        s=100,
        label=f"Best Val Loss @ iter {best_iter}",
    )

    title = "Mini GPT-2 Training / Validation Loss"
    if args.run_name is not None:
        title += f" ({args.run_name})"

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    print(f"Saved loss curve to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()