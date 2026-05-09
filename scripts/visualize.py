import os
import sys

sys.path.append(os.path.abspath("."))

import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/train_log.csv",
        help="Path to training log CSV file",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/loss_curve.png",
        help="Path to save loss curve figure",
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        raise FileNotFoundError(f"Log file not found: {args.log_path}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    df = pd.read_csv(args.log_path)

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
    print(f"Log path        : {args.log_path}")
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

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Mini GPT-2 Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(args.save_path, dpi=200)
    print(f"Saved loss curve to: {args.save_path}")

    plt.show()


if __name__ == "__main__":
    main()