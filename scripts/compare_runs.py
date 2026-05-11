import os
import sys
import json

sys.path.append(os.path.abspath("."))

import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple Mini GPT-2 experiment runs."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs",
        help="Root directory containing experiment runs.",
    )

    parser.add_argument(
        "--run_names",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Run names to compare. "
            "If not provided, all subdirectories under out_dir will be used."
        ),
    )

    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Path to save comparison CSV summary.",
    )

    parser.add_argument(
        "--fig_path",
        type=str,
        default=None,
        help="Path to save comparison loss figure.",
    )

    return parser.parse_args()


def discover_runs(out_dir: str):
    run_names = []

    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"out_dir not found: {out_dir}")

    for name in os.listdir(out_dir):
        run_dir = os.path.join(out_dir, name)
        log_path = os.path.join(run_dir, "logs", "train_log.csv")

        if os.path.isdir(run_dir) and os.path.exists(log_path):
            run_names.append(name)

    run_names = sorted(run_names)
    return run_names


def load_run_log(out_dir: str, run_name: str):
    log_path = os.path.join(out_dir, run_name, "logs", "train_log.csv")

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found for run '{run_name}': {log_path}")

    df = pd.read_csv(log_path)

    required_cols = {"iter", "train_loss", "val_loss"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Log file {log_path} must contain columns {required_cols}, "
            f"but got {set(df.columns)}"
        )

    return df, log_path


def load_run_config(out_dir: str, run_name: str):
    config_path = os.path.join(out_dir, run_name, "config.json")

    if not os.path.exists(config_path):
        return None, config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config, config_path


def summarize_run(run_name: str, df: pd.DataFrame, config: dict | None = None):
    best_idx = df["val_loss"].idxmin()

    best_iter = int(df.loc[best_idx, "iter"])
    best_train_loss = float(df.loc[best_idx, "train_loss"])
    best_val_loss = float(df.loc[best_idx, "val_loss"])

    final_iter = int(df.iloc[-1]["iter"])
    final_train_loss = float(df.iloc[-1]["train_loss"])
    final_val_loss = float(df.iloc[-1]["val_loss"])

    if config is not None:
        gpt_config = config.get("gpt_config", {})
        model_info = config.get("model_info", {})

        n_layer = gpt_config.get("n_layer", None)
        n_head = gpt_config.get("n_head", None)
        n_embd = gpt_config.get("n_embd", None)
        block_size = gpt_config.get("block_size", None)
        total_params = model_info.get("total_params", None)
        total_params_m = model_info.get("total_params_m", None)
    else:
        n_layer = None
        n_head = None
        n_embd = None
        block_size = None
        total_params = None
        total_params_m = None

    return {
        "run_name": run_name,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "block_size": block_size,
        "total_params": total_params,
        "total_params_m": total_params_m,
        "num_eval_points": len(df),
        "best_iter": best_iter,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "final_iter": final_iter,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "overfit_gap_final": final_val_loss - final_train_loss,
    }


def print_summary_table(summary_df: pd.DataFrame):
    print("=" * 100)
    print("Experiment Comparison Summary")
    print("=" * 100)

    display_cols = [
        "run_name",
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "total_params_m",
        "best_iter",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "overfit_gap_final",
    ]

    print(summary_df[display_cols].to_string(index=False))
    print("=" * 100)


def plot_compare_loss(out_dir: str, run_names: list[str], fig_path: str):
    plt.figure(figsize=(10, 6))

    for run_name in run_names:
        df, _ = load_run_log(out_dir, run_name)

        plt.plot(
            df["iter"],
            df["val_loss"],
            marker="o",
            label=f"{run_name} val",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison Across Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=200)

    print(f"Saved comparison figure to: {fig_path}")
    plt.show()


def main():
    args = parse_args()

    if args.run_names is None or len(args.run_names) == 0:
        run_names = discover_runs(args.out_dir)
    else:
        run_names = args.run_names

    if len(run_names) == 0:
        raise ValueError(
            f"No valid runs found under {args.out_dir}. "
            f"Expected logs at runs/<run_name>/logs/train_log.csv"
        )

    summary_rows = []

    print("=" * 100)
    print("Loading Runs")
    print("=" * 100)

    for run_name in run_names:
        df, log_path = load_run_log(args.out_dir, run_name)
        config, config_path = load_run_config(args.out_dir, run_name)
    
        print(f"{run_name:<30} | log: {log_path}")
        print(f"{'':<30} | cfg: {config_path}")
    
        row = summarize_run(run_name, df, config)
        summary_rows.append(row)

    print("=" * 100)
    print()

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("best_val_loss").reset_index(drop=True)

    print_summary_table(summary_df)

    summary_path = args.summary_path or os.path.join(
        args.out_dir,
        "compare_summary.csv",
    )

    fig_path = args.fig_path or os.path.join(
        args.out_dir,
        "compare_loss.png",
    )

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved comparison summary to: {summary_path}")

    plot_compare_loss(
        out_dir=args.out_dir,
        run_names=run_names,
        fig_path=fig_path,
    )


if __name__ == "__main__":
    main()