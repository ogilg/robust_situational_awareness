#!/usr/bin/env python3
"""
Bar chart for ID-Leverage condition breakdowns.

Reads a CSV like experiments/results/idlev_conditions_qwen-2.5-14b-instruct.csv
and plots paired bars (plain=blue, sp=red) per identity_phrase on the x-axis,
sorted by identity_true.

Usage:
  python -m experiments.plot_idlev_conditions \
    --csv experiments/results/idlev_conditions_qwen-2.5-14b-instruct.csv \
    --out experiments/results/idlev_conditions_qwen.png

Optional:
  --model qwen-2.5-14b-instruct   # filter rows for a specific model_id
  --true-first                    # put identity_true=1 first (default)
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def slug(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )


def load_conditions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize types
    for col in ["model_id", "variant", "identity_phrase"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    if "identity_true" in df.columns:
        df["identity_true"] = pd.to_numeric(df["identity_true"], errors="coerce").fillna(0).astype(int)
    if "n" in df.columns:
        df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    return df


def plot_idlev_conditions(
    df: pd.DataFrame,
    *,
    out_path: str | None,
    model_id: str | None,
    true_first: bool,
):
    if df.empty:
        print("No rows found.")
        return

    if model_id:
        df = df[df["model_id"].str.lower() == model_id.lower()]
        if df.empty:
            print(f"No rows for model_id={model_id}.")
            return

    # Ensure variants normalized
    df["variant"] = df["variant"].astype(str).str.lower()

    # Sort identity phrases by identity_true, then by phrase for stability
    df_sorted = df.sort_values(by=["identity_true", "identity_phrase"], ascending=[not true_first, True])
    phrases = df_sorted["identity_phrase"].drop_duplicates().tolist()

    # Pivot accuracy per variant
    d_plain = df_sorted[df_sorted["variant"] == "plain"].set_index("identity_phrase")
    d_sp = df_sorted[df_sorted["variant"] == "sp"].set_index("identity_phrase")

    acc_plain = [float(d_plain.loc[p, "accuracy"]) if p in d_plain.index else 0.0 for p in phrases]
    acc_sp = [float(d_sp.loc[p, "accuracy"]) if p in d_sp.index else 0.0 for p in phrases]

    x = list(range(len(phrases)))
    bar_width = 0.4
    offs_plain = [xi - bar_width / 2.0 for xi in x]
    offs_sp = [xi + bar_width / 2.0 for xi in x]

    plt.figure(figsize=(max(8, 0.4 * len(phrases) + 4), 5))
    ax = plt.gca()
    ax.bar(offs_plain, acc_plain, width=bar_width, color="blue", label="plain")
    ax.bar(offs_sp, acc_sp, width=bar_width, color="red", label="sp")

    plt.xticks(x, phrases, rotation=45, ha="right")
    plt.ylim(0, 1)
    title = "ID-Leverage conditions (accuracy by phrase and variant)"
    if model_id:
        title += f" | {model_id}"
    plt.title(title)
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.tight_layout()

    if not out_path:
        out_dir = os.path.join("experiments", "results")
        os.makedirs(out_dir, exist_ok=True)
        name = f"idlev_conditions_bars_{slug(model_id or 'all')}"
        out_path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(out_path, dpi=200)
    print(f"Wrote chart: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot ID-Leverage condition accuracies by variant")
    ap.add_argument("--csv", required=True, help="Path to idlev_conditions_{model}.csv")
    ap.add_argument("--model", default=None, help="Optional model_id filter")
    ap.add_argument("--out", default=None, help="Output image path")
    ap.add_argument("--true-first", action="store_true", help="Place identity_true=1 first (default)")
    args = ap.parse_args()

    df = load_conditions(args.csv)
    plot_idlev_conditions(df, out_path=args.out, model_id=args.model, true_first=True if args.true_first else True)


if __name__ == "__main__":
    main()


