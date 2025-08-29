#!/usr/bin/env python3
"""
Tiny script: make a bar chart of model performance across 4 tasks.

Reads all scores_*.csv from a results directory, filters rows by a given
comment (substring match), and plots accuracy per task. Each series in the
chart corresponds to a (model_id, variant) pair and the legend includes a
compact sample size.

Usage:
  python -m experiments.plot_scores \
    --results-dir experiments/results \
    --comment "qwen" \
    --out experiments/results/score_bars.png
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


EXPECTED_TASKS = [
    "stages_oversight",
    "self_recognition_who",
    "output_control",
    "id_leverage_generic",
]


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


def load_scores(results_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "scores_*.csv")))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Normalize columns we use
    for col in ["model", "model_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "model_id" not in df.columns and "model" in df.columns:
        df["model_id"] = df["model"]
    if "variant" in df.columns:
        df["variant"] = df["variant"].astype(str)
    if "task" in df.columns:
        df["task"] = df["task"].astype(str)
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    if "n_processed" in df.columns:
        df["n_processed"] = pd.to_numeric(df["n_processed"], errors="coerce").fillna(0).astype(int)
    if "comment" in df.columns:
        df["comment"] = df["comment"].astype(str)
    return df


def plot_bars(df: pd.DataFrame, *, comment: str | None, out_path: str | None):
    if df.empty:
        print("No score rows found.")
        return
    if comment:
        df = df[df["comment"].str.contains(comment, case=False, na=False)]
        if df.empty:
            print(f"No rows match comment substring: {comment!r}")
            return

    # Keep only expected tasks
    df = df[df["task"].isin(EXPECTED_TASKS)]
    if df.empty:
        print("No rows for expected tasks.")
        return

    # Series per (model_id, variant)
    series_keys = sorted(df[["model_id", "variant"]].drop_duplicates().itertuples(index=False, name=None))

    # Build accuracy table: rows=tasks, cols=series
    tasks = EXPECTED_TASKS
    x = range(len(tasks))
    width = max(0.1, 0.8 / max(len(series_keys), 1))

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    for idx, (model_id, variant) in enumerate(series_keys):
        dsub = df[(df["model_id"] == model_id) & (df["variant"] == variant)]
        acc = [
            float(dsub.loc[dsub["task"] == t, "accuracy"].values[0])
            if not dsub.loc[dsub["task"] == t, "accuracy"].empty else 0.0
            for t in tasks
        ]
        # Derive a compact n label (median n_processed across tasks)
        n_vals = dsub.set_index("task").reindex(tasks)["n_processed"].dropna().astype(int).tolist()
        n_label = f"n={int(pd.Series(n_vals).median()) if n_vals else 0}"
        label = f"{model_id} ({variant}, {n_label})"
        offs = [xi + (idx - (len(series_keys)-1)/2) * width for xi in x]
        ax.bar(offs, acc, width=width, label=label)

    plt.xticks(list(x), [
        "stages",
        "self_rec",
        "output_control",
        "id_leverage",
    ])
    # Add random baseline (50%) markers only for stages and self_recognition
    baseline_tasks = {"stages_oversight", "self_recognition_who"}
    for i, t in enumerate(tasks):
        if t in baseline_tasks:
            # Span approximately the full width of the grouped bars at this x
            xmin = i - 0.45
            xmax = i + 0.45
            ax.hlines(0.5, xmin, xmax, colors="red", linestyles="dotted", linewidth=1)
    plt.ylim(0, 1)
    title = "Model performance across tasks"
    if comment:
        title += f" | comment: {comment}"
    plt.title(title)
    plt.ylabel("accuracy")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if not out_path:
        out_dir = os.path.join("experiments", "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"score_bars_{slug(comment or 'all')}.png")
    plt.savefig(out_path, dpi=200)
    print(f"Wrote chart: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Bar chart of model performance across tasks")
    ap.add_argument("--results-dir", default=os.path.join("experiments", "results"))
    ap.add_argument("--comment", default=None, help="Filter rows by substring match on comment")
    ap.add_argument("--out", default=None, help="Output image path")
    args = ap.parse_args()

    df = load_scores(args.results_dir)
    plot_bars(df, comment=args.comment, out_path=args.out)


if __name__ == "__main__":
    main()


