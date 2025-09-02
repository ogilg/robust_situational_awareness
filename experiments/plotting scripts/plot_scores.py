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

    # We draw paired bars (plain=blue, sp=red) per task, for each model_id.
    tasks = EXPECTED_TASKS
    x = list(range(len(tasks)))
    model_ids = sorted(df["model_id"].drop_duplicates().tolist())
    num_models = max(1, len(model_ids))
    group_width = 0.8 / num_models
    bar_width = group_width / 2.0

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    handles = []
    labels = []

    for midx, model_id in enumerate(model_ids):
        dmodel = df[df["model_id"] == model_id]
        d_plain = dmodel[dmodel["variant"].str.lower() == "plain"].set_index("task")
        d_sp = dmodel[dmodel["variant"].str.lower() == "sp"].set_index("task")

        # Accuracies per task for each variant
        acc_plain = [float(d_plain.loc[t, "accuracy"]) if t in d_plain.index else 0.0 for t in tasks]
        acc_sp = [float(d_sp.loc[t, "accuracy"]) if t in d_sp.index else 0.0 for t in tasks]

        # n label: max of n_processed across variants and tasks
        n_vals = []
        if "n_processed" in d_plain:
            n_vals.extend(d_plain.reindex(tasks)["n_processed"].dropna().astype(int).tolist())
        if "n_processed" in d_sp:
            n_vals.extend(d_sp.reindex(tasks)["n_processed"].dropna().astype(int).tolist())
        n_label = f"n={max(n_vals) if n_vals else 0}"

        # Base offsets per model group
        base_offs = [xi + (midx - (num_models - 1) / 2.0) * group_width for xi in x]
        offs_plain = [bo - bar_width / 2.0 for bo in base_offs]
        offs_sp = [bo + bar_width / 2.0 for bo in base_offs]

        h_plain = ax.bar(offs_plain, acc_plain, width=bar_width, color="blue")
        h_sp = ax.bar(offs_sp, acc_sp, width=bar_width, color="red")

        handles.append((h_plain, h_sp))
        labels.append(f"{model_id} ({n_label})")

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
    # Build a custom legend: one entry per model_id showing both variants
    from matplotlib.patches import Patch
    legend_entries = []
    for lbl in labels:
        # Show variant colors once globally; append model label separately
        pass
    # Global variant legend
    variant_patches = [Patch(color="blue", label="plain"), Patch(color="red", label="sp")]
    model_patches = [Patch(facecolor="none", edgecolor="none", label=lbl) for lbl in labels]
    ax.legend(handles=variant_patches + model_patches, loc="best", fontsize=8)
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


