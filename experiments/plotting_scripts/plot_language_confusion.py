#!/usr/bin/env python3
"""
Plot a language confusion matrix from a language_details CSV.

Usage:
  python experiments/analysis/plot_language_confusion.py \
    --csv experiments/results/language_details_qwen-2.5-14b-instruct.csv \
    --out experiments/results/language_confusion_qwen-2.5-14b-instruct.png
"""

import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot language confusion matrix")
    parser.add_argument("--csv", required=True, help="Path to language_details_*.csv")
    parser.add_argument("--out", required=True, help="Path to save confusion matrix PNG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    # If identity_true is false, the expected (target) language should be English
    target = df["target_language"].astype(str).str.lower()
    identity_true = df.get("identity_true", True)
    try:
        identity_true = identity_true.astype(bool)
    except Exception:
        identity_true = identity_true.apply(lambda x: str(x).strip().lower() in {"1", "true", "t"})

    y_true = target.where(identity_true, other="english")
    y_pred = df["detected_language"].astype(str).str.lower()

    # Summary error metrics
    fp = int(((y_true == "english") & (y_pred != "english")).sum())
    fn = int(((y_true != "english") & (y_pred == "english")).sum())
    print(f"False positives (should be english, predicted non-english): {fp}")
    print(f"False negatives (should be non-english, predicted english): {fn}")

    labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Detected language")
    ax.set_ylabel("Target language")
    ax.set_title("Language Confusion Matrix")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()


