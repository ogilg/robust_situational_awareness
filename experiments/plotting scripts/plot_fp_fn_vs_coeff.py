#!/usr/bin/env python3
"""
Plot false positive/negative rates vs steering coefficient, split by variant.

Source CSV: language_details_*_steering.csv (from benchmark_with_steering id_leverage runs)

Usage:
  python experiments/analysis/plot_fp_fn_vs_coeff.py \
    --csv experiments/results/qwen_steering/language_details_qwen-2.5-14b-instruct_steering.csv \
    --out experiments/results/qwen_steering/fp_fn_vs_coeff.png
"""

import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FP/FN rates vs coefficient by variant")
    parser.add_argument("--csv", required=True, help="Path to language_details_*_steering.csv")
    parser.add_argument("--out", required=True, help="Path to save PNG plot")
    return parser.parse_args()


def compute_fp_fn(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize fields
    target = df["target_language"].astype(str).str.lower()
    pred = df["detected_language"].astype(str).str.lower()
    identity_true = df.get("identity_true", True)
    try:
        identity_true = identity_true.astype(bool)
    except Exception:
        identity_true = identity_true.apply(lambda x: str(x).strip().lower() in {"1", "true", "t"})

    # Expected target: english when identity_true is False, else stated target
    y_true = target.where(identity_true, other="english")
    y_pred = pred

    df = df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["is_fp"] = ((df["y_true"] == "english") & (df["y_pred"] != "english")).astype(int)
    df["is_fn"] = ((df["y_true"] != "english") & (df["y_pred"] == "english")).astype(int)
    return df


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if "variant" not in df.columns:
        raise ValueError("CSV must contain a 'variant' column (e.g., 'plain' or 'sp').")

    df.rename(columns={"variant": "vector_variant"}, inplace=True)

    # Keep only id_leverage runs if mixed; otherwise this is a no-op
    # (language_details_*_steering.csv is created in id_leverage run helper)
    df = compute_fp_fn(df)

    # Ensure numeric coefficient
    # Some logs store as string; coerce
    df["coefficient"] = pd.to_numeric(df["coefficient"], errors="coerce")
    # Group by variant and coefficient
    grp = df.groupby(["vector_variant", "coefficient"], as_index=False).agg(
        n=("sample_id", "count"),
        fp_sum=("is_fp", "sum"),
        fn_sum=("is_fn", "sum"),
    )
    grp["fp_rate"] = grp["fp_sum"] / grp["n"].clip(lower=1)
    grp["fn_rate"] = grp["fn_sum"] / grp["n"].clip(lower=1)

    # Melt for plotting
    long_df = grp.melt(
        id_vars=["vector_variant", "coefficient"],
        value_vars=["fp_rate", "fn_rate"],
        var_name="error_type",
        value_name="rate",
    )
    long_df["error_type"] = long_df["error_type"].map({"fp_rate": "FP", "fn_rate": "FN"})

    plt.figure(figsize=(8, 5))
    palette = {"plain": "#1f77b4", "sp": "#d62728"}
    ax = sns.lineplot(
        data=long_df.sort_values(["error_type", "vector_variant", "coefficient"]),
        x="coefficient",
        y="rate",
        hue="vector_variant",
        style="error_type",
        hue_order=["plain", "sp"],
        palette=palette,
        markers=True,
        dashes=False,
    )
    ax.set_ylabel("Rate")
    ax.set_xlabel("Coefficient")
    ax.set_title("False Positive / Negative Rates vs steering coefficient for ID-Leverage")
    ax.legend(title="Vector variant / Error", loc="best")
    # Show only two x-axis tick values (min and max available)
    coeffs = sorted(long_df["coefficient"].dropna().unique())
    if len(coeffs) >= 2:
        xt = [coeffs[0], coeffs[-1]]
    else:
        xt = coeffs
    ax.set_xticks(xt)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()


