#!/usr/bin/env python3
"""
Plot steering sweep results.

Generates two sets of plots from a scores CSV produced by
experiments/benchmark_with_steering.py:

1) Accuracy vs Layer (one figure per constant-params group; one line per task)
2) Accuracy vs Coefficient (one figure per constant-params group; one line per task)

Notes:
- Expects 'layers' to be a single layer per row (e.g., "[40]"). Rows with
  multi-layer interventions are skipped in layer-based grouping.
- Attempts to parse vector-variant from the 'comment' field if it contains
  a token like "vv=sp" or "vv=plain".
"""

import argparse
import ast
import os
import re
from typing import Optional, List, Dict

import pandas as pd
import matplotlib.pyplot as plt


def parse_single_layer(layers_value: str | int | float) -> Optional[int]:
    """Parse the CSV 'layers' field to a single integer layer if possible.

    Accepts formats like "[40]" or 40. Returns None if ambiguous or invalid.
    """
    if layers_value is None:
        return None
    # Already an int
    if isinstance(layers_value, int):
        return layers_value
    # Strings: try literal_eval when looks like list/number
    if isinstance(layers_value, str):
        txt = layers_value.strip()
        try:
            val = ast.literal_eval(txt)
            if isinstance(val, int):
                return val
            if isinstance(val, list) and len(val) == 1 and isinstance(val[0], int):
                return int(val[0])
        except Exception:
            # Fallback: strip brackets and try int
            if txt.startswith("[") and txt.endswith("]"):
                inner = txt[1:-1].strip()
                try:
                    return int(inner)
                except Exception:
                    return None
            try:
                return int(txt)
            except Exception:
                return None
    # Floats should not appear, but handle gracefully
    if isinstance(layers_value, float):
        iv = int(layers_value)
        return iv
    return None


def parse_vector_variant_from_comment(comment: str | None) -> Optional[str]:
    """Extract vector variant token like 'sp' or 'plain' from comment if present.

    Looks for a pattern like 'vv=sp' or 'vv=plain'. Returns None if not found.
    """
    if not comment:
        return None
    match = re.search(r"\bvv=(\w+)\b", str(comment))
    if match:
        return match.group(1)
    return None


def ensure_out_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def sanitize(value: object) -> str:
    s = str(value)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    return s


def build_group_title(prefix: str, group_vals: Dict[str, object]) -> str:
    parts = [prefix]
    for key in [
        "model_id",
        "variant",
        "vector_source",
        "coefficient",
        "positions",
        "vector_variant",
        "layer_single",
    ]:
        if key in group_vals and group_vals[key] is not None:
            parts.append(f"{key}={group_vals[key]}")
    return " | ".join(parts)


TASK_COLORS: Dict[str, str] = {
    "stages_oversight": "tab:blue",
    "output_control": "tab:orange",
    "id_leverage_generic": "tab:green",
}


def plot_accuracy_vs_layer(df: pd.DataFrame, out_dir: str) -> None:
    """Create line plots of accuracy vs layer for each constant-params group.

    Grouping keys keep everything constant except the x-axis (layer) and task.
    """
    # Only consider rows with a single layer and steering sweeps (mode != none)
    df_l = df.copy()
    df_l["layer_single"] = df_l["layers"].apply(parse_single_layer)
    df_l = df_l[df_l["layer_single"].notna()].copy()
    df_l = df_l[df_l["steering_mode"] != "none"].copy()

    if df_l.empty:
        return

    group_keys = [
        "model_id",
        "variant",
        "steering_mode",
        "vector_source",
        "coefficient",
        "hook_variant",
        "positions",
        "vector_variant",
    ]

    for group_vals, sub in df_l.groupby(group_keys, dropna=False):
        sub = sub.copy()
        # Ensure multiple layers exist for a meaningful line plot
        if sub["layer_single"].nunique() < 2:
            continue
        # Aggregate duplicates by mean accuracy per task, per layer
        agg = (
            sub.groupby(["task", "layer_single"], as_index=False)["accuracy"].mean()
            .sort_values(["task", "layer_single"]) 
        )

        plt.figure(figsize=(8, 5))
        for task, task_df in agg.groupby("task"):
            color = TASK_COLORS.get(task)
            plt.plot(
                task_df["layer_single"],
                task_df["accuracy"],
                marker="o",
                label=task,
                color=color,
            )
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        gv = dict(zip(group_keys, group_vals))
        short_title = f"vector variant={gv.get('vector_variant', '')} | source={gv.get('vector_source', '')}"
        plt.title(short_title)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Task", loc="best")
        fname = (
            "layers_"
            + "_".join(f"{k}-{sanitize(v)}" for k, v in zip(group_keys, group_vals))
            + ".png"
        )
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def plot_accuracy_vs_coefficient(df: pd.DataFrame, out_dir: str) -> None:
    """Create line plots of accuracy vs coefficient for each constant-params group.

    Grouping keys keep everything constant except the x-axis (coefficient) and task.
    """
    df_c = df.copy()
    df_c["layer_single"] = df_c["layers"].apply(parse_single_layer)
    df_c = df_c[df_c["layer_single"].notna()].copy()
    df_c = df_c[df_c["steering_mode"] != "none"].copy()

    if df_c.empty:
        return

    # Overlay across layers: do not group by layer; draw one line per (task, layer)
    group_keys = [
        "model_id",
        "variant",
        "vector_source",
        "positions",
        "vector_variant",
    ]

    for group_vals, sub in df_c.groupby(group_keys, dropna=False):
        sub = sub.copy()
        # Ensure multiple coefficients exist for a meaningful line plot
        if sub["coefficient"].nunique() < 2:
            continue
        # Aggregate duplicates by mean accuracy per task, per layer, per coefficient
        agg = (
            sub.groupby(["task", "layer_single", "coefficient"], as_index=False)["accuracy"].mean()
              .sort_values(["task", "layer_single", "coefficient"]) 
        )

        plt.figure(figsize=(8, 5))
        # Assign linestyles by layer to distinguish overlays
        unique_layers = sorted([int(x) for x in agg["layer_single"].dropna().unique().tolist()])
        linestyles = ["-", "--", ":", "-."]
        layer_to_style = {layer: linestyles[i % len(linestyles)] for i, layer in enumerate(unique_layers)}

        for (task, layer), task_layer_df in agg.groupby(["task", "layer_single"]):
            color = TASK_COLORS.get(task)
            style = layer_to_style.get(int(layer), "-")
            label = f"{task} L{int(layer)}"
            plt.plot(
                task_layer_df["coefficient"],
                task_layer_df["accuracy"],
                marker="o",
                label=label,
                color=color,
                linestyle=style,
            )
        plt.xlabel("Coefficient")
        plt.ylabel("Accuracy")
        gv = dict(zip(group_keys, group_vals))
        short_title = f"vector variant={gv.get('vector_variant', '')} | source={gv.get('vector_source', '')}"
        plt.title(short_title)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Task/Layer", loc="best")
        fname = (
            "coeffs_"
            + "_".join(f"{k}-{sanitize(v)}" for k, v in zip(group_keys, group_vals))
            + ".png"
        )
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot steering sweep scores")
    parser.add_argument("--scores-csv", required=True, help="Path to scores CSV")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write figures (default: <scores_csv_dir>/figures)",
    )
    args = parser.parse_args()

    scores_csv = os.path.abspath(args.scores_csv)
    out_dir = args.out_dir or os.path.join(os.path.dirname(scores_csv), "figures")
    ensure_out_dir(out_dir)

    df = pd.read_csv(scores_csv)

    # Normalize fields
    # accuracy may be string formatted; ensure float
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["coefficient"] = pd.to_numeric(df["coefficient"], errors="coerce")
    # add vector_variant from comment when available
    df["vector_variant"] = df.get("comment", pd.Series([None] * len(df))).apply(
        parse_vector_variant_from_comment
    )

    plot_accuracy_vs_layer(df, out_dir)
    plot_accuracy_vs_coefficient(df, out_dir)

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()


