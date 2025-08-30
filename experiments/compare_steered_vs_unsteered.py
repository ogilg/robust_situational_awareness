#!/usr/bin/env python3
"""
Compare steered vs unsteered performance for a chosen coefficient and layer,
across both prompt variants (plain, sp).

Inputs:
- scores CSV from experiments/benchmark_with_steering.py (steered)
- chosen coefficient and layer
- optional baseline CSV of unsteered rows (e.g., experiments/results/scores_*.csv)
- or a comment token to identify unsteered rows within the steered CSV

Outputs:
- Figures: bar charts per variant (plain, sp) with accuracy bars for unsteered vs steered per task
- CSV summary with columns: model, task, variant, coefficient, layer, acc_unsteered, acc_steered, delta
"""

import argparse
import ast
import os
import glob
import re
from typing import Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt


def ensure_out_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def parse_single_layer(layers_value) -> Optional[int]:
    if layers_value is None:
        return None
    if isinstance(layers_value, int):
        return layers_value
    if isinstance(layers_value, float):
        return int(layers_value)
    if isinstance(layers_value, str):
        txt = layers_value.strip()
        try:
            val = ast.literal_eval(txt)
            if isinstance(val, int):
                return val
            if isinstance(val, list) and len(val) == 1 and isinstance(val[0], int):
                return int(val[0])
        except Exception:
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
    return None


def parse_vector_variant_from_comment(comment: Optional[str]) -> Optional[str]:
    if not comment:
        return None
    m = re.search(r"\bvv=(\w+)\b", str(comment))
    return m.group(1) if m else None


TASK_ORDER = [
    "stages_oversight",
    "output_control",
    "id_leverage_generic",
]

TASK_COLORS: Dict[str, str] = {
    "stages_oversight": "tab:blue",
    "output_control": "tab:orange",
    "id_leverage_generic": "tab:green",
}


def compute_summary(
    steered_df: pd.DataFrame,
    unsteered_df: pd.DataFrame,
    *,
    coefficient: float,
    layer: int,
    vector_source: Optional[str],
    steered_vector_variant: Optional[str],
) -> pd.DataFrame:
    sd = steered_df.copy()
    sd["layer_single"] = sd["layers"].apply(parse_single_layer)
    sd["accuracy"] = pd.to_numeric(sd["accuracy"], errors="coerce")
    sd["coefficient"] = pd.to_numeric(sd["coefficient"], errors="coerce")
    sd["vector_variant"] = sd.get("comment", pd.Series([None] * len(sd))).apply(parse_vector_variant_from_comment)

    # Filter by chosen layer and coefficient for steered
    steered = sd[(sd["steering_mode"] != "none") & (sd["layer_single"] == int(layer)) & (sd["coefficient"] == float(coefficient))]
    if vector_source:
        steered = steered[steered["vector_source"] == vector_source]
    inferred_vv = None
    if steered_vector_variant:
        steered = steered[steered["vector_variant"] == steered_vector_variant]
        inferred_vv = steered_vector_variant
    else:
        # Try to infer vector-variant from steered rows' comments
        uniq_vv = sorted([v for v in steered["vector_variant"].dropna().unique().tolist() if isinstance(v, str)])
        if len(uniq_vv) == 1:
            inferred_vv = uniq_vv[0]

    # Prepare unsteered baseline from provided dataframe
    ud = unsteered_df.copy()
    ud["accuracy"] = pd.to_numeric(ud["accuracy"], errors="coerce")
    # Always use unsteered baseline from plain prompt variant only
    if "variant" in ud.columns:
        ud = ud[ud["variant"].astype(str).str.lower() == "plain"].copy()

    # Aggregate steered per (model_id, task, vector_variant)
    st_cols = ["model_id", "task", "vector_variant", "accuracy"]
    st_keep = steered[st_cols].copy() if not steered.empty else pd.DataFrame(columns=st_cols)
    if steered_vector_variant:
        st_keep = st_keep[st_keep["vector_variant"] == steered_vector_variant]
    st_agg_vv = (
        st_keep.groupby(["model_id", "task", "vector_variant"], as_index=False)["accuracy"].mean()
               .rename(columns={"accuracy": "acc_steered", "vector_variant": "steered_vector_variant"})
    )
    # Unsteered aggregate per (model_id, task)
    un_cols = ["model_id", "task", "variant", "accuracy"]
    un_keep = ud[un_cols].copy() if not ud.empty else pd.DataFrame(columns=un_cols)
    un_agg = (
        un_keep.groupby(["model_id", "task"], as_index=False)["accuracy"].mean()
               .rename(columns={"accuracy": "acc_unsteered"})
    )

    # Merge per task; carry steered vv
    merged = pd.merge(st_agg_vv, un_agg, on=["model_id", "task"], how="left")
    # Ensure numeric and compute delta without FutureWarning
    merged["acc_steered"] = pd.to_numeric(merged["acc_steered"], errors="coerce")
    merged["acc_unsteered"] = pd.to_numeric(merged["acc_unsteered"], errors="coerce")
    merged["delta"] = merged["acc_steered"].fillna(0.0) - merged["acc_unsteered"].fillna(0.0)
    merged["coefficient"] = float(coefficient)
    merged["layer"] = int(layer)
    if vector_source:
        merged["vector_source"] = vector_source
    # Ensure column present without overwriting inferred per-row vv
    if "steered_vector_variant" not in merged.columns:
        vv_value = steered_vector_variant or inferred_vv or "unknown"
        merged["steered_vector_variant"] = vv_value
    return merged


def plot_bars_by_vector_variant(summary: pd.DataFrame, out_dir: str, coefficient: float, layer: int) -> None:
    for vv, sub in summary.groupby("steered_vector_variant", dropna=False):
        sub = sub.copy()
        # Order tasks
        sub["task"] = pd.Categorical(sub["task"], categories=TASK_ORDER, ordered=True)
        sub = sub.sort_values("task")

        tasks = sub["task"].astype(str).tolist()
        x = range(len(tasks))
        width = 0.35

        plt.figure(figsize=(8, 5))
        colors = [TASK_COLORS.get(t, "tab:gray") for t in tasks]

        # Unsteered bars (only where available)
        mask_un = sub["acc_unsteered"].notna().tolist()
        x_un = [xi for xi, m in zip(x, mask_un) if m]
        acc_un = sub.loc[sub["acc_unsteered"].notna(), "acc_unsteered"].tolist()
        colors_un = [c for c, m in zip(colors, mask_un) if m]
        if x_un:
            plt.bar([xi - width/2 for xi in x_un], acc_un, width=width, color=colors_un, alpha=0.6, label="Unsteered")

        # Steered bars (only where available)
        mask_st = sub["acc_steered"].notna().tolist()
        x_st = [xi for xi, m in zip(x, mask_st) if m]
        acc_st = sub.loc[sub["acc_steered"].notna(), "acc_steered"].tolist()
        colors_st = [c for c, m in zip(colors, mask_st) if m]
        if x_st:
            plt.bar([xi + width/2 for xi in x_st], acc_st, width=width, color=colors_st, alpha=0.9, label="Steered", hatch="//")

        plt.xticks(list(x), tasks, rotation=20)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Accuracy")
        title_vv = vv if isinstance(vv, str) and vv else "unknown"
        plt.title(f"vv={title_vv} | coeff={coefficient} layer={layer}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend(loc="best")

        fname = f"compare_vv-{title_vv}_coeff-{coefficient}_layer-{layer}.png"
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare steered vs unsteered by variant")
    parser.add_argument("--scores-csv", required=True, help="Path to steered scores CSV")
    parser.add_argument("--baseline-csv", default=None, help="Optional unsteered baseline CSV (e.g., experiments/results/scores_*.csv)")
    parser.add_argument("--coefficient", type=float, required=True, help="Chosen steering coefficient")
    parser.add_argument("--layer", type=int, required=True, help="Chosen intervention layer")
    parser.add_argument("--unsteered-comment", default="no steering", help="Substring to identify unsteered rows within steered CSV (used if no baseline CSV)")
    parser.add_argument("--vector-source", default=None, help="Optional filter for vector_source (e.g., id_leverage)")
    parser.add_argument("--steered-vector-variant", default=None, choices=["plain", "sp"], help="Filter steered rows by vector-variant parsed from comment (vv=plain/sp)")
    parser.add_argument("--out-dir", default=None, help="Directory for outputs (default: <scores_dir>/figures_compare)")
    args = parser.parse_args()

    scores_csv = os.path.abspath(args.scores_csv)
    base_dir = os.path.dirname(scores_csv)
    out_dir = args.out_dir or os.path.join(base_dir, "figures_compare")
    ensure_out_dir(out_dir)

    steered_df = pd.read_csv(scores_csv)
    if args.baseline_csv:
        unsteered_df = pd.read_csv(os.path.abspath(args.baseline_csv))
    else:
        # Try to infer unsteered from steered CSV first
        tmp = steered_df.copy()
        unsteered_df = tmp[(tmp.get("steering_mode") == "none") & (tmp.get("comment").astype(str).str.contains(args.unsteered_comment, na=False))]
        # If none found, attempt to locate a baseline scores CSV in experiments/results/
        if unsteered_df.empty:
            try:
                model_ids = [m for m in steered_df.get("model_id", []).unique().tolist() if isinstance(m, str)]
            except Exception:
                model_ids = []
            candidate = None
            if model_ids:
                model = model_ids[0]
                # Prefer sibling results directory one level up
                candidate_path = os.path.abspath(os.path.join(base_dir, "..", f"scores_{model}.csv"))
                if os.path.exists(candidate_path):
                    candidate = candidate_path
                else:
                    # Fallback: search under experiments/results
                    results_root = os.path.abspath(os.path.join(base_dir, ".."))
                    for path in glob.glob(os.path.join(results_root, "scores_*.csv")):
                        if model in os.path.basename(path):
                            candidate = path
                            break
            if candidate and os.path.exists(candidate):
                unsteered_df = pd.read_csv(candidate)

    summary = compute_summary(
        steered_df,
        unsteered_df,
        coefficient=args.coefficient,
        layer=args.layer,
        vector_source=args.vector_source,
        steered_vector_variant=args.steered_vector_variant,
    )

    # Save summary CSV
    sum_path = os.path.join(out_dir, f"summary_coeff-{args.coefficient}_layer-{args.layer}.csv")
    summary.to_csv(sum_path, index=False)

    # Print concise table
    if not summary.empty:
        printable = summary[["model_id", "steered_vector_variant", "task", "acc_unsteered", "acc_steered", "delta"]].sort_values(["steered_vector_variant", "task"]).fillna("")
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(printable.to_string(index=False))
    else:
        print("No matching rows found. Tips: pass --baseline-csv, check --vector-source, --steered-vector-variant, --coefficient, and --layer.")

    # Plot per steered vector-variant
    plot_bars_by_vector_variant(summary, out_dir, args.coefficient, args.layer)

    print(f"Wrote comparison figures and summary to: {out_dir}")


if __name__ == "__main__":
    main()


