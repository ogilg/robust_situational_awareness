#!/usr/bin/env python3
"""
Compare cosine similarities between versions of weighted vectors for a given
task (e.g., id_leverage_generic) across NPZ files.

Now produces a pairwise matrix of mean cosine across overlapping layers and
plots a heatmap, plus a CSV of the matrix. Also prints layer overlap counts.

Examples:
  python -m experiments.compare_vectors_versions \
    --task id_leverage_generic \
    --vectors-sp-v1 experiments/vectors/weighted_vectors_qwen-2.5-14b-instruct__sp.npz \
    --vectors-sp-v2 experiments/vectors/weighted_vectors_qwen-2.5-14b-instruct__sp_v2.npz \
    --vectors-sp-v3 experiments/vectors/weighted_vectors_qwen-2.5-14b-instruct__sp_v3.npz \
    --vectors-plain experiments/vectors/weighted_vectors_qwen-2.5-14b-instruct__plain.npz \
    --output-dir experiments/results/qwen_analysis_sp

Outputs:
  - CSV with pairwise mean cosine matrix across overlapping layers
  - PNG heatmap of the matrix
  - Console summary with overlaps
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List


def parse_vector_key(key: str) -> Tuple[str, int]:
    """Return (task_with_variant, layer_idx) from keys like:
    'id_leverage_generic__sp__weighted__layer_47' or 'id_leverage_generic__weighted__layer_47'.
    """
    if "__weighted__layer_" not in key:
        raise ValueError
    task_part, layer_part = key.split("__weighted__layer_")
    return task_part, int(layer_part)


def strip_variant_suffix(task_with_variant: str) -> Tuple[str, str]:
    parts = task_with_variant.split("__")
    if len(parts) >= 2 and parts[-1] in {"plain", "sp"}:
        return "__".join(parts[:-1]), parts[-1]
    return task_with_variant, ""


def load_task_vectors(npz_path: str, task_name: str, only_variant: str = "") -> Dict[int, np.ndarray]:
    """Load vectors for a base task name, returning {layer_idx: vector}.

    Matches keys regardless of whether a variant suffix exists in the stored task.
    """
    if npz_path is None or not os.path.exists(npz_path):
        return {}
    data = np.load(npz_path)
    out: Dict[int, np.ndarray] = {}
    for key in data.files:
        try:
            task_with_variant, layer_idx = parse_vector_key(key)
        except Exception:
            continue
        base_task, variant = strip_variant_suffix(task_with_variant)
        if base_task == task_name and (not only_variant or variant == only_variant):
            out[layer_idx] = data[key]
    return out


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    n1 = float(np.linalg.norm(u))
    n2 = float(np.linalg.norm(v))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(u, v) / max(n1 * n2, 1e-12))


def mean_cosine_across_overlap(task_vectors_a: Dict[int, np.ndarray], task_vectors_b: Dict[int, np.ndarray]) -> Tuple[float, int]:
    layers = sorted(set(task_vectors_a.keys()) & set(task_vectors_b.keys()))
    if not layers:
        return 0.0, 0
    vals: List[float] = []
    for l in layers:
        vals.append(cosine(task_vectors_a[l], task_vectors_b[l]))
    return float(np.mean(vals)), len(layers)


def main():
    parser = argparse.ArgumentParser(description="Compare cosine similarities between versions of weighted vectors")
    parser.add_argument("--task", default="id_leverage_generic", help="Base task name (no variant suffix)")
    parser.add_argument("--vectors-sp-v1", required=True, help="NPZ path containing SP v1 vectors (can be __sp or __both)")
    parser.add_argument("--vectors-sp-v2", required=True, help="NPZ path containing SP v2 vectors (can be __sp_v2 or __both)")
    parser.add_argument("--vectors-sp-v3", default=None, help="NPZ path containing SP v3 vectors (can be __sp_v3 or __both)")
    parser.add_argument("--vectors-plain", default=None, help="NPZ path containing PLAIN vectors (can be __plain or __both)")
    parser.add_argument("--output-dir", default=os.path.join("experiments", "results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tv_sp_v1 = load_task_vectors(args.vectors_sp_v1, args.task, only_variant="sp")
    tv_sp_v2 = load_task_vectors(args.vectors_sp_v2, args.task, only_variant="sp")
    tv_sp_v3 = load_task_vectors(args.vectors_sp_v3, args.task, only_variant="sp") if args.vectors_sp_v3 else {}
    tv_plain = load_task_vectors(args.vectors_plain, args.task, only_variant="plain") if args.vectors_plain else {}

    # Build list of versions present with human-readable SP labels
    labels: List[str] = ["Full SP", "Just first line"]
    dicts: List[Dict[int, np.ndarray]] = [tv_sp_v1, tv_sp_v2]
    if tv_sp_v3:
        labels.append("All but first line")
        dicts.append(tv_sp_v3)
    if tv_plain:
        labels.append("plain")
        dicts.append(tv_plain)

    # Compute pairwise mean cosine matrix
    n = len(labels)
    mat = np.zeros((n, n), dtype=float)
    overlaps = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
                overlaps[i, j] = len(dicts[i])
            else:
                m, k = mean_cosine_across_overlap(dicts[i], dicts[j])
                mat[i, j] = m
                overlaps[i, j] = k

    # Print summary
    print(f"Comparing task='{args.task}' across versions:")
    for i in range(n):
        for j in range(i + 1, n):
            print(f"  {labels[i]} vs {labels[j]}: mean={mat[i,j]:.3f} over {overlaps[i,j]} overlapping layers")
            if overlaps[i, j] == 0:
                # As a fallback, compute cosine between per-version mean vectors so the matrix is still informative
                def mean_vec(d):
                    if not d:
                        return None
                    arr = np.stack([v for _, v in sorted(d.items())], axis=0)
                    return arr.mean(axis=0)
                vi = mean_vec(dicts[i])
                vj = mean_vec(dicts[j])
                if vi is not None and vj is not None:
                    fallback = cosine(vi, vj)
                    print(f"    (no overlapping layers; fallback cosine of mean vectors = {fallback:.3f})")

    # Save matrix CSV
    import csv
    csv_path = os.path.join(args.output_dir, f"compare_versions_matrix_{args.task}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label"] + labels)
        for i in range(n):
            w.writerow([labels[i]] + [f"{mat[i,j]:.6f}" for j in range(n)])
    print(f"Saved matrix CSV: {csv_path}")

    # Heatmap
    try:
        import seaborn as sns
        plt.figure(figsize=(5 + n * 0.5, 4 + n * 0.5))
        sns.heatmap(mat, annot=True, fmt='.3f', cmap='coolwarm', center=0, xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Mean cosine'})
        plt.title(f"Mean Cosine Across Layers for {args.task}")
        plt.tight_layout()
        png_path = os.path.join(args.output_dir, f"compare_versions_matrix_{args.task}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Saved matrix heatmap: {png_path}")
        plt.close()
    except Exception:
        # Fallback simple save if seaborn unavailable
        plt.figure(figsize=(6, 4))
        plt.imshow(mat, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.xticks(range(n), labels, rotation=45, ha='right')
        plt.yticks(range(n), labels)
        plt.title(f"Mean Cosine Across Layers for {args.task}")
        plt.tight_layout()
        png_path = os.path.join(args.output_dir, f"compare_versions_matrix_{args.task}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Saved matrix heatmap (fallback): {png_path}")
        plt.close()


if __name__ == "__main__":
    main()


