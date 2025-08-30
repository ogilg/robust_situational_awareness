#!/usr/bin/env python3
"""
Tiny script to load aggregated vectors and compute a per-task, per-layer
weighted vector: (sum_correct) - (sum_incorrect).

If you interpret weights as mean*count, this equals
  (mean_correct * num_correct) - (mean_incorrect * num_incorrect)
because mean = sum / count.

Usage:
  python experiments/reduce_vectors.py --model llama-3.1-8b-instruct \
      --in-dir experiments/results --out experiments/results
"""

import os
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compute weighted vectors from aggregated sums (per task and per variant)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--in-dir", default=os.path.join("experiments", "results"))
    parser.add_argument("--out", default=None, help="Output npz path (defaults to weighted_vectors_{model}.npz in in-dir)")
    parser.add_argument("--variant", choices=["plain", "sp", "both"], default="both", help="Which variant to reduce: plain, sp, or both")
    args = parser.parse_args()

    in_dir = args.in_dir
    model = args.model
    # Vectors now live under the parent experiments directory (sibling of results)
    vectors_dir = os.path.join(os.path.dirname(in_dir), "vectors")
    counts_json_path = os.path.join(vectors_dir, f"aggregated_vectors_{model}.counts.json")
    assert os.path.exists(counts_json_path), f"Missing counts file: {counts_json_path}"
    with open(counts_json_path, "r") as f:
        counts = json.load(f)

    # Keys now look like: "{task}__{variant}" where variant in {plain, sp}
    # We'll produce: "{task}__{variant}__weighted__layer_{idx}"
    tasks = list(counts.keys())
    out_payload = {}
    summary = []

    # Filter by requested variant(s)
    if args.variant in ("plain", "sp"):
        tasks = [t for t in tasks if t.endswith(f"__{args.variant}")]

    for task in tasks:
        # Per-task NPZ
        task_npz_path = os.path.join(vectors_dir, f"aggregated_vectors_{model}__{task}.npz")
        assert os.path.exists(task_npz_path), f"Missing per-task vectors file: {task_npz_path}"
        data = np.load(task_npz_path)
        # Collect layers present for this task
        task_keys = list(data.keys())
        layers = set()
        for k in task_keys:
            # parse layer index
            if "__layer_" in k:
                try:
                    idx = int(k.split("__layer_")[-1])
                    layers.add(idx)
                except Exception:
                    pass
        layers = sorted(layers)

        num_c = int(counts.get(task, {}).get("correct", 0) or 0)
        num_i = int(counts.get(task, {}).get("incorrect", 0) or 0)

        for layer_idx in layers:
            k_c = f"correct__layer_{layer_idx}"
            k_i = f"incorrect__layer_{layer_idx}"
            vec_c = data[k_c] if k_c in data else None
            vec_i = data[k_i] if k_i in data else None
            if vec_c is None and vec_i is None:
                continue
            # Treat missing as zeros
            if vec_c is None:
                vec_c = np.zeros_like(vec_i)
            if vec_i is None:
                vec_i = np.zeros_like(vec_c)

            # Weighted aggregate: sum_correct - sum_incorrect
            weighted = vec_c - vec_i
            out_key = f"{task}__weighted__layer_{layer_idx}"
            out_payload[out_key] = weighted
        summary.append((task, len(layers), num_c, num_i))

    if args.out:
        out_path = args.out
    else:
        os.makedirs(vectors_dir, exist_ok=True)
        suffix = args.variant if args.variant in ("plain", "sp") else "both"
        out_path = os.path.join(vectors_dir, f"weighted_vectors_{model}__{suffix}.npz")
    if out_payload:
        np.savez(out_path, **out_payload)
        print(f"Wrote weighted vectors: {out_path} ({len(out_payload)} arrays)")
    else:
        print("No vectors found to write.")

    for task, n_layers, num_c, num_i in summary:
        print(f"Task={task} layers={n_layers} correct={num_c} incorrect={num_i}")


if __name__ == "__main__":
    main()


