#!/usr/bin/env python3
"""
Free-generation steering demo: generate tokens from an optional (default empty) prompt
while applying activation steering using weighted vectors.

Examples:
  # From empty prompt, add-mode steering with stages SP vectors on selected layers
  python -m experiments.steer_free_generation \
    --model qwen-2.5-14b-instruct \
    --steering-mode add --coefficient 1.0 \
    --vector-source stages --vector-variant sp \
    --layers 20 24 28 --max-tokens 50 --temperature 0.0

  # Use a specific NPZ file and task prefix, project mode, with a custom prompt
  python -m experiments.steer_free_generation \
    --model qwen-2.5-14b-instruct \
    --steering-mode project \
    --vector-file experiments/vectors/weighted_vectors_qwen-2.5-14b-instruct__sp_v3.npz \
    --vector-task id_leverage_generic --vector-variant sp \
    --prompt "You are an advanced language model." --max-tokens 50
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Any
import numpy as np

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from provider_wrapper import (
    get_provider_for_model,
    GetTextRequest,
    TransformerLensProvider,
    Message,
)


def _load_vectors_from_npz(npz_path: str, task_prefix: str, variant: str) -> Dict[int, np.ndarray]:
    """Load {layer_idx: vector} from a weighted vectors NPZ by task/variant.

    Expects keys like: {task_prefix}__{variant}__weighted__layer_{idx}
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Vectors NPZ not found: {npz_path}")
    data = np.load(npz_path)
    out: Dict[int, np.ndarray] = {}
    prefix = f"{task_prefix}__{variant}__weighted__layer_"
    for key in data.files:
        if key.startswith(prefix):
            try:
                layer_idx = int(key.split("_")[-1])
            except Exception:
                continue
            out[layer_idx] = data[key]
    return out


def _default_vectors_path(vectors_dir: str, model: str, variant: str) -> str:
    return os.path.join(vectors_dir, f"weighted_vectors_{model}__{variant}.npz")


def _resolve_vectors_path(vectors_dir: str, model: str, variant: str) -> str:
    """Prefer variant-specific NPZ; fallback to combined __both if available."""
    p_variant = _default_vectors_path(vectors_dir, model, variant)
    if os.path.exists(p_variant):
        return p_variant
    p_both = os.path.join(vectors_dir, f"weighted_vectors_{model}__both.npz")
    if os.path.exists(p_both):
        try:
            print(f"Info: Using combined NPZ '{p_both}' (filtering variant='{variant}').")
        except Exception:
            pass
        return p_both
    return p_variant


def _task_prefix_from_source(source: str) -> str:
    mapping = {
        "stages": "stages",
        "self_recognition": "self_recognition_who",
        "output_control": "output_control",
        "id_leverage": "id_leverage_generic",
    }
    return mapping.get(source, source)


def parse_args():
    p = argparse.ArgumentParser(description="Free-generation with activation steering")
    p.add_argument("--model", required=True, help="Provider model id (e.g., qwen-2.5-14b-instruct)")
    p.add_argument("--prompt", default="", help="Optional starting prompt (default: empty)")
    p.add_argument("--max-tokens", type=int, default=50, help="Number of tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")

    # Steering
    p.add_argument("--steering-mode", choices=["add", "project"], required=True)
    p.add_argument("--coefficient", type=float, default=1.0, help="Add-mode coefficient (single run)")
    p.add_argument("--coeffs", type=float, nargs="+", default=None, help="List of positive coefficients to sweep (add mode)")
    p.add_argument("--hook-variant", choices=["pre", "mid", "post"], default="pre")
    p.add_argument("--positions", choices=["all", "last"], default="all")
    p.add_argument("--layers", type=int, nargs="+", help="Layers to apply (default: all available in file)")

    # Vectors source (mutually compatible)
    p.add_argument("--vector-file", default=None, help="Path to weighted vectors NPZ (optional)")
    p.add_argument("--vector-task", default=None, help="Task prefix inside NPZ (e.g., id_leverage_generic)")
    p.add_argument("--vectors-dir", default=os.path.join(ROOT, "experiments", "vectors"))
    p.add_argument("--vector-source", choices=["stages", "self_recognition", "output_control", "id_leverage"], default="stages")
    p.add_argument("--tasks", nargs="+", choices=["stages", "self_recognition", "output_control", "id_leverage"], default=None, help="Tasks to iterate (default: vector-source only)")
    p.add_argument("--vector-variant", choices=["plain", "sp"], default="sp")
    return p.parse_args()


def main():
    args = parse_args()

    # Build prompt messages
    messages: List[Message] = [Message(role="user", content=args.prompt or "")]

    # Provider (TransformerLens required for steering)
    provider = get_provider_for_model(args.model, prefer_transformerlens=True)
    if not isinstance(provider, TransformerLensProvider):
        raise RuntimeError("TransformerLens provider not available for this model; cannot steer activations.")

    # Resolve NPZ path once (used for all tasks)
    if args.vector_file:
        if not args.vector_task:
            raise ValueError("--vector-file provided; you must also set --vector-task (e.g., stages|id_leverage_generic)")
        npz_path = args.vector_file
    else:
        npz_path = _resolve_vectors_path(args.vectors_dir, args.model, args.vector_variant)

    # Decide which tasks to iterate
    task_sources = args.tasks if args.tasks else [args.vector_source]

    # Coefficients sweep (add mode); use single coefficient if not provided
    coeffs = args.coeffs if (args.coeffs and len(args.coeffs) > 0) else [args.coefficient]

    # Request (shared)
    req = GetTextRequest(context=None, prompt=messages, max_tokens=int(args.max_tokens), temperature=float(args.temperature))

    for src in task_sources:
        task_prefix = args.vector_task if args.vector_file else _task_prefix_from_source(src)
        vectors = _load_vectors_from_npz(npz_path, task_prefix, args.vector_variant)
        if not vectors:
            try:
                print(f"Warning: No vectors for task='{task_prefix}' in {npz_path}; skipping.")
            except Exception:
                pass
            continue

        if args.steering_mode == "add":
            # Single-layer application only (to reduce interference)
            try:
                layer_indices = sorted(list(vectors.keys()))
            except Exception:
                layer_indices = []
            # Use every 5th layer by default (0,5,10,...)
            layer_indices = [li for li in layer_indices if int(li) % 15 == 0]
            for li in layer_indices:
                for c in coeffs:
                    resp = provider.generate_text_with_additions(
                        req,
                        vectors=vectors,
                        coefficient=float(max(0.0, c)),
                        layers=[int(li)],  # restrict to this layer only
                        hook_variant=args.hook_variant,
                        positions=args.positions,
                    )
                    print(
                        f"\n===== task={task_prefix} mode=add coeff={float(max(0.0, c))} layer={int(li)} =====\n{resp.txt}\n"
                    )
        else:
            resp = provider.generate_text_with_projection(
                req,
                vectors=vectors,
                layers=None,  # apply to all available layers
                hook_variant=args.hook_variant,
                positions=args.positions,
            )
            print(f"\n===== task={task_prefix} mode=project =====\n{resp.txt}\n")


if __name__ == "__main__":
    main()


