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
    p.add_argument("--vector-file", required=True, help="Path to steering vectors NPZ file")
    p.add_argument("--steering-mode", choices=["add", "project"], default="add", help="Steering mode")
    p.add_argument("--coeffs", type=float, nargs="+", default=[1.0], help="List of coefficients to sweep")
    p.add_argument("--prompt", default="", help="Optional starting prompt (default: empty)")
    p.add_argument("--max-tokens", type=int, default=50, help="Number of tokens to generate")
    p.add_argument("--layers", type=int, nargs="+", default=[25], help="Layers to apply steering (default: [25])")
    return p.parse_args()


def main():
    args = parse_args()

    # Build prompt messages
    messages: List[Message] = [Message(role="user", content=args.prompt or "")]

    # Provider (TransformerLens required for steering)
    provider = get_provider_for_model(args.model, prefer_transformerlens=True)
    if not isinstance(provider, TransformerLensProvider):
        raise RuntimeError("TransformerLens provider not available for this model; cannot steer activations.")

    # Load vectors from file
    if not os.path.exists(args.vector_file):
        raise FileNotFoundError(f"Vector file not found: {args.vector_file}")
    
    data = np.load(args.vector_file)
    
    # Extract vectors by layer from the file
    vectors = {}
    for key in data.files:
        if "__layer_" in key:
            try:
                layer_idx = int(key.split("__layer_")[-1])
                vectors[layer_idx] = data[key]
            except:
                continue
    
    if not vectors:
        print(f"No vectors found in {args.vector_file}")
        return

    # Request
    req = GetTextRequest(context=None, prompt=messages, max_tokens=args.max_tokens, temperature=1.0)

    # Apply steering using specified layers
    layers_to_use = [l for l in args.layers if l in vectors]
    if not layers_to_use:
        print(f"None of specified layers {args.layers} available in vector file")
        return
        
    if args.steering_mode == "add":
        for coeff in args.coeffs:
            for run in range(1):
                resp = provider.generate_text_with_additions(
                    req,
                    vectors=vectors,
                    coefficient=coeff,
                    layers=layers_to_use,
                    hook_variant="pre",
                    positions="all",
                )
                print(f"\n===== Steering mode=add coeff={coeff} layers={layers_to_use} run={run+1} =====\n{resp.txt}\n")
    else:  # project mode
        for run in range(5):
            resp = provider.generate_text_with_projection(
                req,
                vectors=vectors,
                layers=layers_to_use,
                hook_variant="pre",
                positions="all",
            )
            print(f"\n===== Steering mode=project layers={layers_to_use} run={run+1} =====\n{resp.txt}\n")


if __name__ == "__main__":
    main()


