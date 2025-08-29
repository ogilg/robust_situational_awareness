#!/usr/bin/env python3
"""
Benchmark a model on SA tasks with TransformerLens activation steering support.

This script combines the functionality of benchmark_model.py with the ability to:
1. Load vectors from experiments/vectors/ 
2. Apply activation steering using TransformerLens methods (additions, projections)
3. Run tasks with and without interventions to compare performance
4. Save results to CSV and example outputs to JSON

Usage:
  python benchmark_with_steering.py --model llama-3.1-8b-instruct --n 20 --steering-mode add --coefficient 1.0
  python benchmark_with_steering.py --model llama-3.1-8b-instruct --n 20 --steering-mode project --vector-source stages
"""

import os
import sys
import csv
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Tasks
from sad.stages.oversight.task_stages import make_task as make_stages_task
from sad.self_recognition.run import self_recognition_who
from sad.anti_imitation.output_control.task_output_control import make_task as make_output_control_task
from sad.id_leverage.entity_id.task_generic import make_task as make_idlev_generic_task

# Provider wrapper
from provider_wrapper import get_provider_for_model, GetTextRequest, TransformerLensProvider


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _last_user_content(messages) -> str:
    try:
        for msg in reversed(messages):
            if getattr(msg, "role", "user") == "user":
                return getattr(msg, "content", "")
    except Exception:
        pass
    return ""


def _write_csv_row(csv_path: str, row: dict) -> None:
    is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    fieldnames = [
        "model_id",
        "task",
        "variant", 
        "steering_mode",
        "vector_source",
        "coefficient",
        "hook_variant",
        "positions",
        "layers",
        "n_requested",
        "n_processed",
        "invalid",
        "accuracy",
        "runtime_seconds",
        "comment",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_vectors_for_task(
    vectors_dir: str, 
    model: str, 
    task_prefix: str, 
    class_name: str = "correct"
) -> Dict[int, np.ndarray]:
    """Load aggregated vectors for a specific task and class."""
    task_npz = os.path.join(vectors_dir, f"aggregated_vectors_{model}__{task_prefix}.npz")
    
    if not os.path.exists(task_npz):
        print(f"Warning: Vector file not found: {task_npz}")
        return {}
    
    vectors = {}
    try:
        with np.load(task_npz) as data:
            for key in data.files:
                if key.startswith(f"{class_name}__layer_"):
                    layer_idx = int(key.split("_")[-1])
                    vectors[layer_idx] = data[key]
        print(f"Loaded {len(vectors)} vectors for {task_prefix}:{class_name}")
    except Exception as e:
        print(f"Error loading vectors from {task_npz}: {e}")
    
    return vectors


def load_vectors_from_source(
    vectors_dir: str,
    model: str, 
    vector_source: str,
    class_name: str = "correct"
) -> Dict[int, np.ndarray]:
    """Load vectors from a specified source task."""
    task_mapping = {
        "stages": "stages",
        "self_recognition": "self_recognition_who", 
        "output_control": "output_control",
        "id_leverage": "id_leverage_generic"
    }
    
    task_prefix = task_mapping.get(vector_source, vector_source)
    return load_vectors_for_task(vectors_dir, model, task_prefix, class_name)


def _generate_example_with_steering(
    provider_model: str, 
    messages, 
    steering_config: Dict[str, Any],
    *, 
    max_tokens: int = 20, 
    temperature: float = 0.0, 
    fallback_text: str | None = None
) -> dict:
    """Generate example with optional steering applied."""
    provider = get_provider_for_model(provider_model, prefer_transformerlens=True)
    req = GetTextRequest(context=None, prompt=messages, max_tokens=max_tokens, temperature=temperature)
    
    # Apply steering if configured and provider supports it
    if (steering_config.get("steering_mode") != "none" and 
        steering_config.get("vectors") and 
        isinstance(provider, TransformerLensProvider)):
        
        if steering_config["steering_mode"] == "add":
            resp = provider.generate_text_with_additions(
                req,
                vectors=steering_config["vectors"],
                coefficient=steering_config.get("coefficient", 1.0),
                layers=steering_config.get("layers"),
                hook_variant=steering_config.get("hook_variant", "pre"),
                positions=steering_config.get("positions", "all")
            )
        elif steering_config["steering_mode"] == "project":
            resp = provider.generate_text_with_projection(
                req,
                vectors=steering_config["vectors"], 
                layers=steering_config.get("layers"),
                hook_variant=steering_config.get("hook_variant", "pre"),
                positions=steering_config.get("positions", "all")
            )
        else:
            resp = provider.generate_text(req)
    else:
        resp = provider.generate_text(req)
    
    prompt_text = _last_user_content(messages) or (fallback_text or "")
    return {"prompt": prompt_text, "response": getattr(resp, "txt", "")}


def run_stages_oversight(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    steering_config: Dict[str, Any],
    comment: str | None = None
) -> List[Dict[str, Any]]:
    stages_task = make_stages_task()
    stages_variant = stages_task.default_variant
    print(f"[Stages] variant={getattr(stages_variant, 'name', stages_variant)}, n={n_per_task}")
    
    t0 = time.time()
    stages_res = stages_task.run(model=model, variant=stages_variant, n=n_per_task, save=False)
    t1 = time.time()
    
    stages_total = int(stages_res["correct"]) + int(stages_res["incorrect"]) + int(stages_res["invalid"])
    stages_acc = (stages_res["correct"] / stages_total) if stages_total else 0.0
    
    print(f"[Stages] correct={stages_res['correct']} incorrect={stages_res['incorrect']} invalid={stages_res['invalid']} acc={stages_acc:.3f}")
    
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "stages_oversight",
            "variant": getattr(stages_variant, "name", str(stages_variant)),
            "steering_mode": steering_config.get("steering_mode", "none"),
            "vector_source": steering_config.get("vector_source", ""),
            "coefficient": steering_config.get("coefficient", ""),
            "hook_variant": steering_config.get("hook_variant", ""),
            "positions": steering_config.get("positions", ""),
            "layers": str(steering_config.get("layers", "")) if steering_config.get("layers") else "",
            "n_requested": n_per_task,
            "n_processed": stages_total,
            "invalid": stages_res["invalid"],
            "accuracy": f"{stages_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    
    # Collect examples
    examples = []
    count = 0
    for sample in stages_task.iter_samples(model=model, variant=stages_variant, n=examples_per_task):
        messages = stages_task._build_messages(sample, stages_variant)
        ex = _generate_example_with_steering(model, messages, steering_config, max_tokens=5, temperature=0.0)
        examples.append({"task": "stages_oversight", **ex})
        count += 1
        if count >= examples_per_task:
            break
    
    return examples


def run_self_recognition(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    steering_config: Dict[str, Any],
    comment: str | None = None
) -> List[Dict[str, Any]]:
    sr_variant = "plain"
    t0 = time.time()
    sr_results = self_recognition_who.run_evaluation(model=model, variant=sr_variant, n=n_per_task, save=False)
    t1 = time.time()
    
    sr_correct = sum(1 for r in sr_results if r.get("is_correct"))
    sr_invalid = sum(1 for r in sr_results if r.get("invalid"))
    sr_total = len(sr_results)
    sr_acc = (sr_correct / sr_total) if sr_total else 0.0
    
    print(f"[Self-Rec] correct={sr_correct} total={sr_total} acc={sr_acc:.3f}")
    
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "self_recognition_who",
            "variant": sr_variant,
            "steering_mode": steering_config.get("steering_mode", "none"),
            "vector_source": steering_config.get("vector_source", ""),
            "coefficient": steering_config.get("coefficient", ""),
            "hook_variant": steering_config.get("hook_variant", ""),
            "positions": steering_config.get("positions", ""),
            "layers": str(steering_config.get("layers", "")) if steering_config.get("layers") else "",
            "n_requested": n_per_task,
            "n_processed": sr_total,
            "invalid": sr_invalid,
            "accuracy": f"{sr_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    
    # Collect examples
    examples = []
    samples = self_recognition_who._get_samples(model, sr_variant, n=examples_per_task)
    for sample in samples[:examples_per_task]:
        messages = self_recognition_who._build_messages(sample, model, sr_variant)
        ex = _generate_example_with_steering(model, messages, steering_config, max_tokens=5, temperature=0.0)
        examples.append({"task": "self_recognition_who", **ex})
    
    return examples


def run_output_control(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    steering_config: Dict[str, Any],
    comment: str | None = None
) -> List[Dict[str, Any]]:
    oc_task = make_output_control_task()
    oc_variant = oc_task.default_variant
    
    t0 = time.time()
    oc_res = oc_task.run(model=model, variant=oc_variant, n=n_per_task, save=False)
    t1 = time.time()
    
    oc_total = int(oc_res["correct"]) + int(oc_res["incorrect"]) + int(oc_res["invalid"])
    oc_acc = (oc_res["correct"] / oc_total) if oc_total else 0.0
    
    print(f"[Output-Control] correct={oc_res['correct']} total={oc_total} acc={oc_acc:.3f}")
    
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "output_control",
            "variant": getattr(oc_variant, "name", str(oc_variant)),
            "steering_mode": steering_config.get("steering_mode", "none"),
            "vector_source": steering_config.get("vector_source", ""),
            "coefficient": steering_config.get("coefficient", ""),
            "hook_variant": steering_config.get("hook_variant", ""),
            "positions": steering_config.get("positions", ""),
            "layers": str(steering_config.get("layers", "")) if steering_config.get("layers") else "",
            "n_requested": n_per_task,
            "n_processed": oc_total,
            "invalid": oc_res["invalid"],
            "accuracy": f"{oc_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    
    # Collect examples
    examples = []
    count = 0
    for sample in oc_task.iter_samples(model=model, variant=oc_variant, n=examples_per_task):
        ex = _generate_example_with_steering(model, sample.prompt, steering_config, max_tokens=2, temperature=0.0)
        examples.append({"task": "output_control", **ex})
        count += 1
        if count >= examples_per_task:
            break
    
    return examples


def run_id_leverage(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    steering_config: Dict[str, Any],
    comment: str | None = None
) -> List[Dict[str, Any]]:
    idlev_task = make_idlev_generic_task()
    id_variant = idlev_task.default_variant
    
    t0 = time.time()
    id_res = idlev_task.run(model=model, variant=id_variant, n=n_per_task, save=False)
    t1 = time.time()
    
    id_total = int(id_res["correct"]) + int(id_res["incorrect"]) + int(id_res["invalid"])
    id_acc = (id_res["correct"] / id_total) if id_total else 0.0
    
    print(f"[ID-Leverage] correct={id_res['correct']} total={id_total} acc={id_acc:.3f}")
    
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "id_leverage_generic",
            "variant": getattr(id_variant, "name", str(id_variant)),
            "steering_mode": steering_config.get("steering_mode", "none"),
            "vector_source": steering_config.get("vector_source", ""),
            "coefficient": steering_config.get("coefficient", ""),
            "hook_variant": steering_config.get("hook_variant", ""),
            "positions": steering_config.get("positions", ""),
            "layers": str(steering_config.get("layers", "")) if steering_config.get("layers") else "",
            "n_requested": n_per_task,
            "n_processed": id_total,
            "invalid": id_res["invalid"],
            "accuracy": f"{id_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    
    # Collect examples
    examples = []
    count = 0
    for sample in idlev_task.iter_samples(model=model, variant=id_variant, n=examples_per_task):
        messages = sample["messages"]
        fallback = sample.get("request_text", "")
        ex = _generate_example_with_steering(model, messages, steering_config, max_tokens=30, temperature=0.0, fallback_text=fallback)
        examples.append({"task": "id_leverage_generic", **ex})
        count += 1
        if count >= examples_per_task:
            break
    
    return examples


def run_benchmark_with_steering(
    *,
    model: str,
    out_dir: str,
    vectors_dir: str,
    n_per_task: int,
    examples_per_task: int,
    steering_mode: str = "none",
    vector_source: str = "",
    coefficient: float = 1.0,
    hook_variant: str = "pre",
    positions: str = "all",
    layers: Optional[List[int]] = None,
    class_name: str = "correct",
    tasks: Optional[List[str]] = None,
    comment: str | None = None,
) -> None:
    _ensure_dir(out_dir)
    ts = int(time.time())
    suffix = f"_{steering_mode}" if steering_mode != "none" else ""
    if vector_source:
        suffix += f"_{vector_source}"
    examples_out = os.path.join(out_dir, f"examples_{model}{suffix}_{ts}.json")
    csv_out = os.path.join(out_dir, f"scores_{model}_steering.csv")

    # Prepare steering configuration
    steering_config = {
        "steering_mode": steering_mode,
        "vector_source": vector_source,
        "coefficient": coefficient,
        "hook_variant": hook_variant,
        "positions": positions,
        "layers": layers,
        "vectors": {}
    }
    
    # Load vectors if steering is enabled
    if steering_mode != "none" and vector_source:
        steering_config["vectors"] = load_vectors_from_source(vectors_dir, model, vector_source, class_name)
        if not steering_config["vectors"]:
            print(f"Warning: No vectors loaded for {vector_source}. Running without steering.")
            steering_config["steering_mode"] = "none"
        else:
            print(f"Loaded {len(steering_config['vectors'])} vectors from {vector_source} for steering")

    all_examples: List[Dict[str, Any]] = []
    task_list = tasks or ["stages", "self_recognition", "output_control", "id_leverage"]

    # Run each task
    if "stages" in task_list:
        all_examples.extend(run_stages_oversight(
            model=model,
            csv_out=csv_out,
            n_per_task=n_per_task,
            examples_per_task=examples_per_task,
            steering_config=steering_config,
            comment=comment,
        ))

    if "self_recognition" in task_list:
        all_examples.extend(run_self_recognition(
            model=model,
            csv_out=csv_out,
            n_per_task=n_per_task,
            examples_per_task=examples_per_task,
            steering_config=steering_config,
            comment=comment,
        ))

    if "output_control" in task_list:
        all_examples.extend(run_output_control(
            model=model,
            csv_out=csv_out,
            n_per_task=n_per_task,
            examples_per_task=examples_per_task,
            steering_config=steering_config,
            comment=comment,
        ))

    if "id_leverage" in task_list:
        all_examples.extend(run_id_leverage(
            model=model,
            csv_out=csv_out,
            n_per_task=n_per_task,
            examples_per_task=examples_per_task,
            steering_config=steering_config,
            comment=comment,
        ))

    # Write examples JSON
    if all_examples:
        with open(examples_out, "w") as f:
            json.dump(
                {
                    "model": model,
                    "timestamp": ts,
                    "steering_config": {k: v for k, v in steering_config.items() if k != "vectors"},
                    "examples": all_examples,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a model on SA tasks with optional TransformerLens steering.")
    parser.add_argument("--model", required=True, help="Model id (as expected by provider_wrapper)")
    parser.add_argument("--n", type=int, default=10, help="Number of samples per task")
    parser.add_argument("--examples", type=int, default=5, help="Number of examples per task to save to JSON")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "experiments", "results"), help="Output directory")
    parser.add_argument("--vectors-dir", default=os.path.join(ROOT, "experiments", "vectors"), help="Vectors directory")
    
    # Steering parameters
    parser.add_argument("--steering-mode", choices=["none", "add", "project"], default="none", 
                       help="Activation steering mode")
    parser.add_argument("--vector-source", choices=["stages", "self_recognition", "output_control", "id_leverage"], 
                       default="", help="Source task for vectors")
    parser.add_argument("--coefficient", type=float, default=1.0, help="Steering coefficient (for add mode)")
    parser.add_argument("--hook-variant", choices=["pre", "mid", "post"], default="pre", 
                       help="Hook variant for steering")
    parser.add_argument("--positions", choices=["all", "last"], default="all", 
                       help="Positions to apply steering")
    parser.add_argument("--layers", type=int, nargs="+", help="Specific layers to apply steering to")
    parser.add_argument("--class-name", choices=["correct", "incorrect"], default="correct",
                       help="Class of vectors to use for steering")
    
    # Task selection
    parser.add_argument("--tasks", nargs="+", 
                       choices=["stages", "self_recognition", "output_control", "id_leverage"],
                       default=["stages", "self_recognition", "output_control", "id_leverage"],
                       help="Tasks to run")
    parser.add_argument("--comment", default=None, help="Optional comment to include in CSV rows")
    
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark_with_steering(
        model=args.model,
        out_dir=args.out_dir,
        vectors_dir=args.vectors_dir,
        n_per_task=args.n,
        examples_per_task=args.examples,
        steering_mode=args.steering_mode,
        vector_source=args.vector_source,
        coefficient=args.coefficient,
        hook_variant=args.hook_variant,
        positions=args.positions,
        layers=args.layers,
        class_name=args.class_name,
        tasks=args.tasks,
        comment=args.comment,
    )
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()