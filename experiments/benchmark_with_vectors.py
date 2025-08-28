#!/usr/bin/env python3
"""
Benchmark a model on SA tasks and store first-token residual activations.

Notes:
- This mirrors experiments/benchmark_model.py for scoring, but also collects
  per-task aggregated first-token residuals via provider hooks.
- Aggregated vectors are saved to experiments/vectors immediately after each
  selected task completes.
"""

import os
import sys
import time
from typing import Any
import numpy as _np
import json as _json

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Reuse CSV helper from the benchmark script
from experiments.benchmark_model import _write_csv_row  # noqa: E402

# Tasks
from sad.stages.oversight.task_stages import make_task as make_stages_task  # noqa: E402
from sad.self_recognition.run import self_recognition_who  # noqa: E402
from sad.anti_imitation.output_control.task_output_control import make_task as make_output_control_task  # noqa: E402
from sad.id_leverage.entity_id.task_generic import make_task as make_idlev_generic_task  # noqa: E402

from provider_wrapper import clear_gpu_memory

def _to_numpy_cpu(x):
    try:
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            return x.detach().cpu().numpy()
        return _np.asarray(x)
    except Exception:
        return _np.asarray(x)


def _last_user_content(messages) -> str:
    try:
        fallback = None
        for msg in reversed(messages):
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            if role == "user":
                return content
            if fallback is None and content:
                fallback = content
        return fallback or ""
    except Exception:
        return ""


def _build_prompt_preview(task, sample, *, model: str, variant_str: str) -> str:
    # Output control: provider_wrapper Sample with .prompt
    if hasattr(sample, "prompt"):
        return _last_user_content(getattr(sample, "prompt", []))
    # ID leverage: dict with messages
    if isinstance(sample, dict) and "messages" in sample:
        return _last_user_content(sample["messages"]) or sample.get("request_text", "")
    # Stages: has body
    if hasattr(task, "_build_messages") and isinstance(sample, dict) and "body" in sample:
        try:
            msgs = task._build_messages(sample, variant_str)  # noqa: SLF001
            return _last_user_content(msgs)
        except Exception:
            return sample.get("body", "")
    # Self-recognition: builder takes (sample, model, variant)
    if hasattr(task, "_build_messages"):
        try:
            msgs = task._build_messages(sample, model, variant_str)  # noqa: SLF001
            return _last_user_content(msgs)
        except Exception:
            return ""
    return ""


def _aggregate_vectors_for_task(
    task,
    *,
    model: str,
    variant,
    n: int,
    task_name: str,
    examples_max: int,
) -> tuple[dict, dict, dict, list[dict]]:
    """
    Iterate samples via task.iter_samples, call evaluate_and_capture_sample,
    and maintain running sums per class (correct/incorrect) and per layer.

    Returns (results_counts, sum_by_class, count_by_class)
    - results_counts: {"correct": int, "incorrect": int, "invalid": int}
    - sum_by_class: {class_name: {layer_idx: np.array(d_model,)}}
    - count_by_class: {class_name: int}
    """
    correct = 0
    incorrect = 0
    invalid = 0
    sum_by_class: dict[str, dict[int, _np.ndarray]] = {"correct": {}, "incorrect": {}}
    count_by_class: dict[str, int] = {"correct": 0, "incorrect": 0}
    examples: list[dict] = []

    # Normalize variant to string name if a Variant object was passed
    variant_str = getattr(variant, "name", variant)

    i = 0
    for sample in task.iter_samples(model=model, variant=variant_str, n=n):
        result, residuals, aux_meta = task.evaluate_and_capture_sample(model=model, sample=sample, variant=variant_str)
        correct += int(result.get("correct", 0))
        incorrect += int(result.get("incorrect", 0))
        invalid += int(result.get("invalid", 0))
        # Capture up to examples_max transformer-lens texts from the same forward pass
        if len(examples) < examples_max:
            txt = ""
            if isinstance(aux_meta, dict):
                txt = str(aux_meta.get("txt", ""))
            prompt_preview = _build_prompt_preview(task, sample, model=model, variant_str=variant_str)
            # Derive human-readable classification
            classification = "invalid" if int(result.get("invalid", 0)) == 1 else ("correct" if int(result.get("correct", 0)) == 1 else "incorrect")
            examples.append({
                "task": task_name,
                "prompt": prompt_preview,
                "response": txt,
                "classification": classification,
            })
        # Only aggregate if residuals have at least one captured layer and sample is not invalid
        if residuals is not None and int(result.get("invalid", 0)) == 0:
            bucket = "correct" if int(result.get("correct", 0)) == 1 else "incorrect"
            aggregated_any = False
            # Accumulate per layer
            for layer_idx, vec in (residuals.items() if hasattr(residuals, "items") else []):
                arr = _to_numpy_cpu(vec).reshape(-1)
                if layer_idx not in sum_by_class[bucket]:
                    sum_by_class[bucket][layer_idx] = arr.copy()
                else:
                    sum_by_class[bucket][layer_idx] += arr
                aggregated_any = True
            if aggregated_any:
                # Only keep aggregate counts; do not save per-sample vectors
                count_by_class[bucket] += 1
        i += 1
        if n is not None and i >= n:
            break  

    return ({"correct": correct, "incorrect": incorrect, "invalid": invalid}, sum_by_class, count_by_class, examples)


def _save_aggregated_for_task(*, vectors_dir: str, model: str, task_prefix: str, sums: dict, counts: dict) -> None:
    """
    Merge-and-save aggregated vectors and counts for a single task immediately.

    - Aggregated vectors now live per-task at experiments/vectors/aggregated_vectors_{model}__{task}.npz
      with keys like 'correct__layer_{i}'.
    - Counts per class live in aggregated_vectors_{model}.counts.json as:
      { task_prefix: {"correct": int, "incorrect": int}, ... }
    """
    os.makedirs(vectors_dir, exist_ok=True)
    task_npz = os.path.join(vectors_dir, f"aggregated_vectors_{model}__{task_prefix}.npz")
    agg_counts_json = os.path.join(vectors_dir, f"aggregated_vectors_{model}.counts.json")

    # Build additions for this task
    additions: dict[str, _np.ndarray] = {}
    for cls_name, layer_dict in sums.items():
        for layer_idx, arr in layer_dict.items():
            key = f"{cls_name}__layer_{layer_idx}"
            additions[key] = arr

    # Merge with existing NPZ if present
    merged_payload: dict[str, _np.ndarray] = {}
    if os.path.exists(task_npz):
        with _np.load(task_npz) as existing:
            for k in existing.files:
                merged_payload[k] = existing[k].copy()

    merged_payload.update(additions)
    if merged_payload:
        _np.savez(task_npz, **merged_payload)
        # Confirm by reloading
        with _np.load(task_npz) as chk:
            pass

    # Merge counts JSON
    import json as _json
    counts_data = {}
    if os.path.exists(agg_counts_json):
        try:
            with open(agg_counts_json, "r") as f:
                counts_data = _json.load(f) or {}
        except Exception:
            counts_data = {}
    counts_data[task_prefix] = counts
    with open(agg_counts_json, "w") as f:
        _json.dump(counts_data, f, indent=2)



def run_task_stages(*, model: str, csv_out: str, vectors_dir: str, n_per_task: int, comment: str | None) -> list[dict[str, str]]:
    stages_task = make_stages_task()
    stages_variant = stages_task.default_variant
    t0 = time.time()
    stages_res, stages_sums, stages_counts, stages_examples = _aggregate_vectors_for_task(
        stages_task, model=model, variant=stages_variant, n=n_per_task, task_name="stages_oversight", examples_max=max(1, min(5, n_per_task))
    )
    t1 = time.time()
    stages_total = int(stages_res["correct"]) + int(stages_res["incorrect"]) + int(stages_res["invalid"])
    stages_acc = (stages_res["correct"] / stages_total) if stages_total else 0.0
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "stages_oversight",
            "variant": getattr(stages_variant, "name", str(stages_variant)),
            "n_requested": n_per_task,
            "n_processed": stages_total,
            "invalid": stages_res["invalid"],
            "accuracy": f"{stages_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    # Save this task's aggregates immediately
    _save_aggregated_for_task(vectors_dir=vectors_dir, model=model, task_prefix="stages", sums=stages_sums, counts=stages_counts)
    return stages_examples


def run_task_self_recognition(*, model: str, csv_out: str, vectors_dir: str, n_per_task: int, comment: str | None) -> list[dict[str, str]]:
    sr_variant = "plain"
    sr_variant_use = self_recognition_who.default_variant if hasattr(self_recognition_who, "default_variant") else sr_variant
    t0 = time.time()
    sr_res, sr_sums, sr_counts, sr_examples = _aggregate_vectors_for_task(
        self_recognition_who, model=model, variant=sr_variant_use, n=n_per_task, task_name="self_recognition_who", examples_max=max(1, min(5, n_per_task))
    )
    t1 = time.time()
    sr_total = int(sr_res["correct"]) + int(sr_res["incorrect"]) + int(sr_res["invalid"])
    sr_acc = (sr_res["correct"] / sr_total) if sr_total else 0.0
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "self_recognition_who",
            "variant": sr_variant,
            "n_requested": n_per_task,
            "n_processed": sr_total,
            "invalid": sr_res["invalid"],
            "accuracy": f"{sr_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    _save_aggregated_for_task(vectors_dir=vectors_dir, model=model, task_prefix="self_recognition_who", sums=sr_sums, counts=sr_counts)
    return sr_examples


def run_task_output_control(*, model: str, csv_out: str, vectors_dir: str, n_per_task: int, comment: str | None) -> list[dict[str, str]]:
    oc_task = make_output_control_task()
    oc_variant = oc_task.default_variant
    t0 = time.time()
    oc_res, oc_sums, oc_counts, oc_examples = _aggregate_vectors_for_task(
        oc_task, model=model, variant=oc_variant, n=n_per_task, task_name="output_control", examples_max=max(1, min(5, n_per_task))
    )
    t1 = time.time()
    oc_total = int(oc_res["correct"]) + int(oc_res["incorrect"]) + int(oc_res["invalid"])
    oc_acc = (oc_res["correct"] / oc_total) if oc_total else 0.0
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "output_control",
            "variant": getattr(oc_variant, "name", str(oc_variant)),
            "n_requested": n_per_task,
            "n_processed": oc_total,
            "invalid": oc_res["invalid"],
            "accuracy": f"{oc_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    _save_aggregated_for_task(vectors_dir=vectors_dir, model=model, task_prefix="output_control", sums=oc_sums, counts=oc_counts)
    return oc_examples


def run_task_id_leverage(*, model: str, csv_out: str, vectors_dir: str, n_per_task: int, comment: str | None) -> list[dict[str, str]]:
    idlev_task = make_idlev_generic_task()
    id_variant = idlev_task.default_variant
    t0 = time.time()
    id_res, id_sums, id_counts, id_examples = _aggregate_vectors_for_task(
        idlev_task, model=model, variant=id_variant, n=n_per_task, task_name="id_leverage_generic", examples_max=max(1, min(5, n_per_task))
    )
    t1 = time.time()
    id_total = int(id_res["correct"]) + int(id_res["incorrect"]) + int(id_res["invalid"])
    id_acc = (id_res["correct"] / id_total) if id_total else 0.0
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "id_leverage_generic",
            "variant": getattr(id_variant, "name", str(id_variant)),
            "n_requested": n_per_task,
            "n_processed": id_total,
            "invalid": id_res["invalid"],
            "accuracy": f"{id_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    _save_aggregated_for_task(vectors_dir=vectors_dir, model=model, task_prefix="id_leverage_generic", sums=id_sums, counts=id_counts)
    return id_examples


def run_benchmark_with_vectors(
    *,
    model: str,
    out_dir: str,
    n_per_task: int,
    tasks: list[str] | None = None,
    comment: str | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_out = os.path.join(out_dir, f"scores_{model}.csv")
    # Place vectors under the parent experiments directory, not inside results
    parent_experiments_dir = os.path.dirname(out_dir)
    vectors_dir = os.path.join(parent_experiments_dir, "vectors")
    os.makedirs(vectors_dir, exist_ok=True)
    ts = int(time.time())
    examples_out = os.path.join(out_dir, f"examples_{model}_{ts}.json")
    examples: list[dict[str, str]] = []
    # Task selection
    all_tasks = [
        "stages",
        "self_recognition_who",
        "output_control",
        "id_leverage_generic",
    ]
    to_run = tasks or all_tasks

    if "stages" in to_run:
        examples.extend(run_task_stages(model=model, csv_out=csv_out, vectors_dir=vectors_dir, n_per_task=n_per_task, comment=comment))
        clear_gpu_memory()

    if "self_recognition_who" in to_run:
        examples.extend(run_task_self_recognition(model=model, csv_out=csv_out, vectors_dir=vectors_dir, n_per_task=n_per_task, comment=comment))
        clear_gpu_memory()

    if "output_control" in to_run:
        examples.extend(run_task_output_control(model=model, csv_out=csv_out, vectors_dir=vectors_dir, n_per_task=n_per_task, comment=comment))
        clear_gpu_memory()

    if "id_leverage_generic" in to_run:
        examples.extend(run_task_id_leverage(model=model, csv_out=csv_out, vectors_dir=vectors_dir, n_per_task=n_per_task, comment=comment))
        clear_gpu_memory()

    # Save examples JSON (like benchmark_model)
    if examples:
        with open(examples_out, "w") as f:
            _json.dump(
                {
                    "model": model,
                    "timestamp": ts,
                    "examples": examples,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark a model and store activation vectors.")
    parser.add_argument("--model", required=True, help="Model id (as expected by provider_wrapper)")
    parser.add_argument("--n", type=int, default=10, help="Number of samples per task")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "experiments", "results"), help="Output directory")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["stages", "self_recognition_who", "output_control", "id_leverage_generic"],
        default=["stages", "self_recognition_who", "output_control", "id_leverage_generic"],
        help="Subset of tasks to run",
    )
    parser.add_argument("--comment", default=None, help="Optional comment to include in CSV rows")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark_with_vectors(
        model=args.model,
        out_dir=args.out_dir,
        n_per_task=args.n,
        tasks=args.tasks,
        comment=args.comment,
    )
    # Note: vectors are aggregated to experiments/vectors
    print(f"Wrote results to {args.out_dir}. Aggregated vectors saved under {os.path.join(os.path.dirname(args.out_dir), 'vectors')}.")


if __name__ == "__main__":
    main()


