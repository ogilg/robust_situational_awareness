#!/usr/bin/env python3
"""
Benchmark a model on the four tasks and also store first-token residual activations.

This mirrors experiments/benchmark_model.py for scoring, but calls
Task.run_with_collected_residuals so that vectors are saved under each task's
vectors/<model>/<variant>/<classification>/ directory.
"""

import os
import sys
import time
from typing import Any
import numpy as _np

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


def _to_numpy_cpu(x):
    try:
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            return x.detach().cpu().numpy()
        return _np.asarray(x)
    except Exception:
        return _np.asarray(x)


def _last_user_content(messages) -> str:
    try:
        for msg in reversed(messages):
            if getattr(msg, "role", "user") == "user":
                return getattr(msg, "content", "")
    except Exception:
        pass
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
            msgs = task._build_messages(sample["body"], variant_str)  # noqa: SLF001
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


def _aggregate_vectors_for_task(task, *, model: str, variant, n: int, task_name: str, examples_max: int) -> tuple[dict, dict, dict, list[dict]]:
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
            examples.append({"task": task_name, "prompt": prompt_preview, "response": txt})
        # Only aggregate if residuals present and sample is not invalid
        if residuals is not None and int(result.get("invalid", 0)) == 0:
            bucket = "correct" if int(result.get("correct", 0)) == 1 else "incorrect"
            count_by_class[bucket] += 1
            # Accumulate per layer
            for layer_idx, vec in residuals.items():
                arr = _to_numpy_cpu(vec).reshape(-1)
                if layer_idx not in sum_by_class[bucket]:
                    sum_by_class[bucket][layer_idx] = arr.copy()
                else:
                    sum_by_class[bucket][layer_idx] += arr
        i += 1
        if n is not None and i >= n:
            break

    return ({"correct": correct, "incorrect": incorrect, "invalid": invalid}, sum_by_class, count_by_class, examples)


def run_benchmark_with_vectors(
    *,
    model: str,
    out_dir: str,
    n_per_task: int,
    comment: str | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_out = os.path.join(out_dir, f"scores_{model}.csv")
    ts = int(time.time())
    examples_out = os.path.join(out_dir, f"examples_{model}_{ts}.json")
    examples: list[dict[str, str]] = []

    # ----- Stages oversight -----
    stages_data_path = os.path.join(ROOT, "tests", "stages_llama_test.jsonl")
    stages_task = make_stages_task(data_path=stages_data_path)
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
            "invalid": stages_res["invalid"],
            "accuracy": f"{stages_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    # Collect example prompts/responses (like benchmark_model)
    examples.extend(stages_examples)

    # ----- Self-recognition (WHO) -----
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
            "invalid": sr_res["invalid"],
            "accuracy": f"{sr_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    # Collect examples
    examples.extend(sr_examples)

    # ----- Output control -----
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
            "invalid": oc_res["invalid"],
            "accuracy": f"{oc_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    # Collect examples
    examples.extend(oc_examples)

    # ----- ID leverage (generic) -----
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
            "invalid": id_res["invalid"],
            "accuracy": f"{id_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    # Collect examples
    examples.extend(id_examples)

    # ----- Save aggregated vectors once at the end -----
    agg_npz = os.path.join(out_dir, f"aggregated_vectors_{model}.npz")
    agg_counts_json = os.path.join(out_dir, f"aggregated_vectors_{model}.counts.json")
    payload = {}

    def add_task_payload(prefix: str, sums: dict, counts: dict):
        for cls_name, layer_dict in sums.items():
            for layer_idx, arr in layer_dict.items():
                key = f"{prefix}__{cls_name}__layer_{layer_idx}"
                payload[key] = arr

    add_task_payload("stages", stages_sums, stages_counts)
    add_task_payload("self_recognition_who", sr_sums, sr_counts)
    add_task_payload("output_control", oc_sums, oc_counts)
    add_task_payload("id_leverage_generic", id_sums, id_counts)

    # Save numeric arrays
    if payload:
        _np.savez(agg_npz, **payload)
    # Save counts for averaging downstream
    import json as _json
    with open(agg_counts_json, "w") as f:
        _json.dump(
            {
                "stages": stages_counts,
                "self_recognition_who": sr_counts,
                "output_control": oc_counts,
                "id_leverage_generic": id_counts,
            },
            f,
            indent=2,
        )

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
    parser.add_argument("--comment", default=None, help="Optional comment to include in CSV rows")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark_with_vectors(
        model=args.model,
        out_dir=args.out_dir,
        n_per_task=args.n,
        comment=args.comment,
    )
    print(f"Wrote results to {args.out_dir}. Vectors saved under each task's vectors/ directory.")


if __name__ == "__main__":
    main()


