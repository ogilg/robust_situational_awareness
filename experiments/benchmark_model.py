#!/usr/bin/env python3
"""
Benchmark a model on four tasks and save:
- Per-task scores to a CSV
- Example prompt/response pairs to a JSON file

Tasks covered:
- Stages oversight
- Self-recognition (WHO)
- Output control
- ID leverage (generic)
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import Any

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
from provider_wrapper import get_provider_for_model, GetTextRequest


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


def _generate_example(provider_model: str, messages, *, max_tokens: int = 10, temperature: float = 0.0, fallback_text: str | None = None) -> dict:
    provider = get_provider_for_model(provider_model)
    req = GetTextRequest(context=None, prompt=messages, max_tokens=max_tokens, temperature=temperature)
    resp = provider.generate_text(req)
    prompt_text = _last_user_content(messages) or (fallback_text or "")
    return {"prompt": prompt_text, "response": getattr(resp, "txt", "")}


def _write_csv_row(csv_path: str, row: dict) -> None:
    is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    fieldnames = [
        "model_id",
        "task",
        "variant",
        "n_requested",
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


def run_stages_oversight(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    comment: str | None = None
) -> list[dict[str, Any]]:
    # Prefer YAML batches under sad/stages/oversight/structs/batch/*.yaml
    stages_data_path = os.path.join(ROOT, "sad", "stages", "oversight", "structs", "batch", "test_oversight.yaml")
    stages_task = make_stages_task(data_path=stages_data_path)
    stages_variant = stages_task.default_variant
    print(f"[Stages] data_path={stages_data_path}, variant={getattr(stages_variant, 'name', stages_variant)}, n={n_per_task}")
    t0 = time.time()
    stages_res = stages_task.run(model=model, variant=stages_variant, n=n_per_task, save=False)
    t1 = time.time()
    stages_total = int(stages_res["correct"]) + int(stages_res["incorrect"]) + int(stages_res["invalid"])
    stages_acc = (stages_res["correct"] / stages_total) if stages_total else 0.0
    print(f"[Stages] summary correct={stages_res['correct']} incorrect={stages_res['incorrect']} invalid={stages_res['invalid']} total={stages_total} time={t1 - t0:.3f}s")
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
    # Collect examples
    examples = []
    count = 0
    for sample in stages_task.iter_samples(model=model, variant=stages_variant, n=examples_per_task):
        messages = stages_task._build_messages(sample, stages_variant)  # noqa: SLF001
        ex = _generate_example(model, messages, max_tokens=5, temperature=0.0)
        examples.append({"task": "stages_oversight", **ex})
        count += 1
        if count >= examples_per_task:
            break
    print(f"[Stages] examples_collected={count}")
    return examples


def run_self_recognition(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    comment: str | None = None
) -> list[dict[str, Any]]:
    examples = []
    sr_variant = "plain"
    t0 = time.time()
    sr_results = self_recognition_who.run_evaluation(model=model, variant=sr_variant, n=n_per_task, save=False)
    t1 = time.time()
    sr_correct = sum(1 for r in sr_results if r.get("is_correct"))
    sr_invalid = sum(1 for r in sr_results if r.get("invalid"))
    sr_total = len(sr_results)
    sr_acc = (sr_correct / sr_total) if sr_total else 0.0
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "self_recognition_who",
            "variant": sr_variant,
            "n_requested": n_per_task,
            "invalid": sr_invalid,
            "accuracy": f"{sr_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    # Collect examples
    samples = self_recognition_who._get_samples(model, sr_variant, n=examples_per_task)  # noqa: SLF001
    for sample in samples[:examples_per_task]:
        messages = self_recognition_who._build_messages(sample, model, sr_variant)  # noqa: SLF001
        ex = _generate_example(model, messages, max_tokens=5, temperature=0.0)
        examples.append({"task": "self_recognition_who", **ex})
    return examples


def run_output_control(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    comment: str | None = None
) -> list[dict[str, Any]]:
    examples = []
    oc_task = make_output_control_task()
    oc_variant = oc_task.default_variant
    t0 = time.time()
    oc_res = oc_task.run(model=model, variant=oc_variant, n=n_per_task, save=False)
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
    count = 0
    for sample in oc_task.iter_samples(model=model, variant=oc_variant, n=examples_per_task):
        ex = _generate_example(model, sample.prompt, max_tokens=5, temperature=0.0)
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
    comment: str | None = None
) -> list[dict[str, Any]]:
    examples = []
    idlev_task = make_idlev_generic_task()
    id_variant = idlev_task.default_variant
    t0 = time.time()
    id_res = idlev_task.run(model=model, variant=id_variant, n=n_per_task, save=False)
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
    count = 0
    for sample in idlev_task.iter_samples(model=model, variant=id_variant, n=examples_per_task):
        messages = sample["messages"]
        fallback = sample.get("request_text", "")
        ex = _generate_example(model, messages, max_tokens=30, temperature=0.0, fallback_text=fallback)
        examples.append({"task": "id_leverage_generic", **ex})
        count += 1
        if count >= examples_per_task:
            break
    return examples


def run_benchmark(
    *,
    model: str,
    out_dir: str,
    n_per_task: int,
    examples_per_task: int,
    comment: str | None = None,
) -> None:
    _ensure_dir(out_dir)
    ts = int(time.time())
    examples_out = os.path.join(out_dir, f"examples_{model}_{ts}.json")
    csv_out = os.path.join(out_dir, f"scores_{model}.csv")

    all_examples: list[dict[str, Any]] = []

    # Run each task (comment out to skip)
    all_examples.extend(run_stages_oversight(
        model=model,
        csv_out=csv_out,
        n_per_task=n_per_task,
        examples_per_task=examples_per_task,
        comment=comment,
    ))
    all_examples.extend(run_self_recognition(
        model=model,
        csv_out=csv_out,
        n_per_task=n_per_task,
        examples_per_task=examples_per_task,
        comment=comment,
    ))
    all_examples.extend(run_output_control(
        model=model,
        csv_out=csv_out,
        n_per_task=n_per_task,
        examples_per_task=examples_per_task,
        comment=comment,
    ))
    all_examples.extend(run_id_leverage(
        model=model,
        csv_out=csv_out,
        n_per_task=n_per_task,
        examples_per_task=examples_per_task,
        comment=comment,
    ))

    # ----- Write examples JSON -----
    with open(examples_out, "w") as f:
        json.dump(
            {
                "model": model,
                "timestamp": ts,
                "examples": all_examples,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark a model on SA tasks and save results.")
    parser.add_argument("--model", required=True, help="Model id (as expected by provider_wrapper)")
    parser.add_argument("--n", type=int, default=10, help="Number of samples per task")
    parser.add_argument("--examples", type=int, default=5, help="Number of examples per task to save to JSON")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "experiments", "results"), help="Output directory")
    parser.add_argument("--comment", default=None, help="Optional comment to include in CSV rows")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(
        model=args.model,
        out_dir=args.out_dir,
        n_per_task=args.n,
        examples_per_task=args.examples,
        comment=args.comment,
    )
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()


