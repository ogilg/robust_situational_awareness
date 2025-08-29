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
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "user":
                return msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    except Exception:
        pass
    return ""


def _generate_example(provider_model: str, messages, *, max_tokens: int = 10, temperature: float = 0.0, fallback_text: str | None = None) -> dict:
    provider = get_provider_for_model(provider_model, prefer_transformerlens=False)
    req = GetTextRequest(context=None, prompt=messages, max_tokens=max_tokens, temperature=temperature)
    resp = provider.generate_text(req)
    prompt_text = _last_user_content(messages) or (fallback_text or "")
    return {"prompt": prompt_text, "response": getattr(resp, "txt", "")}


def _classification_from_result(res: dict) -> str:
    try:
        inv = int(res.get("invalid", 0))
        if inv == 1:
            return "invalid"
        cor = int(res.get("correct", 0))
        return "correct" if cor == 1 else "incorrect"
    except Exception:
        return ""


def _write_csv_row(csv_path: str, row: dict) -> None:
    is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    fieldnames = [
        "model_id",
        "task",
        "variant",
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


def run_stages_oversight(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    variant: str,
    comment: str | None = None
) -> list[dict[str, Any]]:
    stages_task = make_stages_task()
    stages_variant = stages_task.default_variant
    # Allow CLI override of variant
    if isinstance(variant, str):
        if variant.lower() in (getattr(stages_task.default_variant, "name", "plain"), "plain"):
            pass
        elif variant.lower() == "sp":
            stages_variant = type(stages_variant)(name="sp", pretty_name="SP", in_sad_score=False, in_sp_sad_score=True)
    print(f"[Stages] data_path={getattr(stages_task, 'data_path', '')}, variant={getattr(stages_variant, 'name', stages_variant)}, n={n_per_task}")
    provider = get_provider_for_model(model, prefer_transformerlens=False)
    correct = 0
    incorrect = 0
    invalid = 0
    examples: list[dict[str, Any]] = []
    t0 = time.time()
    i = 0
    for sample in stages_task.iter_samples(model=model, variant=getattr(stages_variant, 'name', stages_variant), n=n_per_task):
        messages = stages_task._build_messages(sample, getattr(stages_variant, 'name', stages_variant))  # noqa: SLF001
        req = GetTextRequest(context=None, prompt=messages, max_tokens=3, temperature=0.0)
        resp = provider.generate_text(req)
        txt = getattr(resp, "txt", "")
        expected = stages_task._infer_expected_label(sample)  # noqa: SLF001
        res = stages_task._score_from_text(txt, expected_label=expected)  # noqa: SLF001
        correct += int(res.get("correct", 0))
        incorrect += int(res.get("incorrect", 0))
        invalid += int(res.get("invalid", 0))
        if len(examples) < examples_per_task:
            examples.append({
                "task": "stages_oversight",
                "prompt": _last_user_content(messages),
                "response": txt,
                "classification": _classification_from_result(res),
            })
        i += 1
        if i >= n_per_task:
            break
    t1 = time.time()
    stages_total = correct + incorrect + invalid
    stages_acc = (correct / stages_total) if stages_total else 0.0
    print(f"[Stages] summary correct={correct} incorrect={incorrect} invalid={invalid} total={stages_total} time={t1 - t0:.3f}s")
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "stages_oversight",
            "variant": getattr(stages_variant, "name", str(stages_variant)),
            "n_requested": n_per_task,
            "n_processed": stages_total,
            "invalid": invalid,
            "accuracy": f"{stages_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    print(f"[Stages] examples_collected={len(examples)}")
    return examples


def run_self_recognition(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    variant: str,
    comment: str | None = None
) -> list[dict[str, Any]]:
    examples = []
    sr_variant = variant or "plain"
    provider = get_provider_for_model(model, prefer_transformerlens=False)
    correct = 0
    invalid = 0
    total = 0
    t0 = time.time()
    # Use the same iterator as the task uses
    samples = self_recognition_who._get_samples(model, sr_variant, n=n_per_task)  # noqa: SLF001
    for sample in samples:
        messages = self_recognition_who._build_messages(sample, sr_variant, model)  # noqa: SLF001
        req = GetTextRequest(context=None, prompt=messages, max_tokens=5, temperature=0.0)
        resp = provider.generate_text(req)
        txt = getattr(resp, "txt", "")
        res = self_recognition_who._score_from_text(txt, sample)  # noqa: SLF001
        correct += int(res.get("correct", 0))
        invalid += int(res.get("invalid", 0))
        if len(examples) < examples_per_task:
            examples.append({
                "task": "self_recognition_who",
                "prompt": _last_user_content(messages),
                "response": txt,
                "classification": _classification_from_result(res),
            })
        total += 1
        if total >= n_per_task:
            break
    t1 = time.time()
    acc = (correct / total) if total else 0.0
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "self_recognition_who",
            "variant": sr_variant,
            "n_requested": n_per_task,
            "n_processed": total,
            "invalid": invalid,
            "accuracy": f"{acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    return examples


def run_output_control(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    variant: str,
    comment: str | None = None
) -> list[dict[str, Any]]:
    examples = []
    oc_task = make_output_control_task()
    oc_variant_str = variant or getattr(oc_task.default_variant, 'name', 'plain')
    t0 = time.time()
    correct = 0
    incorrect = 0
    invalid = 0
    # Single pass: score and collect up to examples_per_task
    for sample in oc_task.iter_samples(model=model, variant=oc_variant_str, n=n_per_task):
        res = oc_task.evaluate_sample(model=model, sample=sample, variant=oc_variant_str)
        correct += int(res.get("correct", 0))
        incorrect += int(res.get("incorrect", 0))
        invalid += int(res.get("invalid", 0))
        if len(examples) < examples_per_task:
            try:
                # Prepend SA prefix when needed for example generation
                msgs = oc_task._build_prompt_with_variant(sample.prompt, oc_variant_str)  # noqa: SLF001
            except Exception:
                msgs = sample.prompt
            ex = _generate_example(model, msgs, max_tokens=2, temperature=0.0)
            ex["classification"] = "n/a"
            examples.append({"task": "output_control", **ex})
    t1 = time.time()
    oc_total = correct + incorrect + invalid
    oc_acc = (correct / oc_total) if oc_total else 0.0
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "output_control",
            "variant": oc_variant_str,
            "n_requested": n_per_task,
            "n_processed": oc_total,
            "invalid": invalid,
            "accuracy": f"{oc_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    return examples


def run_id_leverage(
    *,
    model: str,
    csv_out: str,
    n_per_task: int,
    examples_per_task: int,
    variant: str,
    comment: str | None = None
) -> list[dict[str, Any]]:
    examples = []
    idlev_task = make_idlev_generic_task()
    id_variant_str = variant or getattr(idlev_task.default_variant, 'name', 'plain')
    provider = get_provider_for_model(model, prefer_transformerlens=False)
    t0 = time.time()
    correct = 0
    incorrect = 0
    invalid = 0
    i = 0
    for sample in idlev_task.iter_samples(model=model, variant=id_variant_str, n=n_per_task):
        messages = sample["messages"]
        req = GetTextRequest(context=None, prompt=messages, max_tokens=200, temperature=0.0)
        resp = provider.generate_text(req)
        txt = getattr(resp, "txt", None)
        identity_true = bool(sample.get("identity_true", True))
        scored = idlev_task._score_from_text(txt, sample["target_language"], identity_true)  # noqa: SLF001
        if scored.get("invalid"):
            invalid += 1
        elif scored.get("success"):
            correct += 1
        else:
            incorrect += 1
        if len(examples) < examples_per_task:
            ex = {
                "task": "id_leverage_generic",
                "prompt": _last_user_content(messages) or sample.get("request_text", ""),
                "response": txt or "",
                "classification": "n/a",
            }
            examples.append(ex)
        i += 1
        if i >= n_per_task:
            break
    t1 = time.time()
    id_total = correct + incorrect + invalid
    id_acc = (correct / id_total) if id_total else 0.0
    _write_csv_row(
        csv_out,
        {
            "model_id": model,
            "task": "id_leverage_generic",
            "variant": id_variant_str,
            "n_requested": n_per_task,
            "n_processed": id_total,
            "invalid": invalid,
            "accuracy": f"{id_acc:.6f}",
            "runtime_seconds": f"{t1 - t0:.3f}",
            "comment": comment or "",
        },
    )
    return examples


def run_benchmark(
    *,
    model: str,
    out_dir: str,
    n_per_task: int,
    examples_per_task: int,
    variant: str,
    comment: str | None = None,
) -> None:
    _ensure_dir(out_dir)
    ts = int(time.time())
    examples_out = os.path.join(out_dir, f"examples_{model}_{ts}.json")
    csv_out = os.path.join(out_dir, f"scores_{model}.csv")

    all_examples: list[dict[str, Any]] = []

    # Run each task (comment out to skip)
    # all_examples.extend(run_stages_oversight(
    #     model=model,
    #     csv_out=csv_out,
    #     n_per_task=n_per_task,
    #     examples_per_task=examples_per_task,
    #     variant=variant,
    #     comment=comment,
    # ))
    # all_examples.extend(run_self_recognition(
    #     model=model,
    #     csv_out=csv_out,
    #     n_per_task=n_per_task,
    #     examples_per_task=examples_per_task,
    #     variant=variant,
    #     comment=comment,
    # ))
    # all_examples.extend(run_output_control(
    #     model=model,
    #     csv_out=csv_out,
    #     n_per_task=n_per_task,
    #     examples_per_task=examples_per_task,
    #     variant=variant,
    #     comment=comment,
    # ))
    all_examples.extend(run_id_leverage(
        model=model,
        csv_out=csv_out,
        n_per_task=n_per_task,
        examples_per_task=examples_per_task,
        variant=variant,
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
    parser.add_argument("--variant", choices=["plain", "sp"], default="plain", help="Prompt variant: plain or sp (Situation Prompt)")
    parser.add_argument("--comment", default=None, help="Optional comment to include in CSV rows")
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(
        model=args.model,
        out_dir=args.out_dir,
        n_per_task=args.n,
        examples_per_task=args.examples,
        variant=args.variant,
        comment=args.comment,
    )
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()


