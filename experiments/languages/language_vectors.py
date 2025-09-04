#!/usr/bin/env python3
"""
Script to benchmark the id_leverage task and aggregate activations by detected output language.

Heavily inspired by benchmark_with_vectors.py but simplified to focus only on id_leverage
and group vectors by detected language instead of correct/incorrect classification.
"""

import os
import sys
import time
from typing import Any
import numpy as _np
import json as _json
from tqdm import tqdm
try:
    import torch as _torch  # for safe dtype casting on TL tensors
except Exception:
    _torch = None

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Tasks
from sad.id_leverage.entity_id.task_generic import make_task as make_idlev_generic_task  # noqa: E402

from provider_wrapper import clear_gpu_memory


def _to_numpy_cpu(x):
    try:
        # Torch tensor path
        if _torch is not None and isinstance(x, _torch.Tensor):
            t = x.detach().to("cpu")
            # numpy does not support bfloat16; cast low-precision to float32 before numpy()
            if t.dtype in (getattr(_torch, "bfloat16", None), getattr(_torch, "float16", None)):
                t = t.to(_torch.float32)
            return t.numpy()
        # Fallbacks
        return _np.asarray(x)
    except Exception:
        try:
            # Last resort cast via list
            return _np.asarray(x.tolist())
        except Exception:
            return _np.asarray(x)


# Import language detection from existing task
from sad.id_leverage.entity_id.task_generic import _detect_language, LANGUAGE_STOPWORDS  # noqa: E402


def _detect_output_language(text: str, target_language: str) -> str:
    """
    Use existing language detection logic from id_leverage task.
    Only checks between the target language and English.
    Returns detected language or 'english' as fallback.
    """
    if not text or not isinstance(text, str):
        return "english"
    
    # Only check target language vs English 
    candidate_languages = [target_language.lower(), "english"]
    detected = _detect_language(text, candidate_languages)
    
    return detected or "english"


def _aggregate_vectors_by_language(
    task,
    *,
    model: str,
    variant: str,
    n: int,
    task_name: str,
) -> tuple[dict, dict, dict]:
    """
    Iterate samples via task.iter_samples, detect output language,
    and maintain running sums per detected language and per layer.

    Returns (language_counts, sum_by_language, count_by_language)
    - language_counts: {"english": int, "spanish": int, "french": int, "german": int, "italian": int}
    - sum_by_language: {language: {layer_idx: np.array(d_model,)}}
    - count_by_language: {language: int}
    """
    # Track english + all possible target languages
    all_languages = {"english", "spanish", "french", "german", "italian"}
    language_counts = {lang: 0 for lang in all_languages}
    sum_by_language: dict[str, dict[int, _np.ndarray]] = {lang: {} for lang in all_languages}
    count_by_language: dict[str, int] = {lang: 0 for lang in all_languages}

    # Always work with simple string variants (e.g., "plain" or "sp")
    variant_str = str(variant)

    i = 0
    for sample in tqdm(task.iter_samples(model=model, variant=variant_str, n=n)):
        result, residuals, aux_meta = task.evaluate_and_capture_sample(model=model, sample=sample, variant=variant_str)
        
        # Get response text and target language for detection
        response_text = ""
        target_language = sample.get("target_language", "English").lower()
        if isinstance(aux_meta, dict):
            response_text = str(aux_meta.get("txt", ""))
        
        # Detect language using existing function with only target + english
        detected_language = _detect_output_language(response_text, target_language)
        language_counts[detected_language] += 1
        
        # Only aggregate if residuals have at least one captured layer and sample is not invalid
        if residuals is not None and int(result.get("invalid", 0)) == 0:
            aggregated_any = False
            # Accumulate per layer
            for layer_idx, vec in (residuals.items() if hasattr(residuals, "items") else []):
                arr = _to_numpy_cpu(vec).reshape(-1)
                if layer_idx not in sum_by_language[detected_language]:
                    sum_by_language[detected_language][layer_idx] = arr.copy()
                else:
                    sum_by_language[detected_language][layer_idx] += arr
                aggregated_any = True
            if aggregated_any:
                count_by_language[detected_language] += 1
        
        i += 1
        if n is not None and i >= n:
            break  

    return (language_counts, sum_by_language, count_by_language)


def _save_language_means(*, vectors_dir: str, model: str, variant_str: str, sums: dict, counts: dict) -> None:
    """
    Save mean vectors per language to experiments/vectors/language_vectors_{model}__{variant}.npz
    """
    os.makedirs(vectors_dir, exist_ok=True)
    
    # Build mean vectors
    mean_payload: dict[str, _np.ndarray] = {}
    
    for language, layer_dict in sums.items():
        count = counts.get(language, 0)
        if count == 0:
            continue
            
        for layer_idx, sum_vec in layer_dict.items():
            mean_vec = sum_vec / float(count)
            key = f"{language}__layer_{layer_idx}"
            mean_payload[key] = mean_vec
    
    # Save means
    if mean_payload:
        out_path = os.path.join(vectors_dir, f"language_vectors_{model}__{variant_str}.npz")
        _np.savez(out_path, **mean_payload)
        print(f"Wrote language vectors: {out_path} ({len(mean_payload)} arrays)")
        
        # Save counts summary
        counts_path = os.path.join(vectors_dir, f"language_vectors_{model}__{variant_str}.counts.json")
        with open(counts_path, "w") as f:
            _json.dump(counts, f, indent=2)
    else:
        print("No vectors found to save.")


def run_id_leverage_language_analysis(*, model: str, variant: str, n: int, vectors_dir: str) -> None:
    """Run id_leverage task and analyze by detected language"""
    idlev_task = make_idlev_generic_task()
    
    print(f"Running id_leverage analysis for {n} samples...")
    t0 = time.time()
    
    language_counts, language_sums, vector_counts = _aggregate_vectors_by_language(
        idlev_task, 
        model=model, 
        variant=variant, 
        n=n, 
        task_name="id_leverage_generic"
    )
    
    t1 = time.time()
    
    # Print summary
    total_samples = sum(language_counts.values())
    total_vectors = sum(vector_counts.values())
    
    print(f"Completed in {t1-t0:.2f} seconds")
    print(f"Total samples: {total_samples}")
    print(f"Samples with vectors: {total_vectors}")
    print("Language distribution:")
    for lang, count in language_counts.items():
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        vec_count = vector_counts[lang]
        print(f"  {lang}: {count} samples ({pct:.1f}%), {vec_count} vectors")
    
    # Save mean vectors
    _save_language_means(
        vectors_dir=vectors_dir,
        model=model,
        variant_str=variant,
        sums=language_sums,
        counts=vector_counts
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze id_leverage task by detected output language.")
    parser.add_argument("--model", required=True, help="Model id (as expected by provider_wrapper)")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--variant", choices=["plain", "sp"], default="plain", help="Prompt variant")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Use same vectors directory structure as benchmark_with_vectors.py
    vectors_dir = os.path.join(ROOT, "experiments", "vectors")
    os.makedirs(vectors_dir, exist_ok=True)
    
    run_id_leverage_language_analysis(
        model=args.model,
        variant=args.variant,
        n=args.n,
        vectors_dir=vectors_dir
    )
    
    clear_gpu_memory()


if __name__ == "__main__":
    main()