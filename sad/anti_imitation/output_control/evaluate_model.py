#!/usr/bin/env python3
"""
CLI to evaluate a single model (base or LoRA fine-tuned) using the output-control experiment.

- Pass a base model ID (e.g., "llama-3.1-8b")
- Optionally pass a LoRA adapter path or Hub repo (e.g., "username/model-lora")
"""

import argparse
import json
import os
import sys

# Ensure we can import sibling module when running from repo root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from run_eval import run_experiment_for_model, update_results_csv, update_metrics
from config import OutputControlExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single model with the output-control experiment")
    parser.add_argument("model", help="Model ID to evaluate (e.g., llama-3.1-8b)")
    parser.add_argument("--num_examples", type=int, default=10, help="Examples per experiment type (default: 10)")
    parser.add_argument(
        "--lora_adapter",
        type=str,
        default=None,
        help="Path or HF repo ID of a LoRA adapter to load during evaluation",
    )
    parser.add_argument(
        "--save_individual",
        dest="save_individual",
        action="store_true",
        default=True,
        help="Save individual model results to JSON (in addition to CSV updates). Enabled by default.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort to configure for GPT-OSS models",
    )
    parser.add_argument(
        "--feed_empty_analysis",
        action="store_true",
        default=True,
        help="Prime an empty analysis channel (GPT-OSS) to discourage thinking",
    )


    args = parser.parse_args()

    # Build config and run
    cfg = OutputControlExperimentConfig(
        model=args.model,
        num_examples=args.num_examples,
        lora_adapter=args.lora_adapter,
        reasoning_effort=args.reasoning_effort,
        feed_empty_analysis=args.feed_empty_analysis,
        save_individual=args.save_individual,
    )

    results = run_experiment_for_model(cfg)

    # Update CSVs (same behavior as the multi-model script)
    update_results_csv(args.model, results)
    update_metrics(cfg, results)

    # Optionally save individual JSON results
    if cfg.save_individual:
        output_file = f"results/output_control_results_{cfg.model.replace('/', '_')}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Individual model results saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


