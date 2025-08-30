#!/usr/bin/env python3
import argparse
import os

from .task_stages import make_task


def main():
    parser = argparse.ArgumentParser(description="Run provider-wrapper based Stages Oversight evaluation")
    parser.add_argument("--model", required=True, help="Model id (as expected by provider_wrapper)")
    parser.add_argument("--data", default=None, help="Path to a YAML/JSONL file or a directory containing batch YAMLs (optional)")
    parser.add_argument("--variant", default="plain", choices=["plain", "sp"], help="Variant")
    parser.add_argument("--n", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    task = make_task(data_path=args.data)
    # Map string to variant object via task.variants
    variant = next(v for v in task.variants if v.name == args.variant)
    task.run(model=args.model, variant=variant, n=args.n, save=True)
    print("Done.")


if __name__ == "__main__":
    main()


