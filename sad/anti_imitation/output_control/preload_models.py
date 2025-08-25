#!/usr/bin/env python3
"""
Script to preload a single model into HuggingFace cache with retry logic.
This separates model downloading from evaluation to handle download failures gracefully.
"""

import argparse
import os
import sys
import time

from provider_wrapper import (
    preload_model,
    clear_model_cache,
    clear_huggingface_disk_cache,
)


def _set_env_defaults():
    """Set environment defaults for more stable downloads."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_DOWNLOAD_MAX_WORKERS", "4")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main():
    parser = argparse.ArgumentParser(description="Preload a single model into HuggingFace cache (with retry)")
    parser.add_argument("model", help="Model ID to preload (e.g., llama-3.1-8b)")
    parser.add_argument(
        "--quantization",
        choices=["bnb", "mxfp4"],
        default="bnb",
        help="Quantization mode: mxfp4 (default, model-native), or bnb (bitsandbytes 4-bit)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during preloading")
    parser.add_argument("--max_attempts", type=int, default=4, help="Maximum retry attempts")
    parser.add_argument("--sleep_between_attempts", type=int, default=10, help="Seconds to sleep between attempts")

    args = parser.parse_args()

    _set_env_defaults()

    # Preload the single model with retries
    print("\n" + "=" * 60)
    print("PRELOADING MODEL")
    print("=" * 60)

    for attempt in range(1, args.max_attempts + 1):
        print(f"Attempt {attempt}/{args.max_attempts}: Preloading {args.model} (quantization={args.quantization})...")
        success = preload_model(args.model, verbose=args.verbose, quantization=args.quantization)

        if success:
            print("\n" + "=" * 60)
            print("PRELOAD SUMMARY")
            print("=" * 60)
            print(f"✓ Successfully preloaded: {args.model}")
            print("\nModel is ready! You can now run evaluations or fine-tuning.")
            return 0

        print(f"Preload failed on attempt {attempt}. Clearing caches and retrying...")
        clear_model_cache(args.model)
        clear_huggingface_disk_cache()
        time.sleep(args.sleep_between_attempts)

    print("\n" + "=" * 60)
    print("PRELOAD SUMMARY")
    print("=" * 60)
    print(f"✗ Failed to preload: {args.model} after {args.max_attempts} attempts")
    print("\nYou may want to retry or check your internet connection.")
    return 1


if __name__ == "__main__":
    sys.exit(main())