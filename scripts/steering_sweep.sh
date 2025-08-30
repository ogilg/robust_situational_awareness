#!/usr/bin/env bash
# Simple steering sweep runner. Edit as you like.
# Assumes you're in repo root and your venv is active.

set -euo pipefail

MODEL="qwen-2.5-14b-instruct"
OUT_DIR="experiments/results/steering_sweep"
VECTORS_DIR="experiments/vectors"
N=100
EXAMPLES=0

mkdir -p "$OUT_DIR"

# ---------------- Add-mode sweep ----------------
# Parameters to sweep
VECTOR_VARIANTS=("sp" "plain")
COEFFICIENTS=(0.01)
LAYERS_LIST=(10 25 40)

for VV in "${VECTOR_VARIANTS[@]}"; do
  for COEF in "${COEFFICIENTS[@]}"; do
    for LAYERS in "${LAYERS_LIST[@]}"; do
      python -m experiments.benchmark_with_steering \
        --model "$MODEL" \
        --n $N --examples $EXAMPLES \
        --steering-mode add --vector-source id_leverage \
        --variant plain --vector-variant "$VV" \
        --coefficient $COEF --layers $LAYERS \
        --out-dir "$OUT_DIR" --comment "vv=$VV coef=$COEF layers=$LAYERS"
    done
  done
done

# # ---------------- Project-mode (SP vectors) ----------------
# python -m experiments.benchmark_with_steering \
#   --model "$MODEL" \
#   --n $N --examples $EXAMPLES \
#   --steering-mode project --vector-source stages \
#   --variant sp --vector-variant sp \
#   --layers 20 30 40 \
#   --out-dir "$OUT_DIR"

# python -m experiments.benchmark_with_steering \
#   --model "$MODEL" \
#   --n $N --examples $EXAMPLES \
#   --steering-mode project --vector-source id_leverage \
#   --variant sp --vector-variant sp \
#   --layers 25 35 45 \
#   --out-dir "$OUT_DIR"

# # ---------------- Variations: positions/hook ----------------
# python -m experiments.benchmark_with_steering \
#   --model "$MODEL" \
#   --n $N --examples $EXAMPLES \
#   --steering-mode add --vector-source stages \
#   --variant sp --vector-variant sp \
#   --coefficient 0.1 --layers 30 \
#   --positions last --hook-variant pre \
#   --out-dir "$OUT_DIR"

# python -m experiments.benchmark_with_steering \
#   --model "$MODEL" \
#   --n $N --examples $EXAMPLES \
#   --steering-mode add --vector-source stages \
#   --variant sp --vector-variant sp \
#   --coefficient 0.1 --layers 30 \
#   --positions all --hook-variant post \
#   --out-dir "$OUT_DIR"

# echo "Done. Results under $OUT_DIR"


