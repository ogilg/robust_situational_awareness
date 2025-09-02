#!/usr/bin/env bash
set -euo pipefail

# Parallel steering sweep across 8 GPUs.
# - Distributes the Cartesian product of sweep params round-robin over GPUs 0..7
# - Mirrors the single-GPU sweep in scripts/steering_sweep.sh

MODEL="qwen-2.5-14b-instruct"
OUT_DIR="experiments/results/steering_sweep"
LOG_DIR="experiments/logs"

# Dataset/run sizes
N=100
EXAMPLES=0

# Sweep space (edit as needed)
STEERING_MODE="add"          # add | project
VECTOR_SOURCE="id_leverage"  # stages | self_recognition | output_control | id_leverage
VARIANT="plain"               # plain | sp (dataset variant)
VECTOR_VARIANTS=("sp" "plain")
COEFFICIENTS=(0.01)
LAYERS_LIST=(10 25 40)

mkdir -p "$OUT_DIR" "$LOG_DIR"

run_job() {
  local gpu="$1"; shift
  local tag="$1"; shift
  local cmd=("$@")
  local log="${LOG_DIR}/${tag}_gpu${gpu}.log"
  echo "[sweep] gpu=${gpu} tag=${tag} -> ${cmd[*]} | log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log}" 2>&1 &
}

# Build and launch jobs
gpu_cycle=(0 1 2 3 4 5 6 7)
gidx=0

ts=$(date +%s)
for VV in "${VECTOR_VARIANTS[@]}"; do
  for COEF in "${COEFFICIENTS[@]}"; do
    for LAYERS in "${LAYERS_LIST[@]}"; do
      gpu="${gpu_cycle[$((gidx % ${#gpu_cycle[@]}))]}"; gidx=$((gidx+1))
      tag="steer_sweep_${STEERING_MODE}_${VECTOR_SOURCE}_vv-${VV}_c-${COEF}_L-${LAYERS}_t-${ts}"
      run_job "$gpu" "$tag" \
        SA_SPLIT=test python -m experiments.benchmark_with_steering \
          --model "$MODEL" \
          --n "$N" --examples "$EXAMPLES" \
          --steering-mode "$STEERING_MODE" --vector-source "$VECTOR_SOURCE" \
          --variant "$VARIANT" --vector-variant "$VV" \
          --coefficient "$COEF" --layers "$LAYERS" \
          --out-dir "$OUT_DIR" \
          --comment "vv=$VV coef=$COEF layers=$LAYERS"
    done
  done
done

echo "Launched all steering sweep jobs. Tailing latest logs (Ctrl-C to stop)..."
sleep 2
ls -1t "$LOG_DIR" | head -n 5 | while read -r f; do echo "==> $LOG_DIR/$f"; tail -n 5 "$LOG_DIR/$f" || true; done

wait
echo "All sweep jobs completed. Logs: $LOG_DIR | Results: $OUT_DIR"


