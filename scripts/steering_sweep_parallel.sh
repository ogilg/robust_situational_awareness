#!/usr/bin/env bash
set -euo pipefail

# Parallel steering sweep across 8 GPUs.
# - Distributes the Cartesian product of sweep params round-robin over GPUs 0..7
# - Mirrors the single-GPU sweep in scripts/steering_sweep.sh

MODEL="qwen-2.5-14b-instruct"
OUT_DIR="experiments/results/steering_sweep"
LOG_DIR="experiments/logs"

# Help avoid CUDA allocator fragmentation OOMs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

# Dataset/run sizes
N=100
EXAMPLES=0

# Sweep space (edit as needed)
STEERING_MODE="add"          # add | project
VECTOR_SOURCE="stages"  # stages | self_recognition | output_control | id_leverage
VARIANT="plain"               # plain | sp (dataset variant)
VECTOR_VARIANTS=("plain" "sp")
COEFFICIENTS=(0.005 0.01 0.02)
LAYERS_LIST=(10 25 40)

mkdir -p "$OUT_DIR" "$LOG_DIR"

run_job() {
  local gpu="$1"; shift
  local tag="$1"; shift
  local cmd=("$@")
  local log="${LOG_DIR}/${tag}_gpu${gpu}.log"
  echo "[sweep] gpu=${gpu} tag=${tag} -> ${cmd[*]} | log=${log}"
  (
    set -o pipefail
    # First attempt
    env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log}" 2>&1 || {
      echo "[retry] ${tag}: first attempt failed, trying once more with stricter allocator settings" | tee -a "${log}"
      # Opportunistic cache clear (mostly a no-op across processes, but harmless)
      python - <<'PY'
try:
    import torch
    torch.cuda.empty_cache()
except Exception:
    pass
PY
      # Second attempt with smaller split size
      env CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32" "${cmd[@]}" >>"${log}" 2>&1
    }
  ) &
}

# Build and launch jobs
gpu_cycle=(0 1)
gidx=0

ts=$(date +%s)
for VV in "${VECTOR_VARIANTS[@]}"; do
  for COEF in "${COEFFICIENTS[@]}"; do
    for LAYERS in "${LAYERS_LIST[@]}"; do
      gpu="${gpu_cycle[$((gidx % ${#gpu_cycle[@]}))]}"; gidx=$((gidx+1))
      tag="steer_sweep_${STEERING_MODE}_${VECTOR_SOURCE}_vv-${VV}_c-${COEF}_L-${LAYERS}_t-${ts}"

      run_job "$gpu" "$tag" \
        env SA_SPLIT=test PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" python -m experiments.benchmark_with_steering \
          --model "$MODEL" \
          --n "$N" --examples "$EXAMPLES" \
          --steering-mode "$STEERING_MODE" --vector-source "$VECTOR_SOURCE" \
          --variant "$VARIANT" --vector-variant "$VV" \
          --coefficient "$COEF" --layers "$LAYERS" \
          --tasks id_leverage \
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


