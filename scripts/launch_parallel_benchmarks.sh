#!/usr/bin/env bash
set -euo pipefail

# Hardcoded 8-GPU parallel launcher for SA benchmarks.
# - Runs benchmark_model.py (HF only), benchmark_with_vectors.py (TL vectors), and benchmark_with_steering.py (TL steering)
# - Tasks: stages, self_recognition, output_control, id_leverage; ab_baseline for vectors only
# - Variants: plain and sp
# - Model: Qwen2.5-14B-Instruct (change MODEL below if needed)

MODEL="qwen-2.5-14b-instruct"
N=200
EXAMPLES=5
LOG_DIR="experiments/logs"
STEERING_MODE="none"   # set to add or project if you want steering batch
VECTOR_SOURCE="stages" # used if STEERING_MODE != none
COEFF=1.0
HOOK_VARIANT="pre"
POSITIONS="all"
LAYERS=(5 10 15)
CLASS_NAME="correct"

mkdir -p "$LOG_DIR"

run_job() {
  local gpu="$1"; shift
  local tag="$1"; shift
  local cmd=("$@")
  local log="${LOG_DIR}/${tag}_gpu${gpu}.log"
  echo "[launch] gpu=${gpu} tag=${tag} -> ${cmd[*]} | log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log}" 2>&1 &
}

run_benchmark_model_jobs() {
  # ------- benchmark_model.py (HF provider, no TL) -------
  bm_tasks=(stages self_recognition output_control id_leverage)
  bm_variants=(plain sp)
  gpu_cycle=(0 1 2 3 4 5 6 7)
  gidx=0
  for t in "${bm_tasks[@]}"; do
    for v in "${bm_variants[@]}"; do
      gpu="${gpu_cycle[$((gidx % ${#gpu_cycle[@]}))]}"; gidx=$((gidx+1))
      tag="bm_${t}_${v}"
      run_job "$gpu" "$tag" SA_SPLIT=train python -m experiments.benchmark_model \
        --model "$MODEL" \
        --n "$N" \
        --examples "$EXAMPLES" \
        --variant "$v" \
        --comment "auto [task=${t}]"
    done
  done
}

run_benchmark_with_vectors_jobs() {
  # ------- benchmark_with_vectors.py (TransformerLens vectors) -------
  bv_tasks=(stages self_recognition output_control id_leverage)
  bv_variants=(plain sp)
  gpu_cycle=(0 1 2 3 4 5 6 7)
  gidx=0
  for t in "${bv_tasks[@]}"; do
    for v in "${bv_variants[@]}"; do
      gpu="${gpu_cycle[$((gidx % ${#gpu_cycle[@]}))]}"; gidx=$((gidx+1))
      # map task names to module choices
      case "$t" in
        self_recognition) task_arg="self_recognition_who";;
        id_leverage) task_arg="id_leverage_generic";;
        stages|output_control) task_arg="$t";;
        *) continue;;
      esac
      tag="bv_${t}_${v}"
      run_job "$gpu" "$tag" SA_SPLIT=train python -m experiments.benchmark_with_vectors \
        --model "$MODEL" \
        --n "$N" \
        --variant "$v" \
        --tasks "$task_arg" \
        --comment "auto"
    done
  done
}

run_benchmark_with_steering_jobs() {
  # ------- benchmark_with_steering.py (TransformerLens steering) -------
  bs_tasks=(stages self_recognition output_control id_leverage)
  bs_variants=(plain sp)
  gpu_cycle=(0 1 2 3 4 5 6 7)
  gidx=0
  for t in "${bs_tasks[@]}"; do
    for v in "${bs_variants[@]}"; do
      gpu="${gpu_cycle[$((gidx % ${#gpu_cycle[@]}))]}"; gidx=$((gidx+1))
      case "$t" in
        self_recognition) task_arg="self_recognition";;
        id_leverage) task_arg="id_leverage";;
        stages|output_control) task_arg="$t";;
        *) continue;;
      esac
      tag="bs_${t}_${v}_${STEERING_MODE}"
      if [[ "$STEERING_MODE" == "none" ]]; then
        run_job "$gpu" "$tag" SA_SPLIT=test python -m experiments.benchmark_with_steering \
          --model "$MODEL" \
          --n "$N" \
          --examples "$EXAMPLES" \
          --tasks "$task_arg" \
          --variant "$v" \
          --steering-mode none \
          --comment "auto"
      else
        # Build layers args
        layer_args=()
        for L in "${LAYERS[@]}"; do layer_args+=("$L"); done
        run_job "$gpu" "$tag" SA_SPLIT=test python -m experiments.benchmark_with_steering \
          --model "$MODEL" \
          --n "$N" \
          --examples "$EXAMPLES" \
          --tasks "$task_arg" \
          --variant "$v" \
          --steering-mode "$STEERING_MODE" \
          --vector-source "$VECTOR_SOURCE" \
          --coefficient "$COEFF" \
          --hook-variant "$HOOK_VARIANT" \
          --positions "$POSITIONS" \
          --layers ${layer_args[*]} \
          --class-name "$CLASS_NAME" \
          --comment "auto"
      fi
    done
  done
}

# Kick off all job groups
run_benchmark_model_jobs
run_benchmark_with_vectors_jobs
run_benchmark_with_steering_jobs

echo "Launched all jobs. Tailing latest logs (Ctrl-C to stop)..."
sleep 2
ls -1t "$LOG_DIR" | head -n 5 | while read -r f; do echo "==> $LOG_DIR/$f"; tail -n 5 "$LOG_DIR/$f" || true; done

wait
echo "All jobs completed. Logs: $LOG_DIR"


