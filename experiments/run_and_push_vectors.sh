#!/usr/bin/env bash
set -euo pipefail

# Usage: ./experiments/run_and_push_vectors.sh [MODEL] [N] [OUT_DIR] [COMMENT]
# Defaults:
#   MODEL   = llama-3.1-8b-instruct
#   N       = 10
#   OUT_DIR = experiments/results
#   COMMENT = (empty)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${1:-llama-3.1-8b-instruct}"
N="${2:-10}"
OUT_DIR_REL="${3:-experiments/results}"
COMMENT="${4:-}"

OUT_DIR_ABS="$PROJECT_ROOT/$OUT_DIR_REL"
mkdir -p "$OUT_DIR_ABS"

echo "Running vector benchmark: model=$MODEL n=$N out_dir=$OUT_DIR_REL"
python3 "$PROJECT_ROOT/experiments/benchmark_with_vectors.py" \
  --model "$MODEL" \
  --n "$N" \
  --out-dir "$OUT_DIR_ABS" \
  ${COMMENT:+--comment "$COMMENT"}

# Stage results (aggregated vectors and scores CSV)
echo "Staging artifacts..."
git add "$OUT_DIR_REL"/*.npz || true
git add "$OUT_DIR_REL"/*.json || true
git add "$OUT_DIR_REL"/*.csv || true

# Commit if there are staged changes
if ! git diff --cached --quiet; then
  TS="$(date -Iseconds)"
  git commit -m "Add aggregated vectors for $MODEL (n=$N) at $TS"${COMMENT:+ -m "$COMMENT"}
  echo "Pushing to origin..."
  git push
else
  echo "No changes to commit."
fi

echo "Done."


