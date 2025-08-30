#!/usr/bin/env bash
set -euo pipefail

HF_HOME_DEFAULT="/workspace/.hf_home"
HF_HUB_DIR="${1:-${HF_HOME_DEFAULT}/hub}"

echo "Clearing Hugging Face cache under: ${HF_HUB_DIR}"
echo "This removes cached models and will re-download on next use."

# Option 1: fast nuke of hub contents
if [ -d "${HF_HUB_DIR}" ]; then
  sudo rm -rf "${HF_HUB_DIR}/models--"* || true
  sudo rm -rf "${HF_HUB_DIR}/datasets--"* || true
  sudo rm -rf "${HF_HUB_DIR}/snapshots" || true
  sudo rm -rf "${HF_HUB_DIR}/blobs" || true
fi

# Also clear xet cache if using it
if [ -d "${HF_HOME_DEFAULT}/xet" ]; then
  echo "Clearing xet cache: ${HF_HOME_DEFAULT}/xet"
  sudo rm -rf "${HF_HOME_DEFAULT}/xet" || true
fi

# Optional: pip and torch caches
pip cache purge || true
rm -rf /root/.cache/torch || true

echo "Done. Current disk usage:"
df -hT