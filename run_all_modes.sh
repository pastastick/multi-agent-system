#!/usr/bin/env bash
# Full produksi+backtest 3 comm_mode sekuensial: text -> kv -> kv_and_text.
# Setiap mode dapat run-dir sendiri (QUANTA_RUN_DIR) + stdout log terpisah.
set -u
source /workspace/runpod_env.sh
# HF_TOKEN di runpod_env.sh kedaluwarsa -> 401 bahkan utk model publik (Qwen3-4B).
# Tanpa token, download publik jalan normal. Perbarui token di runpod_env.sh bila
# butuh repo privat lagi.
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_API_TOKEN
source /workspace/project/multi-agent-system/.venv/bin/activate
cd /workspace/project/multi-agent-system

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
SUMMARY=/workspace/project/multi-agent-system/backend/runs/prod_${STAMP}_summary.log
echo "batch start $(date)" | tee "$SUMMARY"

for MODE in text kv kv_and_text; do
  RUN_DIR="/workspace/project/multi-agent-system/backend/runs/prod_${MODE}_${STAMP}"
  mkdir -p "$RUN_DIR"
  echo "=== [$(date '+%F %T')] START mode=$MODE run_dir=$RUN_DIR ===" | tee -a "$SUMMARY"
  # --direction WAJIB: tanpa ini planning menghasilkan [None]*n (silent skip,
  # factor_mining.py "elif planning_enabled") dan proposal berjalan tanpa arah.
  # Seed = contoh kanonik README, konsisten dengan run referensi 23 Jun.
  QUANTA_RUN_DIR="$RUN_DIR" PYTHONPATH=backend python launcher.py mine \
      --direction "price-volume momentum factor" \
      --config_path "configs/experiment_${MODE}.yaml" \
      > "$RUN_DIR/stdout.log" 2>&1
  RC=$?
  echo "=== [$(date '+%F %T')] END   mode=$MODE rc=$RC ===" | tee -a "$SUMMARY"
done
echo "batch done $(date)" | tee -a "$SUMMARY"
