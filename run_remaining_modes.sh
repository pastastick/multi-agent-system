#!/usr/bin/env bash
# Lanjutan batch 2026-07-05_09-36-48: mode kv & kv_and_text (text sudah selesai).
# Run dir memakai stamp yang sama agar monitor tail -F tetap tersambung.
# Summary ditulis dengan >> (bukan tee) — pipe stdout bisa saja sudah mati.
set -u
source /workspace/runpod_env.sh
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_API_TOKEN
source /workspace/project/multi-agent-system/.venv/bin/activate
cd /workspace/project/multi-agent-system

STAMP=2026-07-05_09-36-48
SUMMARY=/workspace/project/multi-agent-system/backend/runs/prod_${STAMP}_summary.log

# Urutan diagnostik (permintaan user 2026-07-05): kv_and_text DULU — semua agen
# emit teks tapi handoff tetap KV, sehingga degenerasi konteks laten (seperti
# construct-degen di run kv 12:06) bisa dilokalisasi per-agen. kv terakhir.
for MODE in kv_and_text kv; do
  RUN_DIR="/workspace/project/multi-agent-system/backend/runs/prod_${MODE}_${STAMP}"
  mkdir -p "$RUN_DIR"
  echo "=== [$(date '+%F %T')] START mode=$MODE run_dir=$RUN_DIR ===" >> "$SUMMARY"
  QUANTA_RUN_DIR="$RUN_DIR" PYTHONPATH=backend python launcher.py mine \
      --direction "price-volume momentum factor" \
      --config_path "configs/experiment_${MODE}.yaml" \
      >> "$RUN_DIR/stdout.log" 2>&1
  echo "=== [$(date '+%F %T')] END   mode=$MODE rc=$? ===" >> "$SUMMARY"
done
echo "batch remaining done $(date)" >> "$SUMMARY"
