#!/usr/bin/env bash
# Skor ulang korpus front-end SATU TAG PER PROSES.
#
# Kenapa bukan satu proses untuk 31 tag: proses yang panjang tumbuh terus
# (pandas menyisakan fragmentasi dari ekspresi bersarang atas ±1 jt baris) dan
# di mesin 8 GB ia dibunuh OOM di tengah korpus — dua kali, terukur. Satu
# proses per tag mengembalikan seluruh memori tiap tag selesai; cache di disk
# (`lab/out/.rescore_cache.json` + `.rescore_series.parquet`) yang membuat
# ekspresi yang sudah dinilai tidak dihitung ulang antar-proses.
#
# Aman diulang: jalankan lagi dan ia melanjutkan dari cache.
#
#   bash lab/rescore_all.sh            # semua tag, urutan prioritas
#   LAB_MAX_WORKERS=8 bash lab/rescore_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=backend
export LAB_MAX_WORKERS="${LAB_MAX_WORKERS:-3}"
PY=.venv/bin/python
LOG=lab/out_rescore.log

# Urutan: tag yang menopang angka headline lebih dulu, supaya berhenti di
# tengah tetap memulihkan sumbu yang paling banyak dikutip dokumen.
TAGS=(
  b14_summary b14_text
  a8_kv_innovate_guided a8_kv_full a8_kv_innovate a8_kv_nodesign
  a8_kv_direct a8_kv_innovate_fid a8_kv_full_guided
  g4_kv_ls10 g4_text g4_kv_and_text_ls10
  g2_kv_ls5 g2_kv_ls10 g3_kv_ls10_gumbelT0.7 g3_kv_ls10_raw
  g3_kv_ls10_sampleT1.0 g6_kv_ls10_realignON g6_kv_ls10_realignOFF
  b5_pre b5_post a10 tahap4_sanity
  px_8B_v0 px_8B_v1 px_4B_v0 px_4B_v1
  g2_kv_ls20 g2_kv_ls40 g2_kv_ls60 g4_kv_and_text_ls60
)

echo "=== rescore per-tag dimulai $(date '+%F %T') · worker=$LAB_MAX_WORKERS ===" | tee -a "$LOG"
for t in "${TAGS[@]}"; do
    [ -f "lab/out/frontend_${t}.json" ] || { echo "lewat (tak ada): $t"; continue; }
    if [ -f "lab/out/icseries_${t}.parquet" ]; then
        echo "lewat (sudah ada): $t" | tee -a "$LOG"
        continue
    fi
    $PY lab/rescore_all.py --tags "$t" 2>&1 | grep -Ev "^\s*$|resource_tracker|warnings.warn" | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && echo "!! $t gagal (rc=$rc) — dilanjut" | tee -a "$LOG"
done
echo "=== selesai $(date '+%F %T') ===" | tee -a "$LOG"
