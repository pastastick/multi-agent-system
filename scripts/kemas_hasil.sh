#!/usr/bin/env bash
# Kemas seluruh hasil eksperimen jadi satu arsip untuk diunduh sebelum pod
# RunPod dihapus.
#
# Kenapa ini perlu padahal sudah di-push ke GitHub: `results/**/llm_outputs/`
# SENGAJA tidak di-track git (puluhan MB dan terus tumbuh), padahal isinya
# transkrip agen dari run yang sudah lewat — satu-satunya salinan. Begitu pod
# dihapus, ia hilang permanen. Arsip ini yang menyelamatkannya.
#
#     bash scripts/kemas_hasil.sh            # semua, termasuk llm_outputs
#     bash scripts/kemas_hasil.sh --ringkas  # tanpa llm_outputs (~2 MB)
set -u

cd /workspace/project/multi-agent-system || exit 1
STAMP=$(date +%Y%m%d_%H%M)
RINGKAS=0
[ "${1:-}" = "--ringkas" ] && RINGKAS=1

# `.cache/` selalu dikecualikan: itu cache data pasar (~60 MB) yang dibangkitkan
# ulang otomatis dari `backend/hf_data/daily_pv.h5` saat `eval/ic.py` dipakai.
# Membawanya hanya memperbesar unduhan tanpa menyelamatkan apa pun.
EXCL="--exclude=.cache"
if [ $RINGKAS -eq 1 ]; then
    OUT="/workspace/hasil_ringkas_${STAMP}.tar.gz"
    EXCL="$EXCL --exclude=llm_outputs"
else
    OUT="/workspace/hasil_lengkap_${STAMP}.tar.gz"
fi

echo "Mengemas → $OUT"
tar czf "$OUT" $EXCL \
    results/ \
    configs/matriks.yaml \
    docs/ \
    2>/dev/null

echo
echo "=== ISI ARSIP ==="
tar tzf "$OUT" | awk -F/ '{print $1"/"$2}' | sort | uniq -c | sort -rn | head -12
echo
echo "=== UKURAN ==="
du -h "$OUT"
echo
echo "Unduh dari mesin lokalmu dengan salah satu cara:"
echo "  scp -P <PORT> root@<HOST>:$OUT ."
echo "  atau lewat file browser RunPod di /workspace/"
