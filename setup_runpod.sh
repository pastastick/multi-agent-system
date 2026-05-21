#!/usr/bin/env bash
# setup_runpod.sh — Full setup script untuk RunPod
# Jalankan sekali setelah project di-clone/rsync ke /workspace
#
# Usage:
#   bash /workspace/project/multi-agent-system/setup_runpod.sh
#
# Setelah selesai, aktifkan env dengan:
#   source /workspace/runpod_env.sh
#   source /workspace/project/multi-agent-system/.venv/bin/activate

set -euo pipefail

PROJECT_ROOT="/workspace/project/multi-agent-system"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "============================================================"
echo " QuantaLatent — RunPod Setup Script"
echo "============================================================"

# ── 0. Source env vars ──────────────────────────────────────────
echo ""
echo "[0/7] Loading runpod env vars..."
source /workspace/runpod_env.sh

# ── 1. Install uv ke /workspace/.local/bin ──────────────────────
echo ""
echo "[1/7] Installing uv to /workspace/.local/bin ..."
mkdir -p /workspace/.local/bin

if [ ! -f /workspace/.local/bin/uv ]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/.local/bin sh
else
    echo "  uv already installed: $(uv --version)"
fi

which uv || { echo "ERROR: uv tidak ditemukan di PATH"; exit 1; }
echo "  uv: $(uv --version) at $(which uv)"

# ── 2. uv sync (install semua deps + pyqlib + pip) ──────────────
echo ""
echo "[2/7] Running uv sync (this may take 5-15 min on first run)..."
cd "$PROJECT_ROOT"

# Force python 3.10 jika tersedia, fallback ke system python
UV_PYTHON_FLAG=""
if uv python list 2>/dev/null | grep -q "3.10"; then
    UV_PYTHON_FLAG="--python 3.10"
fi

uv sync $UV_PYTHON_FLAG

echo "  venv: $(ls $PROJECT_ROOT/.venv/bin/python)"

# ── 3. Aktivasi venv ─────────────────────────────────────────────
echo ""
echo "[3/7] Activating venv..."
source "$PROJECT_ROOT/.venv/bin/activate"
echo "  python: $(which python) — $(python --version)"

# ── 4. Reinstall torch yang kompatibel dengan driver CUDA ────────
echo ""
echo "[4/7] Checking torch / CUDA compatibility..."

CUDA_DRIVER_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' || echo "unknown")
echo "  NVIDIA Driver CUDA Version: $CUDA_DRIVER_VER"

CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
echo "  torch: $TORCH_VER | cuda available: $CUDA_OK"

if [ "$CUDA_OK" != "True" ]; then
    echo "  WARNING: CUDA tidak tersedia! Reinstalling torch dengan cu128..."
    # Driver 580 supports CUDA 13.0 — cu128 wheels bekerja dengan driver >=525
    uv pip install --reinstall \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

    CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
    echo "  [after reinstall] torch: $TORCH_VER | cuda available: $CUDA_OK"

    if [ "$CUDA_OK" != "True" ]; then
        echo "  ERROR: CUDA masih tidak tersedia setelah reinstall."
        echo "  Coba manual: uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
    fi
else
    echo "  CUDA OK! torch $TORCH_VER kompatibel."
fi

# ── 5. Buat folder output & data ─────────────────────────────────
echo ""
echo "[5/7] Creating required directories..."

mkdir -p "$BACKEND_DIR/data/qlib"
mkdir -p "$BACKEND_DIR/data/results/workspace"
mkdir -p "$BACKEND_DIR/data/results/pickle_cache"
mkdir -p "$BACKEND_DIR/hf_data"
mkdir -p "$BACKEND_DIR/log"
mkdir -p "$BACKEND_DIR/debug/llm_outputs"
mkdir -p "$BACKEND_DIR/git_ignore_folder/factor_implementation_source_data"
mkdir -p "$BACKEND_DIR/git_ignore_folder/factor_implementation_source_data_debug"
mkdir -p /workspace/.cache/huggingface
mkdir -p /workspace/.cache/torch
mkdir -p /workspace/.cache/pip
mkdir -p /workspace/.cache/uv

echo "  Directories created."

# ── 6. Download HuggingFace dataset (cn_data + HDF5) ────────────
echo ""
echo "[6/7] Downloading dataset from HuggingFace (QuantaAlpha/qlib_csi300)..."
echo "  This may take 5-20 min depending on connection speed..."

cd "$BACKEND_DIR"

# Check HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    echo "  WARNING: HF_TOKEN tidak di-set, download mungkin gagal jika dataset private."
fi

# Download hanya jika file belum ada
NEED_DOWNLOAD=0
[ ! -f "hf_data/cn_data.zip" ] && NEED_DOWNLOAD=1
[ ! -f "hf_data/daily_pv.h5" ] && NEED_DOWNLOAD=1
[ ! -f "hf_data/daily_pv_debug.h5" ] && NEED_DOWNLOAD=1

if [ $NEED_DOWNLOAD -eq 1 ]; then
    # Load HF_TOKEN dari .env jika belum di-set di env
    if [ -z "${HF_TOKEN:-}" ] && [ -f "$PROJECT_ROOT/.env" ]; then
        HF_TOKEN=$(grep "^HF_TOKEN=" "$PROJECT_ROOT/.env" | cut -d= -f2- | tr -d '"')
        export HF_TOKEN
    fi

    hf download QuantaAlpha/qlib_csi300 \
        --repo-type dataset \
        --local-dir ./hf_data \
        --token "$HF_TOKEN" 2>/dev/null || \
    python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='QuantaAlpha/qlib_csi300',
    repo_type='dataset',
    local_dir='hf_data',
    token=os.environ.get('HF_TOKEN'),
)
print('Download selesai via snapshot_download.')
"
    echo "  Download selesai."
else
    echo "  File sudah ada, skip download."
fi

# ── 6b. Extract cn_data.zip ──────────────────────────────────────
if [ ! -d "$BACKEND_DIR/data/qlib/cn_data/features" ]; then
    echo ""
    echo "  Extracting cn_data.zip..."
    cd "$BACKEND_DIR"
    python -c "
import zipfile, os
src = 'hf_data/cn_data.zip'
dst = 'data/qlib/'
if os.path.exists(src):
    print(f'  Extracting {src} -> {dst}')
    zipfile.ZipFile(src).extractall(dst)
    print('  Done.')
else:
    print(f'  ERROR: {src} tidak ditemukan!')
"
else
    echo "  cn_data sudah ter-extract, skip."
fi

# ── 6c. Tempatkan HDF5 file ──────────────────────────────────────
cd "$BACKEND_DIR"
DEST_MAIN="git_ignore_folder/factor_implementation_source_data/daily_pv.h5"
DEST_DEBUG="git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5"

if [ ! -f "$DEST_MAIN" ] && [ -f "hf_data/daily_pv.h5" ]; then
    echo "  Copying daily_pv.h5 -> $DEST_MAIN"
    cp hf_data/daily_pv.h5 "$DEST_MAIN"
fi

if [ ! -f "$DEST_DEBUG" ] && [ -f "hf_data/daily_pv_debug.h5" ]; then
    echo "  Copying daily_pv_debug.h5 -> $DEST_DEBUG (as daily_pv.h5)"
    cp hf_data/daily_pv_debug.h5 "$DEST_DEBUG"
fi

# ── 7. Verifikasi final ──────────────────────────────────────────
echo ""
echo "[7/7] Final verification..."

cd "$PROJECT_ROOT"

echo ""
echo "--- Binaries ---"
echo "  uv:     $(which uv)"
echo "  python: $(which python)"

echo ""
echo "--- Cache dirs (should be under /workspace) ---"
echo "  HF_HOME:       $HF_HOME"
echo "  UV_CACHE_DIR:  $UV_CACHE_DIR"

echo ""
echo "--- Data files ---"
[ -d "$BACKEND_DIR/data/qlib/cn_data/features" ] && \
    echo "  cn_data: OK ($(ls $BACKEND_DIR/data/qlib/cn_data/features/ | wc -l) instruments)" || \
    echo "  cn_data: MISSING"

[ -f "$BACKEND_DIR/git_ignore_folder/factor_implementation_source_data/daily_pv.h5" ] && \
    echo "  daily_pv.h5 (main):  OK ($(du -sh $BACKEND_DIR/git_ignore_folder/factor_implementation_source_data/daily_pv.h5 | cut -f1))" || \
    echo "  daily_pv.h5 (main):  MISSING"

[ -f "$BACKEND_DIR/git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5" ] && \
    echo "  daily_pv.h5 (debug): OK ($(du -sh $BACKEND_DIR/git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5 | cut -f1))" || \
    echo "  daily_pv.h5 (debug): MISSING"

echo ""
echo "--- pip check (jangan ada warning 'No module named pip') ---"
python -m pip --version 2>&1

echo ""
echo "--- CUDA check ---"
python -c "import torch; print('  torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"

echo ""
echo "--- Quick import test ---"
PYTHONPATH=backend python -c "
from pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING
print('  settings OK:', ALPHA_AGENT_FACTOR_PROP_SETTING.latent_model_name)
" 2>&1 || echo "  Import test GAGAL — cek PYTHONPATH atau dependency"

echo ""
echo "============================================================"
echo " Setup selesai!"
echo ""
echo " Langkah selanjutnya:"
echo "   source /workspace/runpod_env.sh"
echo "   source /workspace/project/multi-agent-system/.venv/bin/activate"
echo "   cd /workspace/project/multi-agent-system"
echo "   PYTHONPATH=backend python launcher.py mine \\"
echo "     --direction 'price-volume momentum factor' \\"
echo "     --config_path configs/experiment.yaml"
echo "============================================================"
