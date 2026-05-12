# QuantaLatent — Panduan Setup RunPod

Proyek ini adalah adaptasi QuantaAlpha dengan **Latent-MAS pipeline**: model lokal (Qwen3) menjalankan latent reasoning via KV-cache, lalu factor mining berjalan sepenuhnya di GPU tanpa API eksternal untuk step utama.

---

## 0. Catatan Penting RunPod — `/workspace` vs `/root`

Di RunPod, **hanya direktori `/workspace` yang persisten** (network storage). Semua yang ada di `/root` (HOME default) akan hilang ketika pod di-stop atau di-restart. Ini berarti:

- ❌ JANGAN install `uv`, `.venv`, atau dependency apa pun di `~/` (`/root/`).
- ✅ Semua artifact (binary `uv`, virtualenv `.venv`, torch wheels, HuggingFace model cache, pip cache, uv cache) **harus** berada di bawah `/workspace/`.
- ✅ Project root proyek ini: **`/workspace/quantalatent`**.

Sebelum apa pun, set environment variable berikut **di awal setiap session SSH baru** (atau tambahkan ke `~/.bashrc` — tapi `~/.bashrc` sendiri tidak persisten, jadi simpan juga salinannya di `/workspace/runpod_env.sh`):

```bash
# /workspace/runpod_env.sh — sumber file ini di awal setiap session
export HOME_REAL=$HOME                                          # cadangan, jika perlu

# uv & cargo binary location (uv installer default ke ~/.local/bin → ephemeral)
export XDG_DATA_HOME=/workspace/.local/share
export XDG_CONFIG_HOME=/workspace/.config
export XDG_CACHE_HOME=/workspace/.cache
export PATH=/workspace/.local/bin:$PATH

# uv cache & virtualenv
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.local/share/uv/python
export UV_TOOL_DIR=/workspace/.local/share/uv/tools

# pip cache (untuk fallback jika tidak pakai uv)
export PIP_CACHE_DIR=/workspace/.cache/pip

# HuggingFace model & dataset cache (default ~/.cache/huggingface → ephemeral)
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub

# Torch hub & inductor cache
export TORCH_HOME=/workspace/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor

# Project-specific
export PYTHONPATH=/workspace/quantalatent/backend
```

Buat sekali, lalu di setiap session baru cukup:

```bash
source /workspace/runpod_env.sh
```

> Tip: tambahkan `source /workspace/runpod_env.sh` ke `~/.bashrc` agar otomatis tiap login. Karena `~/.bashrc` ephemeral, simpan juga template-nya di `/workspace/bashrc.template` dan re-copy setelah pod restart.

---

## 1. Spesifikasi Pod RunPod

| Komponen | Minimum | Rekomendasi |
|---|---|---|
| GPU | RTX 4090 (24 GB VRAM) | A100 40 GB |
| RAM | 32 GB | 64 GB |
| Container Disk | 30 GB | 50 GB |
| **Volume Disk (`/workspace`)** | **100 GB** | **200 GB** |
| Python | 3.10 | 3.10 |
| CUDA | 11.8+ | 12.1+ |

> **Catatan model**: `Qwen3-14B` butuh ~28 GB VRAM (float16). Untuk 4090 24 GB, gunakan `Qwen3-4B` (~8 GB VRAM). Lihat bagian [Ganti Model](#5-ganti-model-untuk-vram-terbatas).

> **Volume disk = network storage RunPod**, mount otomatis di `/workspace`. Pastikan ukurannya cukup untuk: model HF (~8 GB Qwen3-4B atau ~28 GB Qwen3-14B) + dataset Qlib (~3 GB) + venv (~10 GB) + cache.

---

## 2. Transfer Proyek ke `/workspace`

Dari mesin lokal (WSL), kirim folder proyek via `rsync` ke `/workspace`:

```bash
# Ganti <user>@<host>:<port> dengan kredensial SSH RunPod kamu
rsync -avz -e "ssh -p <port>" \
  --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='data/' --exclude='hf_data/' --exclude='log/' \
  /root/projects/first-experiment/quantalatent/ \
  <user>@<host>:/workspace/quantalatent/
```

> Data Qlib dan HDF5 tidak perlu di-transfer karena akan didownload langsung di RunPod.

---

## 3. Install Dependencies di RunPod (semuanya ke `/workspace`)

```bash
# WAJIB: load env var dulu agar uv/.venv/cache semua ke /workspace
source /workspace/runpod_env.sh

cd /workspace/quantalatent

# Install uv ke /workspace/.local/bin (BUKAN ~/.local/bin)
curl -LsSf https://astral.sh/uv/install.sh | \
  env UV_INSTALL_DIR=/workspace/.local/bin sh

# Verifikasi uv terinstall di /workspace
which uv   # harus: /workspace/.local/bin/uv

# Buat venv di /workspace/quantalatent/.venv dan install semua deps
uv sync

# Aktivasi venv
source .venv/bin/activate

# Verifikasi venv aktif dari /workspace
which python   # harus: /workspace/quantalatent/.venv/bin/python
```

> Karena `XDG_*` dan `UV_CACHE_DIR` sudah diarahkan ke `/workspace/.cache`, semua wheel cache, python interpreter, dan tool uv tidak akan menyentuh `/root`.

### 3a. Pastikan torch cocok dengan driver CUDA

`uv sync` akan menarik torch versi default (saat ini `2.11.0+cu130`) yang **butuh driver NVIDIA ≥ 580**. Driver lama (mis. RunPod RTX 4090 dengan driver 570 → CUDA 12.8) menyebabkan `torch.cuda.is_available() == False` dan model di-load ke CPU sehingga pipeline macet di `Workflow Progress: 0/5`.

Cek dulu:

```bash
nvidia-smi | grep "CUDA Version"
.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available(), '| torch:', torch.__version__)"
```

Jika `cuda: False`, install ulang torch sesuai versi driver (cache wheel tersimpan di `/workspace/.cache/uv`):

```bash
# Driver support CUDA 12.8 (driver ≥ 525)
uv pip install --reinstall \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# Driver support CUDA 12.6
uv pip install --reinstall \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
```

Verifikasi: `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"` harus mengembalikan `True`.

---

## 4. Konfigurasi `.env`

```bash
cd /workspace/quantalatent
cp configs/.env.example .env
```

Edit `.env` sesuai environment RunPod:

```bash
nano .env   # atau vim, code, dsb.
```

Isi minimal yang **wajib** diset (semua path di bawah `/workspace`):

```env
# === Paths ===
QLIB_DATA_DIR=/workspace/quantalatent/backend/data/qlib/cn_data
QLIB_PROVIDER_URI=/workspace/quantalatent/backend/data/qlib/cn_data

# === Workspace & cache (auto-derive dari lokasi file jika tidak di-set) ===
WORKSPACE_PATH=/workspace/quantalatent/backend/data/results/workspace
PICKLE_CACHE_FOLDER_PATH_STR=/workspace/quantalatent/backend/data/results/pickle_cache

# === Data HDF5 untuk factor mining (auto-derive jika tidak di-set) ===
FACTOR_CoSTEER_DATA_FOLDER=/workspace/quantalatent/backend/git_ignore_folder/factor_implementation_source_data
FACTOR_CoSTEER_DATA_FOLDER_DEBUG=/workspace/quantalatent/backend/git_ignore_folder/factor_implementation_source_data_debug

# === HuggingFace (cache otomatis ke /workspace/.cache/huggingface via HF_HOME) ===
HF_TOKEN=hf_...

# === LLM API (opsional jika latent_enabled=true) ===
# Dibutuhkan hanya jika latent.enabled=false di experiment.yaml
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://your-llm-provider/v1
CHAT_MODEL=your-model-name
REASONING_MODEL=your-model-name
```

> Variabel `WORKSPACE_PATH`, `PICKLE_CACHE_FOLDER_PATH_STR`, `FACTOR_CoSTEER_DATA_FOLDER`, dan `FACTOR_CoSTEER_DATA_FOLDER_DEBUG` **tidak wajib** diset jika kamu tidak mengubah struktur folder — nilainya otomatis di-derive dari lokasi instalasi.

> Jika `latent.enabled: true` di `configs/experiment.yaml` (default), pipeline menggunakan model lokal Qwen3 — API key tidak diperlukan untuk step utama (propose, construct, feedback).

---

## 5. Download Data Qlib + HDF5

> Semua perintah di bagian ini dijalankan dari `/workspace/quantalatent/backend/`. Folder `backend/data/`, `backend/hf_data/`, dan `backend/git_ignore_folder/` di-ignore oleh git (lihat `.gitignore`).

### 5a. Download dataset dari HuggingFace

```bash
source /workspace/runpod_env.sh
source /workspace/quantalatent/.venv/bin/activate

uv pip install huggingface_hub   # jika belum ada

cd /workspace/quantalatent/backend

# Download semua file sekaligus (cache otomatis ke /workspace/.cache/huggingface)
hf download QuantaAlpha/qlib_csi300 \
  --repo-type dataset \
  --local-dir ./hf_data
```

### 5b. Extract dan tempatkan Qlib data

```bash
# cwd: /workspace/quantalatent/backend
mkdir -p data/qlib
unzip hf_data/cn_data.zip -d data/qlib/
# Hasil: backend/data/qlib/cn_data/ berisi calendars/, features/, instruments/
```

### 5c. Tempatkan HDF5 untuk factor mining

```bash
# cwd: /workspace/quantalatent/backend
mkdir -p git_ignore_folder/factor_implementation_source_data
mkdir -p git_ignore_folder/factor_implementation_source_data_debug

cp hf_data/daily_pv.h5 \
   git_ignore_folder/factor_implementation_source_data/daily_pv.h5

cp hf_data/daily_pv_debug.h5 \
   git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5
```

> `daily_pv_debug.h5` harus di-rename jadi `daily_pv.h5` di folder debug.

### 5d. Buat folder output

```bash
# cwd: /workspace/quantalatent/backend
mkdir -p data/results
mkdir -p log
mkdir -p debug/llm_outputs
```

---

## 6. Verifikasi Setup

```bash
source /workspace/runpod_env.sh
source /workspace/quantalatent/.venv/bin/activate

cd /workspace/quantalatent

# Cek semua artifact ada di /workspace, BUKAN /root
which uv          # /workspace/.local/bin/uv
which python      # /workspace/quantalatent/.venv/bin/python
echo $HF_HOME     # /workspace/.cache/huggingface
echo $UV_CACHE_DIR # /workspace/.cache/uv

# Cek struktur data
ls backend/data/qlib/cn_data/          # harus ada: calendars/ features/ instruments/
ls backend/git_ignore_folder/factor_implementation_source_data/  # harus ada: daily_pv.h5

# Test import
PYTHONPATH=backend python -c "from pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING; print('OK:', ALPHA_AGENT_FACTOR_PROP_SETTING.latent_model_name)"

# Cek GPU
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

---

## 7. Jalankan Factor Mining

```bash
source /workspace/runpod_env.sh
source /workspace/quantalatent/.venv/bin/activate
cd /workspace/quantalatent

# Mode standar (latent pipeline dengan Qwen3)
PYTHONPATH=backend python launcher.py mine \
  --direction "price-volume momentum factor" \
  --config_path configs/experiment.yaml

# Dengan arah custom
PYTHONPATH=backend python launcher.py mine \
  --direction "microstructure alpha from bid-ask spread" \
  --config_path configs/experiment.yaml
```

Progress disimpan di `log/` dan `data/results/`. Factor library tersimpan di `all_factors_library*.json`.

### Jalankan di background (agar tidak terputus saat koneksi SSH putus):

```bash
nohup PYTHONPATH=backend python launcher.py mine \
  --direction "price-volume momentum factor" \
  --config_path configs/experiment.yaml \
  > log/mining_run.log 2>&1 &

echo "PID: $!"
tail -f log/mining_run.log
```

---

## 8. Jalankan Backtest

```bash
source /workspace/runpod_env.sh
source /workspace/quantalatent/.venv/bin/activate
cd /workspace/quantalatent

# Backtest dengan factor library hasil mining
PYTHONPATH=backend python launcher.py backtest \
  --factor-source custom \
  --factor-json all_factors_library.json \
  -c configs/backtest.yaml

# Backtest dengan Alpha158 baseline
PYTHONPATH=backend python launcher.py backtest \
  --factor-source alpha158_20 \
  -c configs/backtest.yaml

# Dry run (cek factor tanpa backtest)
PYTHONPATH=backend python launcher.py backtest \
  --factor-source custom \
  --factor-json all_factors_library.json \
  -c configs/backtest.yaml \
  --dry-run
```

---

## 9. Ganti Model untuk VRAM Terbatas

Edit `configs/experiment.yaml`, bagian `latent:`:

```yaml
latent:
  enabled: true
  model_name: "Qwen/Qwen3-4B"   # ganti dari Qwen3-14B ke 4B
  device: "cuda"
  steps: 2
  max_new_tokens: 1024           # kurangi untuk hemat VRAM
  kv_max_tokens: 1024            # kurangi untuk hemat VRAM
```

| Model | VRAM | Keterangan |
|---|---|---|
| `Qwen/Qwen3-4B` | ~8 GB | Cocok untuk RTX 4090 |
| `Qwen/Qwen3-14B` | ~28 GB | Perlu A100 40 GB |

---

## 10. Konfigurasi Experiment

File utama konfigurasi: `configs/experiment.yaml`

Parameter penting yang sering diubah:

```yaml
execution:
  max_loops: 2          # jumlah iterasi mining per direction
  steps_per_loop: 5     # step per loop (propose/construct/calculate/backtest/feedback)

evolution:
  enabled: true         # aktifkan evolutionary exploration
  max_rounds: 3         # jumlah ronde evolusi (Original + Mutation + Crossover)

planning:
  enabled: true
  num_directions: 2     # berapa arah eksplorasi yang di-generate

latent:
  enabled: true         # true = gunakan local Qwen3, false = gunakan API eksternal
  model_name: "Qwen/Qwen3-4B"
  kv_max_tokens: 2048   # kurangi jika OOM
```

---

## 11. Troubleshooting

### Setelah pod restart, `uv` / `.venv` / model HF "hilang"

**Penyebab**: artifact tersimpan di `/root` (ephemeral), bukan `/workspace`. Cek:

```bash
ls /workspace/.local/bin/uv             # harus ada
ls /workspace/quantalatent/.venv/bin    # harus ada
ls /workspace/.cache/huggingface/hub    # harus ada model snapshot
```

Jika kosong, ulangi bagian [0](#0-catatan-penting-runpod--workspace-vs-root) dan [3](#3-install-dependencies-di-runpod-semuanya-ke-workspace) — pastikan `source /workspace/runpod_env.sh` dijalankan **sebelum** install apa pun.

### Pipeline stuck di `Workflow Progress: 0/5`, `nvidia-smi` 0 MiB

**Gejala**: log berhenti tepat setelah `__init__ took ...s` lalu progress bar muncul tapi tidak pernah maju. `nvidia-smi` menunjukkan 0 MiB usage dan "No running processes found" padahal `python launcher.py mine ...` masih berjalan.

**Penyebab**: torch yang terinstall tidak kompatibel dengan driver NVIDIA — `torch.cuda.is_available()` mengembalikan `False`, model fallback ke CPU sehingga generate praktis terbekukan.

**Diagnosa**:

```bash
.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available(), '| torch:', torch.__version__)"
nvidia-smi | grep "CUDA Version"
```

Kalau `cuda: False` dan versi torch (misal `cu130`) lebih tinggi dari yang didukung driver, install ulang torch sesuai bagian [3a](#3a-pastikan-torch-cocok-dengan-driver-cuda).

### Pesan `No module named pip` di stderr

Bising tidak fatal. Berasal dari library pihak ketiga (vllm `collect_env`, numba `sysinfo`) yang shell-out memanggil `python -m pip list` untuk diagnostik. Venv yang dibuat lewat `uv` memang tidak menyertakan modul `pip`. Untuk diam-kan:

```bash
uv pip install pip
```

### CUDA Out of Memory

```bash
# Kurangi kv_max_tokens dan max_new_tokens di experiment.yaml
# Atau ganti model ke Qwen3-4B
# Cek VRAM usage:
nvidia-smi -l 1
```

### ImportError / ModuleNotFoundError

```bash
# Pastikan PYTHONPATH sudah di-set (sesuaikan dengan path proyek kamu)
export PYTHONPATH=/workspace/quantalatent/backend
# Atau jalankan selalu dari root proyek dengan prefix PYTHONPATH=backend
PYTHONPATH=backend python launcher.py mine ...
```

### HDF5 file not found

```bash
# Pastikan file ada di path yang benar
ls backend/git_ignore_folder/factor_implementation_source_data/daily_pv.h5
# Env var override path HDF5 (opsional):
export FACTOR_CoSTEER_DATA_FOLDER=/custom/path/to/factor_source_data
```

### Qlib data tidak lengkap

```bash
# Pastikan ada features/ folder
ls backend/data/qlib/cn_data/features/ | head -5
# Jika kosong, re-extract cn_data.zip
```

### SSH terputus saat mining

Gunakan `nohup` seperti di bagian 7, atau gunakan `screen`/`tmux`:

```bash
tmux new -s mining
# jalankan mining di dalam tmux
# Ctrl+B, D untuk detach
# tmux attach -t mining untuk kembali
```

### Disk `/workspace` penuh

```bash
# Cek disk usage breakdown
du -sh /workspace/.cache/* /workspace/quantalatent/* | sort -h

# Bersihkan cache uv & pip yang tidak terpakai
uv cache prune
rm -rf /workspace/.cache/pip/*

# Bersihkan model HF lama (hati-hati, akan re-download saat dipakai lagi)
rm -rf /workspace/.cache/huggingface/hub/models--<old-model>
```

---

## Struktur Folder (setelah setup)

```
/workspace/                              # PERSISTENT network storage
├── runpod_env.sh                        # env vars (HF_HOME, UV_CACHE_DIR, dll)
├── .local/bin/uv                        # binary uv (BUKAN ~/.local/bin)
├── .cache/
│   ├── uv/                              # uv wheel cache
│   ├── pip/                             # pip cache (fallback)
│   ├── huggingface/hub/                 # model & dataset cache HF
│   └── torch/                           # torch hub cache
└── quantalatent/                        # PROJECT ROOT
    ├── .venv/                           # virtualenv (BUKAN ~/.venv)
    ├── .env                             # konfigurasi environment (dari configs/.env.example)
    ├── launcher.py                      # entry point utama
    ├── configs/
    │   ├── experiment.yaml              # parameter experiment & latent pipeline
    │   ├── backtest.yaml                # parameter backtest
    │   └── .env.example                 # template konfigurasi environment
    ├── backend/
    │   ├── pipeline/
    │   │   ├── factor_mining.py         # orchestrator utama
    │   │   └── settings.py              # konfigurasi class pipeline
    │   ├── llm/
    │   │   └── client.py                # LocalLLMBackend (Qwen3 inference)
    │   ├── core/
    │   │   └── conf.py                  # RDAgentSettings (workspace/cache paths)
    │   ├── data/                        # [git-ignored]
    │   │   ├── qlib/cn_data/            # Qlib market data (hasil unzip cn_data.zip)
    │   │   │   ├── calendars/
    │   │   │   ├── features/
    │   │   │   └── instruments/
    │   │   └── results/                 # output experiment (auto-created)
    │   │       ├── workspace/           # temp workspace per coding task
    │   │       └── pickle_cache/        # cache komputasi
    │   ├── hf_data/                     # [git-ignored] cache HuggingFace dataset
    │   ├── log/                         # [git-ignored] trace logs per iterasi
    │   ├── debug/llm_outputs/           # [TRACKED] snapshot per LLM call (JSON)
    │   └── git_ignore_folder/           # [git-ignored]
    │       ├── factor_implementation_source_data/
    │       │   └── daily_pv.h5          # HDF5 price-volume data (398 MB)
    │       └── factor_implementation_source_data_debug/
    │           └── daily_pv.h5          # HDF5 debug subset (1.4 MB)

/root/                                   # EPHEMERAL — JANGAN simpan apa pun penting di sini
```
