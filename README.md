# QuantaAlpha — Panduan Setup JarvisLabs

Proyek ini adalah adaptasi QuantaAlpha dengan **Latent-MAS pipeline**: model lokal (Qwen3) menjalankan latent reasoning via KV-cache, lalu factor mining berjalan sepenuhnya di GPU tanpa API eksternal untuk step utama.

---

## 1. Spesifikasi Instance JarvisLabs

| Komponen | Minimum | Rekomendasi |
|---|---|---|
| GPU | RTX 4090 (24 GB VRAM) | A100 40 GB |
| RAM | 32 GB | 64 GB |
| Disk | 100 GB | 200 GB |
| Python | 3.10 | 3.10 |
| CUDA | 11.8+ | 12.1+ |

> **Catatan model**: `Qwen3-14B` butuh ~28 GB VRAM (float16). Untuk 4090 24 GB, gunakan `Qwen3-4B` (~8 GB VRAM). Lihat bagian [Ganti Model](#5-ganti-model-untuk-vram-terbatas).

---

## 2. Transfer Proyek ke JarvisLabs

Dari mesin lokal (WSL), kirim folder proyek via `rsync`:

```bash
# Ganti <user>@<host> dengan kredensial SSH JarvisLabs kamu
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='data/' --exclude='hf_data/' \
  /root/projects/first-experiment/ai-agent/ \
  <user>@<host>:/workspace/ai-agent/
```

> Data Qlib dan HDF5 tidak perlu di-transfer karena akan didownload langsung di JarvisLabs.

---

## 3. Install Dependencies di JarvisLabs

```bash
cd /workspace/ai-agent

# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # atau source ~/.zshrc

# Buat venv dan install semua dependensi dari lockfile
uv sync

# Atau jika tidak pakai uv:
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Setelah `uv sync`, aktivasi venv dengan: `source .venv/bin/activate`

---

## 4. Konfigurasi `.env`

```bash
cd /workspace/ai-agent
cp configs/.env.example .env
```

Edit `.env` sesuai environment JarvisLabs:

```bash
nano .env   # atau vim, code, dsb.
```

Isi minimal yang **wajib** diset:

```env
# === Paths ===
QLIB_DATA_DIR=/workspace/ai-agent/data/qlib/cn_data
DATA_RESULTS_DIR=/workspace/ai-agent/data/results
QLIB_PROVIDER_URI=/workspace/ai-agent/data/qlib/cn_data

# === HuggingFace ===
HF_TOKEN=.....

# === LLM API (opsional jika latent_enabled=true) ===
# Dibutuhkan hanya jika latent.enabled=false di experiment.yaml
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://your-llm-provider/v1
CHAT_MODEL=your-model-name
REASONING_MODEL=your-model-name
```

> Jika `latent.enabled: true` di `configs/experiment.yaml` (default), pipeline menggunakan model lokal Qwen3 — API key tidak diperlukan untuk step utama (propose, construct, feedback).

---

## 5. Download Data Qlib + HDF5

### 5a. Download dataset dari HuggingFace

```bash
cd /workspace/ai-agent

pip install huggingface_hub   # jika belum ada

# Download semua file sekaligus
huggingface-cli download QuantaAlpha/qlib_csi300 \
  --repo-type dataset \
  --local-dir ./hf_data
```

### 5b. Extract dan tempatkan Qlib data

```bash
mkdir -p data/qlib
unzip hf_data/cn_data.zip -d data/qlib/
# Hasil: data/qlib/cn_data/ berisi calendars/, features/, instruments/
```

### 5c. Tempatkan HDF5 untuk factor mining

```bash
mkdir -p backend/git_ignore_folder/factor_implementation_source_data
mkdir -p backend/git_ignore_folder/factor_implementation_source_data_debug

cp hf_data/daily_pv.h5 \
   backend/git_ignore_folder/factor_implementation_source_data/daily_pv.h5

cp hf_data/daily_pv_debug.h5 \
   backend/git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5
```

> `daily_pv_debug.h5` harus di-rename jadi `daily_pv.h5` di folder debug.

### 5d. Buat folder output

```bash
mkdir -p data/results
mkdir -p log
mkdir -p debug/llm_outputs
```

---

## 6. Verifikasi Setup

```bash
# Aktivasi venv
source .venv/bin/activate

# Cek struktur data
ls data/qlib/cn_data/          # harus ada: calendars/ features/ instruments/
ls backend/git_ignore_folder/factor_implementation_source_data/  # harus ada: daily_pv.h5

# Test import
cd /workspace/ai-agent
PYTHONPATH=backend python -c "from pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING; print('OK:', ALPHA_AGENT_FACTOR_PROP_SETTING.latent_model_name)"

# Cek GPU
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

---

## 7. Jalankan Factor Mining

```bash
cd /workspace/ai-agent
source .venv/bin/activateF

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
cd /workspace/ai-agent
source .venv/bin/activate

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
  model_name: "Qwen/Qwen3-14B"
  kv_max_tokens: 2048   # kurangi jika OOM
```

---

## 11. Troubleshooting

### CUDA Out of Memory

```bash
# Kurangi kv_max_tokens dan max_new_tokens di experiment.yaml
# Atau ganti model ke Qwen3-4B
# Cek VRAM usage:
nvidia-smi -l 1
```

### ImportError / ModuleNotFoundError

```bash
# Pastikan PYTHONPATH sudah di-set
export PYTHONPATH=/workspace/ai-agent/backend
# Atau jalankan selalu dari /workspace/ai-agent dengan prefix PYTHONPATH=backend
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
ls data/qlib/cn_data/features/ | head -5
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

---

## Struktur Folder (setelah setup)

```
/workspace/ai-agent/
├── .env                          # konfigurasi environment
├── launcher.py                   # entry point utama
├── configs/
│   ├── experiment.yaml           # parameter experiment & latent pipeline
│   └── backtest.yaml             # parameter backtest
├── backend/
│   ├── pipeline/
│   │   ├── factor_mining.py      # orchestrator utama
│   │   └── settings.py           # konfigurasi class pipeline
│   ├── llm/
│   │   └── client.py             # LocalLLMBackend (Qwen3 inference)
│   └── git_ignore_folder/
│       ├── factor_implementation_source_data/
│       │   └── daily_pv.h5       # HDF5 price-volume data (398 MB)
│       └── factor_implementation_source_data_debug/
│           └── daily_pv.h5       # HDF5 debug subset (1.4 MB)
├── data/
│   ├── qlib/cn_data/             # Qlib market data
│   │   ├── calendars/
│   │   ├── features/
│   │   └── instruments/
│   └── results/                  # output experiment
├── log/                          # trace logs per iterasi
└── debug/llm_outputs/            # JSONL log output LLM per call
```
