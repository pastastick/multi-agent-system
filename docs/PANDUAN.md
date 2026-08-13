# Panduan Operasional QuantaLatent

> Dokumen ini adalah panduan menjalankan repo dalam bahasa Indonesia:
> peta repo, setup RunPod, cara menjalankan kedua lengan eksperimen, dan
> masalah yang sering muncul. Ringkasan berbahasa Inggris beserta hasil
> utamanya ada di [`../README.md`](../README.md).
>
> **Pertanyaan penelitiannya**: apakah keunggulan keluarga relaksasi diskret
> (`gumbel`/`moi`/`sample`) atas persamaan langkah laten resmi LatentMAS
> (`raw`, ridge $W_a$) — yang di Tahap 0 terbukti mutlak pada muatan
> **simbolik** — juga bertahan pada benchmark penalaran umum tempat LatentMAS
> asli dievaluasi? Lima persamaan × tiga medium komunikasi × empat tugas.
>
> **Baca dulu**: [`DESAIN_EKSPERIMEN.md`](DESAIN_EKSPERIMEN.md) (apa yang
> diukur dan kenapa) lalu [`HASIL_TAHAP0.md`](HASIL_TAHAP0.md) (angka yang
> mendasari rancangan ini).

Basis kode ini berasal dari **perombakan total** `exp/alt3-gumbel-fidelitas`
(`99721ec`) yang dikerjakan di branch `exp/empat-metode-v1` lalu dijadikan isi
`main`: `lab/` dilebur ke `backend/`, seluruh warisan RD-Agent/QuantaAlpha yang
tak terpakai dihapus (pipeline evolusi, CoSTEER, `core/`, agen eksternal,
loader dokumen — ±95 berkas Python), dan dua lengan eksperimen baru dibangun.
**Tidak ada kode yang hilang**: semuanya tetap ada di branch lama dan di
riwayat `main`.

---

## 1. Peta repo

```
backend/
  llm/        mesin LLM: model, KV-cache, latent_pass, _latent_step_vec   ← SUMBU A
  mas/        agen + operasi KV + pipeline rantai faktor                  ← SUMBU B
  bench/      lengan replikasi LatentMAS  (data · scoring · pipeline · run_bench · compare)
  factor/     lengan faktor alpha         (run_factor.py)
  dsl/        parser ekspresi · AST · pustaka fungsi (71 fungsi)
  gate/       gate mutu ekspresi: regulator, arity, redundansi, kompleksitas
  eval/       ic.py · backtest.py · stats.py · fidelity.py · channel_capacity.py
              compare_modes.py · realign_probe.py · b7_probe.py · rescore_all.py
  prompts/    factor.yaml (QuantaLatent) · bench.yaml (port LatentMAS)
  paths.py    jalur kanonik + bootstrap sys.path      qlog.py  logger (loguru)
configs/      matriks.yaml — daftar sel eksperimen (sumber kebenaran tunggal)
scripts/      gen_perintah.py (turunkan perintah dari matriks.yaml) ·
              jalankan_matriks.py (runner antrean bergerbang VRAM) ·
              hitung_token.py · kumpulkan_pendukung.py · plot_readme_figures.py
reference/    LatentMAS @9a9e4d3 · mixinputs @7aef34b (rujukan, READ-ONLY)
docs/         PANDUAN.md (berkas ini) · DESAIN_EKSPERIMEN.md · HASIL_TAHAP0.md ·
              HASIL_TAHAP4.md · AUDIT_KRITIS.md
results/      keluaran run — probe/ (artefak Tahap 0) · bench/ · factor/ · pendukung/
assets/       gambar README, dibangkitkan ulang oleh scripts/plot_readme_figures.py
```

Aturan import: `backend/` adalah root paket. Jalankan apa pun dengan
`PYTHONPATH=backend`, atau panggil skrip langsung — tiap skrip CLI memanggil
`paths.bootstrap()` sendiri.

---

## 2. Setup RunPod

Di RunPod **hanya `/workspace` yang persisten**; `/root` hilang saat pod
restart. Semua artefak (uv, `.venv`, cache HF, model) harus di bawah
`/workspace`.

```bash
# sekali per session SSH baru
source /workspace/runpod_env.sh          # salinan ada di repo: runpod_env.sh
cd /workspace/project/multi-agent-system
uv sync                                   # torch 2.6.0+cu124 dari index cu124
source .venv/bin/activate
```

Spesifikasi pod: A40 46 GB, volume disk ≥ 100 GB, CUDA ≥ 12.1. **Maksimal dua
sel paralel** — lihat §3 untuk angka VRAM terukur per lengan.

### `.env`

```env
HF_TOKEN=hf_...            # opsional; Qwen3 publik bisa diakses anonim
HF_HOME=/workspace/.cache/huggingface
HF_LOCAL_ONLY=0            # 0 = boleh unduh; 1 = paksa offline
```

> Jangan pernah menaruh token asli di berkas yang di-track git. Insiden token
> bocor di `runpod_env.sh` tercatat di `HASIL_TAHAP0.md` §2.

### Data pasar (hanya untuk lengan faktor)

```bash
cd /workspace/project/multi-agent-system/backend
hf download QuantaAlpha/qlib_csi300 --repo-type dataset --local-dir ./hf_data
python -c "import zipfile; zipfile.ZipFile('hf_data/cn_data.zip').extractall('data/qlib/')"
```

`backend/hf_data/daily_pv.h5` adalah satu-satunya berkas yang dibutuhkan
`eval/ic.py`; `data/qlib/cn_data/` dipakai kalau backtest Qlib penuh
dihidupkan lagi. Keduanya gitignored.

### Model

```bash
hf download Qwen/Qwen3-8B     # ~16 GB, atau biarkan terunduh saat run pertama
```

---

## 3. Menjalankan

Satu proses = **satu sel** matriks. Ini disengaja: satu sel tidak menghabiskan
GPU sendirian, jadi beberapa sel dijalankan bersamaan.

Pemakaian VRAM **terukur** di A40 46 GB (`nvidia-smi --query-compute-apps`,
10 Agustus 2026):

| lengan | VRAM per proses | slot paralel aman |
|---|---:|---|
| bench (`bench/run_bench.py`, `--max-new-tokens 2048`) | ~18 GB | 2 |
| faktor (`factor/run_factor.py`, `--max-new-tokens 4096`, rantai 3 agen) | ~21 GB | 2 (= 93% VRAM, tanpa margin) |

> ⚠️ `HASIL_TAHAP0.md` §8.7 menulis "~16 GB → 2–3 run paralel". Angka itu
> berasal dari probe/bench pendek dan **tidak berlaku untuk lengan faktor**;
> menyetel 3 sel faktor paralel akan OOM. Pakai `scripts/jalankan_matriks.py`
> yang punya gerbang VRAM (`--vram-bebas-min`, satuan MiB) dan antre-ulang
> otomatis untuk sel yang OOM (`--ulang-maks`).

### Lengan 1 — benchmark ala LatentMAS

```bash
PYTHONPATH=backend python backend/bench/run_bench.py \
    --task gsm8k --latent-mode gumbel --comm-mode kv \
    --limit 100 --sample-seed 0 --seed 0
```

`--task` ∈ `gsm8k` (math) · `arc_challenge` (commonsense) · `humanevalplus` (code)
`--latent-mode` ∈ `raw` `gumbel` `moi` `sample` (+ kontrol `soft`)
`--comm-mode` ∈ `kv` `kv_and_text` `text`; `--baseline` = agen tunggal

> `--sample-seed` HARUS sama di semua sel — itu yang membuat semua metode
> melihat soal yang sama dan uji berpasangannya sah. `bench/compare.py`
> memverifikasinya lewat sidik jari dan **mengeluarkan** sel yang tak cocok.

### Lengan 2 — faktor alpha (simbolik/DSL)

```bash
PYTHONPATH=backend python backend/factor/run_factor.py \
    --comm-mode kv --latent-mode gumbel --latent-steps 10 \
    --seeds 0,1,2 --directions d0,d1 --tag kv_gumbel
```

### Turunkan seluruh matriks dari config

```bash
python scripts/gen_perintah.py --arm bench                 # 36 sel
python scripts/gen_perintah.py --arm factor                # 11 sel

# Runner antrean: menjaga jumlah slot, menunggu VRAM bebas sebelum start sel
# baru, dan mengantre ulang sel yang OOM di belakang antrean.
python scripts/jalankan_matriks.py --arm all --slots 2
python scripts/jalankan_matriks.py --arm all --dry-run     # lihat rencananya dulu
```

### Analisis (tanpa GPU)

```bash
python backend/bench/compare.py --out results/bench/analisis.json   # McNemar + CI bootstrap
PYTHONPATH=backend python backend/eval/rescore_all.py               # skor ulang korpus faktor
python backend/eval/compare_modes.py                                # probe kapasitas kanal (Tahap 0)
PYTHONPATH=backend python backend/eval/backtest.py                  # smoke metrik portofolio
```

---

## 4. Verifikasi setup (CPU, tanpa GPU)

```bash
PYTHONPATH=backend python -c "
import llm.client, mas.pipeline, bench.pipeline, gate, dsl.expr_parser, eval.ic
print('import ok')
from eval.ic import Lab; lab = Lab(mode='fast')
print(lab.ic('RANK(\$volume)'))         # ~ <IC=-0.04493 t=-6.72 n=243 ...>
"
```

Jalur ini diverifikasi setelah perombakan: 243 hari OOS, ~4370 saham/hari,
IC identik dengan angka produksi lama.

---

## 5. Ganti model untuk VRAM terbatas

| Model | Unduh | VRAM bobot |
|---|---|---|
| `Qwen/Qwen3-4B` | ~8 GB | ~8 GB |
| **`Qwen/Qwen3-8B`** (dipakai skripsi) | ~16 GB | ~16 GB |
| `Qwen/Qwen3-14B` | ~28 GB | ~28 GB |

Ganti lewat `--model` di kedua runner dan `model:` di `configs/matriks.yaml`.
**Seluruh angka skripsi dipatok Qwen3-8B** — mencampur backbone membuat sel tak
sebanding.

---

## 6. Masalah yang sering muncul

**`uv` / `.venv` / model HF hilang setelah pod restart** — semuanya di `/root`
yang ephemeral. Pastikan `source /workspace/runpod_env.sh` dijalankan SEBELUM
`uv sync`, dan `XDG_*`/`UV_CACHE_DIR`/`HF_HOME` menunjuk ke `/workspace`.

**`We couldn't connect to 'https://huggingface.co'`** — set `HF_LOCAL_ONLY=0`
di `.env`, atau pre-download modelnya. Kalau `HF_TOKEN` di-set tapi sudah
kedaluwarsa, error 401 membuat transformers gagal total alih-alih jatuh ke
akses anonim — hapus tokennya.

**CUDA OOM** — turunkan `--max-new-tokens`, kurangi proses paralel, atau turun
ke Qwen3-4B. Cek `--empty-cache-every` di `run_bench.py`.

**`ModuleNotFoundError`** — jalankan dengan `PYTHONPATH=backend` dari root repo,
bukan dari dalam `backend/`.

**Skoring korpus faktor kena OOM di mesin kecil** — `eval/ic.py` membatasi
worker joblib lewat `LAB_MAX_WORKERS` (default 3); turunkan ke 1 bila perlu.

---

## 7. Apa yang dihapus saat perombakan

Semuanya masih ada di `exp/alt3-gumbel-fidelitas`, branch `prod/*`, dan di
riwayat `main` sebelum commit perombakan.

| dihapus | alasan |
|---|---|
| `backend/pipeline/` (evolusi, loop, planning, factor_mining) | lengan faktor kini single-pass; evolusi menambah variabel perancu |
| `backend/coder/costeer/`, `backend/core/` | kerangka RD-Agent; hanya `core/conf.py` yang tersisa → `backend/conf.py` |
| `backend/log/` (wrapper `rdagent.log`) | diganti `backend/qlog.py` (loguru polos) — repo tak lagi butuh RD-Agent |
| `backend/factors/` selain DSL + regulator + template | proposal/runner/feedback/qlib terikat ke pipeline lama |
| `backend/eksternal/`, `app/`, `components/`, `debug/`, `experiments/` | agen makro/berita & harness lama, di luar pertanyaan branch ini |
| `backend/runs/`, `backend/log/<ts>/`, `try/`, `books/` | artefak run lama (±440 MB) |
| `configs/experiment*.yaml`, `backtest.yaml` | dibaca pipeline evolusi yang dihapus; diganti `configs/matriks.yaml` |
| `launcher.py`, `backend/cli.py`, `main.py` | CLI RD-Agent; diganti dua runner + `scripts/gen_perintah.py` |
