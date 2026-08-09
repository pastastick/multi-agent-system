# Hasil Tahap 0 — Gumbel vs Ridge pada Kapasitas Kanal Laten

> Dijalankan 2026-08-09 di RunPod (A40 46GB, CUDA 12.8, torch 2.6.0+cu124),
> model `Qwen/Qwen3-8B`. Ini adalah gerbang keputusan Alt 3
> (`skripsi/alternatif_gumbel_latentmas.md` §6 Tahap 0) — satu-satunya
> pertanyaan yang dijawab: **apakah mengganti persamaan langkah laten
> LatentMAS dari ridge $W_a$ resmi paper ke relaksasi Gumbel-softmax menaikkan
> kapasitas kanal laten murni (`kv_latent_only`) untuk muatan simbolik?**

## 0. Ringkasan satu paragraf

**Ya, meyakinkan.** Pada `latent_steps=10` (setelan produksi), ridge $W_a$
resmi paper (Teorema A.1 LatentMAS) memberi kapasitas kanal laten murni
**recall = 0,000** — nol, bukan sekadar rendah — pada kedua jenis muatan (nama
fungsi DSL dan token acak), dan ini **tidak berubah** baik memakai matriks
ridge maupun tanpanya (mode `raw` dengan `use_realign=True` vs `False` — yaitu
konfigurasi *default* repo resmi LatentMAS — hasilnya identik bit-per-bit).
Gumbel-softmax memberi recall 0,350 (dsl) dan 0,190 (token) pada `m=10`, naik
ke 0,840 dan 0,760 pada `m=40` — keduanya signifikan secara statistik
(Wilcoxon p≤0,001, CI bootstrap 95% tak pernah menyentuh nol) — sementara
`raw` tetap presisi nol di `m=40` juga. Analisis tambahan (`soft`, proyeksi
convex-hull TANPA noise Gumbel) memisahkan efek "berada di manifold embedding"
dari efek "entropi tambahan": `soft` memulihkan hampir seluruh keunggulan
gumbel pada muatan `dsl` (0,340 vs 0,350, TIDAK signifikan, p=0,975) tetapi
jauh tertinggal pada muatan `token` (0,060 vs 0,190, signifikan, p=0,005).
**Kesimpulan gerbang**: `gumbel` > `raw` terbukti kuat → lanjut ke Tahap 1
(`alternatif_gumbel_latentmas.md` §6).

## 1. Setup

| komponen | nilai |
|---|---|
| GPU | NVIDIA A40, 46068 MiB, CUDA 12.8, driver 570.211.01 |
| Environment | uv + `.venv` Python 3.10, torch 2.6.0+cu124 (dari `pyproject.toml`) |
| Model | `Qwen/Qwen3-8B` (diunduh anonim dari HF Hub, ~16GB) |
| Skrip | `lab/channel_capacity.py` (A9, tak berubah logikanya — hanya ditambah `--no-realign` dan `load_dotenv`) |
| Payload | `dsl` (5 nama fungsi dari pustaka 71 fungsi produksi) dan `token` (5 pseudo-kata acak) |
| k, trials, seed | k=5, trials=20, seed=0 — identik dengan run `gumbel` yang sudah ada, supaya berpasangan |
| Lengan diuji | `kv_latent_only` (kanal laten murni — satu-satunya lengan yang menguji ekspresivitas vektor laten) |

Empat konfigurasi dijalankan:

| tag | `--latent-mode` | `--no-realign` | `--latent-steps` | makna |
|---|---|:-:|---:|---|
| `raw_m10` | `raw` | tidak | 10 | ridge $W_a$ aktif (Teorema A.1 LatentMAS), m=produksi |
| `raw_m40` | `raw` | tidak | 40 | ridge $W_a$ aktif, m besar |
| `raw_norealign_m10` | `raw` | **ya** | 10 | $M=I$ — **default resmi repo LatentMAS** (`--latent_space_realign` OFF) |
| `soft_m10` | `soft` | — | 10 | proyeksi convex-hull TANPA noise Gumbel (kontrol untuk memisahkan efek) |

Dibandingkan terhadap data `gumbel` yang sudah ada dari sesi sebelumnya
(`channel_capacity_Qwen_Qwen3-8B_m10.json`, `..._m40.json`), dengan
konfigurasi identik (k, trials, seed, payload) sehingga perbandingannya
**berpasangan** (paired) — muatan yang sama persis dipakai di semua mode.

## 2. Catatan implementasi

Dua bug/gap kecil ditemukan dan diperbaiki sebelum run:

1. **Secret leak**: `runpod_env.sh` (berkas *git-tracked*) memiliki token HF
   asli yang hardcoded (`export HF_TOKEN="hf_otarf...gcQv"`, commit `33dd614`).
   Token itu berstatus *expired* di Hub dan menyebabkan
   `RepositoryNotFoundError` bahkan untuk model publik (Qwen3, Apache-2.0)
   karena error 401 membuat transformers gagal total alih-alih jatuh ke akses
   anonim. Token dihapus dari berkas, diganti komentar yang menjelaskan token
   sebenarnya ada di `.env` (git-ignored) dan sudah otomatis dimuat. **Token
   lama itu sebaiknya di-revoke di huggingface.co/settings/tokens** karena
   sudah bocor ke riwayat git terlepas dari statusnya sekarang.
2. **`.env` tidak pernah dibaca `lab/*.py`**: hanya `launcher.py` dan `cli.py`
   yang memanggil `load_dotenv()`. Skrip di `lab/` berjalan dengan env shell
   apa adanya. Ditambahkan `load_dotenv(QL / ".env", override=False)` di
   awal `channel_capacity.py` (lihat komentar di kode) — tanpa menimpa
   variabel yang sudah di-export shell, supaya `HF_LOCAL_ONLY=1 python
   lab/...` tetap bisa memaksa mode offline.

Perubahan pada `lab/channel_capacity.py` (selain dua fix di atas):
`--no-realign` (mode `raw` tanpa matriks ridge), pencatatan `use_realign` di
`_meta`, dan nama berkas keluaran otomatis memuat `{mode}[_norealign]_m{m}`
supaya dua run dengan mode/m berbeda tidak saling menimpa.

Skrip baru `lab/compare_channel_modes.py` dibuat untuk analisis: memuat semua
`channel_capacity_*.json` yang ada, mengelompokkannya per sel eksperimen
(model, k, trials, m, seed), memverifikasi muatannya benar-benar identik
antar-mode (bukan diasumsikan), lalu menjalankan uji berpasangan (Wilcoxon
signed-rank untuk recall, McNemar eksak untuk exact-match) + CI bootstrap 95%.

## 3. Hasil lengkap

### 3.1. m=10 (setelan produksi) — empat mode dibandingkan

**Payload `dsl`** (n=20 tiap sel):

| mode | recall | exact | halusinasi | posisi (p1..p5) |
|---|---:|---:|---:|---|
| gumbel | **0,350** | 0,000 | 0,083 | 0,90 0,60 0,25 0,00 0,00 |
| soft | 0,340 | 0,000 | 0,105 | 0,80 0,70 0,20 0,00 0,00 |
| raw | 0,000 | 0,000 | 0,000 | 0,00 0,00 0,00 0,00 0,00 |
| raw(M=I) | 0,000 | 0,000 | 0,000 | 0,00 0,00 0,00 0,00 0,00 |

**Payload `token`** (n=20 tiap sel):

| mode | recall | exact | halusinasi | posisi (p1..p5) |
|---|---:|---:|---:|---|
| gumbel | **0,190** | 0,000 | 0,259 | 0,65 0,30 0,00 0,00 0,00 |
| soft | 0,060 | 0,000 | 0,050 | 0,25 0,05 0,00 0,00 0,00 |
| raw | 0,000 | 0,000 | 0,000 | 0,00 0,00 0,00 0,00 0,00 |
| raw(M=I) | 0,000 | 0,000 | 0,000 | 0,00 0,00 0,00 0,00 0,00 |

**Uji berpasangan** (Wilcoxon signed-rank untuk Δrecall, CI bootstrap 95%):

| perbandingan | payload | Δrecall | CI95% | p (Wilcoxon) |
|---|---|---:|---|---:|
| gumbel − raw | dsl | +0,350 | [+0,270, +0,430] | **<0,001** |
| gumbel − raw | token | +0,190 | [+0,120, +0,250] | **0,001** |
| gumbel − raw(M=I) | dsl | +0,350 | [+0,270, +0,430] | **<0,001** |
| gumbel − raw(M=I) | token | +0,190 | [+0,120, +0,250] | **0,001** |
| gumbel − soft | dsl | +0,010 | [−0,090, +0,120] | 0,975 (tak signifikan) |
| gumbel − soft | token | +0,130 | [+0,060, +0,200] | **0,005** |
| raw − raw(M=I) | dsl & token | 0,000 | [0,000, 0,000] | 1,000 (identik) |
| soft − raw | dsl | +0,340 | [+0,250, +0,420] | **<0,001** |
| soft − raw | token | +0,060 | [+0,020, +0,110] | **0,034** |

### 3.2. m=40 — gerbang WAJIB kedua

| mode | payload | recall | exact | halusinasi |
|---|---|---:|---:|---:|
| gumbel | dsl | **0,840** | 0,700 | 0,000 |
| raw | dsl | 0,000 | 0,000 | 0,000 |
| gumbel | token | **0,760** | 0,600 | 0,020 |
| raw | token | 0,000 | 0,000 | 0,000 |

| perbandingan | payload | Δrecall | CI95% | p | Δexact | p (McNemar) |
|---|---|---:|---|---:|---:|---:|
| gumbel − raw | dsl | +0,840 | [+0,690, +0,960] | **<0,001** | +0,700 | **<0,001** (14/0) |
| gumbel − raw | token | +0,760 | [+0,570, +0,920] | **<0,001** | +0,600 | **<0,001** (12/0) |

Pada m=40, gumbel bahkan mencapai **exact-match 0,70 (dsl) dan 0,60 (token)**
— memulihkan seluruh 5 item persis benar pada mayoritas percobaan — sementara
`raw` tetap presisi nol di exact maupun recall, di kedua m.

## 4. Tiga temuan

**(a) Kegagalan `raw` bukan soal matriks ridge-nya — ini gagal secara
struktural.** `raw` (ridge $W_a$ aktif) dan `raw(M=I)` (default resmi
LatentMAS, tanpa realignment sama sekali) menghasilkan angka **identik
bit-per-bit** di semua sel yang diuji (Δrecall=0,000, CI=[0,000, 0,000]).
Mengaktifkan atau menonaktifkan mekanisme realignment resmi paper tidak
mengubah apa pun — kanal tetap presisi nol. ini memperkuat (bukan
melemahkan) klaim dari `lab/AUDIT_KRITIS.md` §4.3 dan `HASIL_TAHAP4.md` §2:
pada Qwen3-8B, matriks ridge $W_a$ nyaris ortogonal terhadap masukannya
(cos=0,011) dan efeknya terhadap fidelitas simbolik kosong — bukan karena ia
diimplementasikan salah, tapi karena memetakan hidden state kembali ke ruang
token diskret lewat satu peta linear tunggal secara struktural tidak cukup,
baik dengan realignment maupun tanpanya.

**(b) `raw` tidak membaik dengan `latent_steps` lebih banyak — `gumbel`
membaik tajam.** Dari m=10 ke m=40, `raw` tetap 0,000→0,000 di kedua
payload, sedangkan `gumbel` naik 0,35→0,84 (dsl) dan 0,19→0,76 (token). Ini
bertentangan dengan intuisi "mungkin raw hanya butuh lebih banyak langkah" —
datanya menunjukkan raw punya **lantai keras di nol**, bukan sekadar lambat.

**(c) Disosiasi proyeksi vs entropi — keduanya berkontribusi, tapi untuk hal
berbeda.** `soft` (proyeksi convex-hull TANPA noise Gumbel) memulihkan hampir
seluruh keunggulan `gumbel` pada payload `dsl` (0,340 vs 0,350, selisih
`tidak` signifikan) tapi jauh tertinggal pada `token` (0,060 vs 0,190,
signifikan p=0,005). Payload `dsl` adalah nama fungsi nyata yang mungkin
sudah dekat dengan token vocab asli (`RANK`, `DELTA`, dst) — proyeksi ke
manifold saja cukup. Payload `token` adalah pseudo-kata yang benar-benar
tanpa prior — di situ entropi Gumbel memberi kontribusi independen dan
signifikan, konsisten dengan temuan lama (`b7_probe.py`) bahwa mode `soft`
menghasilkan vektor laten yang **identik/deterministik** (`hidden_identical:
true`, `cos: 1.0`) — mode-collapse yang membatasi keragaman yang bisa dibawa
kanal untuk muatan yang benar-benar baru.

## 5. Batas berlaku

- **n=20/sel, 1 seed.** Sama seperti A9 asli, ini bukan replikasi
  multi-seed. Variasi antar-seed tidak terukur (lihat batasan yang sama di
  `alternatif_fidelitas_simbol.md` §7).
- **Hanya `latent_steps` ∈ {10, 40}, hanya Qwen3-8B.** Belum diuji di
  backbone lain (4B/14B) atau nilai m lain (20, 80, 160 dari Figure 8 paper).
- **`kv_latent_only` di sini bukan replika `comm_mode` produksi apa pun** —
  ia isolasi kanal murni untuk tujuan pengukuran (lihat catatan kejujuran di
  docstring `channel_capacity.py`).
- **Tahap 0 ini TIDAK mengukur akurasi hilir (benchmark LatentMAS
  asli)** — hanya kapasitas kanal simbolik k=5. Tahap 1
  (`alternatif_gumbel_latentmas.md` §6) yang memindahkan pengujian ke
  benchmark bergaya paper (HumanEval+/MBPP+) belum dijalankan.
- **`raw` di sini SELALU dengan `latent_early_stop_cos=1.0` (nonaktif)** —
  sama dengan seluruh data `gumbel` lama, jadi perbandingannya adil, tapi
  berarti hasil ini tidak mencerminkan interaksi dengan early-stop (B6)
  produksi.

## 6. Keputusan gerbang

Sesuai kriteria di `README.md` §7 dan `alternatif_gumbel_latentmas.md` §6
Tahap 0:

> `gumbel` > `raw` meyakinkan → lanjut Tahap 1.

**Kriteria terpenuhi dengan sangat jelas** — bukan hanya lolos ambang
signifikansi, tapi dengan effect size besar (Δrecall 0,19–0,84) dan pola yang
konsisten di 2 nilai m × 2 payload × 2 varian raw (4 dari 4 perbandingan
gumbel-vs-raw signifikan di p≤0,001; hanya perbandingan gumbel-vs-soft pada
payload dsl yang tidak signifikan, dan itu justru temuan yang bermakna, lihat
§4c).

**Rekomendasi: lanjut ke Tahap 1** (`alternatif_gumbel_latentmas.md` §6):
probe simbolik pada tugas bergaya LatentMAS (subsample HumanEval+/MBPP+,
raw vs gumbel), lalu Tahap 2 (kontrol *gist* — GSM8K/MedQA subsample) untuk
menegakkan bentuk disosiasi (§4 dokumen itu): kolaborasi laten mungkin
memperbaiki keandalan/format tanpa memulihkan fidelitas simbolik penuh.

## 7. Cara mereproduksi

```bash
source /workspace/runpod_env.sh
source /workspace/project/multi-agent-system/.venv/bin/activate
cd /workspace/project/multi-agent-system
export PYTHONPATH=backend

# 4 run Tahap 0 (~5-8 menit GPU tiap satu)
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode raw \
    --latent-steps 10 --k 5 --trials 20 --seed 0
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode raw \
    --latent-steps 40 --k 5 --trials 20 --seed 0
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode raw \
    --no-realign --latent-steps 10 --k 5 --trials 20 --seed 0
python lab/channel_capacity.py --model Qwen/Qwen3-8B --latent-mode soft \
    --latent-steps 10 --k 5 --trials 20 --seed 0

# Analisis statistik berpasangan (Wilcoxon + McNemar + bootstrap CI)
python lab/compare_channel_modes.py --out lab/out/tahap0_analysis.json
```

Berkas mentah: `lab/out/channel_capacity_Qwen_Qwen3-8B_{raw_m10,raw_m40,
raw_norealign_m10,soft_m10}.json` (baru) + `..._m10.json`, `..._m40.json`
(gumbel, sudah ada sebelumnya). Ringkasan uji: `lab/out/tahap0_analysis.json`.
