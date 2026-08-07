# Hasil §8 AUDIT_KRITIS di GPU — Qwen3-8B (A40 46 GB)

**Tanggal**: 2026-08-07 · **Backbone utama**: `Qwen/Qwen3-8B` (permintaan user;
AUDIT_KRITIS ditulis untuk 4B) · **Kontrol**: `Qwen/Qwen3-4B`.

Semua angka dihasilkan di mesin ini; perintah reproduksinya ada di §12.
Skoring ekspresi memakai `lab/core.py`, divalidasi ulang hari ini:
`RANK($volume) * (TS_RANK($return,1) ? -1 : 1)` → IC = +0,04493, t = +6,72 —
identik dengan `0 - RANK($volume)` dan dengan angka yang dikutip AUDIT §2.1.

**Perubahan protokol atas permintaan user**: G1 memakai **60 langkah** (bukan 80);
uji cepat G2 memakai **{5, 10, 20, 40}** (nilai produksi 60 tetap dijalankan
sebagai pembanding); seluruh pengujian memakai **Qwen3-8B**.

**Metode yang membuat ini muat dalam anggaran GPU**: satu trajectory produksi
≈ 13 menit, mayoritasnya backtest LightGBM gabungan — metrik yang oleh AUDIT
§S3/B8 justru dinyatakan TIDAK boleh dipakai membandingkan mode. Jadi hanya
bagian yang butuh GPU (front-end LLM) yang dijalankan, lalu ekspresinya diskor
di CPU dengan metrik jujur (per-factor RankIC OOS). Totalnya **108 run front-end
→ 446 ekspresi** dalam ~4 jam GPU, dibanding ~23 jam kalau lewat jalur backtest
penuh. Pipeline lengkap (termasuk backtest Qlib) tetap divalidasi sekali secara
end-to-end — lihat §10.

---

## 0. Ringkasan

| # | Pertanyaan | Jawaban terukur |
|---|---|---|
| G1 | Apakah klaim §3.2 bertahan di backbone Qwen3? | **Satu dari tiga harus dicabut.** cos ke embedding terdekat **+0,162** (4B) / **+0,074** (8B), bukan −0,09. Titik tetap & entropi nol **bertahan dan lebih keras**. |
| G2 | `latent_steps` rendah: mutu tetap, degenerasi turun? | **Ya, jauh lebih tajam dari dugaan.** ls=5 → 6/6 run berhasil; ls=20 → 1/6; ls=40 & **ls=60 (nilai produksi) → 0/6**. |
| G3 | Apakah langkah laten `gumbel` memperbaiki? | **Ya untuk keandalan** (lolos gate 54%→91%, klaster sinyal 6→9). **Tidak untuk kekuatan sinyal** (t = −0,27). |
| G4 | Replikasi ≥3 seed per comm_mode | Selesai, n=6 run/mode. **Peringkat mode berbalik lagi**: `kv_and_text ≈ kv > text` pada \|IC\|/run — kebalikan AUDIT §3.3. |
| G5 | Laju tolak gate semantik pada run nyata | **31%** dari 302 kandidat — **di bawah ambang 60%** yang ditetapkan AUDIT. Kebocoran cacat-semantik = **0**. 3 kelas kebocoran BARU ditemukan. |
| G6 | Ablasi `use_realign` | **Kini sah di 8B** (untied). M menyimpang **104%** dari identitas di 8B vs **0,0002%** di 4B. End-to-end: **mematikan realign justru lebih baik** (6/6 vs 5/6 run, gate 83% vs 54%). |
| **G7** | *(tambahan)* Ruang laten saat BERTUMBUH lintas hop | **Sumber masalah yang selama ini tak terukur.** 59,5% prompt construct adalah duplikat verbatim; blok laten antar-agen 85% identik pada ls=60; menaikkan ls justru MENGECILKAN pengaruh kanal laten 4×. |
| **Q** | *(tambahan)* Ekspresi buruk: model bodoh atau prompt bermasalah? | **Keduanya, tapi pada sumbu berbeda — dan tak satu pun menjelaskan IC lemah.** Lihat §9. |

**Satu kalimat**: konfigurasi produksi saat ini (`latent.steps: 60`) membuat mode
`kv` dan `kv_and_text` **tidak menghasilkan satu ekspresi pun** di Qwen3-8B;
setelah diturunkan ke 5, sistem hidup kembali dan cakupan pencariannya naik dari
2 klaster sinyal (batch lama) menjadi 16.

---

## 1. G1 — dinamika rollout laten di backbone sebenarnya

`lab/latent_dynamics.py`, 60 langkah, 3 prompt arah × 3 seed, bfloat16.
`cos_emb` = kosinus vektor laten ke embedding token TERDEKAT; `beku@` = langkah
pertama saat cos(h_k, h_{k−1}) > 0,999.

### Qwen3-8B (`tie_word_embeddings: false` → realignment aktif)

| varian langkah laten | H akhir | cos antar-langkah | cos ke embedding | beku@ | token unik | identik antar-seed | identik antar-ARAH |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw` (tanpa realign) | 0,10 | 0,99884 | **+0,074** | **34** | 2/60 | 9/9 | **3/3** |
| **`raw_realign` (= produksi 8B)** | 5,99 | 0,95739 | +0,275 | — | 6/60 | **9/9** | 0/3 |
| `raw` + noise Gauss 0,1 | 0,11 | 0,99862 | +0,074 | 34 | 2/60 | 9/9 | 3/3 |
| `soft` T=1 | 1,10 | 0,83845 | +0,898 | — | 20/60 | 9/9 | 0/3 |
| `soft` T=2 | 0,03 | 0,99989 | +0,811 | 18 | 8/60 | 9/9 | 0/3 |
| **`gumbel` T=0,7** | 0,00 | **0,70788** | **+0,940** | — | **47/60** | **0/9** | 0/3 |
| `gumbel` T=1,0 | 0,00 | 0,53245 | +0,895 | — | 38/60 | 0/9 | 0/3 |
| `sample` T=1 | 0,00 | 0,62160 | +1,000 | — | 42/60 | 0/9 | 0/3 |

### Qwen3-4B (`tie_word_embeddings: true` → realignment = identitas)

| varian | H akhir | cos antar-langkah | cos ke embedding | beku@ | token unik | identik antar-seed | identik antar-ARAH |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`raw` (= produksi 4B)** | 2,96 | 0,99952 | **+0,162** | **12** | 5/60 | **9/9** | 1/3 |
| `raw_realign` | 3,18 | 0,99966 | +0,161 | 13 | 4/60 | 9/9 | 0/3 |
| `raw` + noise | 2,94 | 0,99952 | +0,163 | 13 | 5/60 | 1/9 | 1/3 |
| `soft` T=1 | 1,61 | 0,62390 | +0,872 | — | 36/60 | 9/9 | 0/3 |
| `soft` T=2 | 0,97 | 0,69705 | +0,916 | — | 46/60 | 9/9 | 0/3 |
| **`gumbel` T=0,7** | 0,58 | 0,65634 | **+0,962** | — | 44/60 | **0/9** | 0/3 |
| `gumbel` T=1,0 | 1,27 | 0,67361 | +0,914 | — | 38/60 | 0/9 | 0/3 |
| `sample` T=1 | 0,00 | 0,56132 | +1,000 | — | 40/60 | 0/9 | 0/3 |

### Yang harus diubah di AUDIT_KRITIS

1. **CABUT klaim "cos = −0,09, bukan sekadar jauh: berlawanan arah".** Itu angka
   GPT-2. Di Qwen3-4B **+0,162**, di Qwen3-8B **+0,074**. AUDIT sendiri menetapkan
   syaratnya ("kalau (a) tidak terkonfirmasi di Qwen3-4B, klaim §3.2 harus
   dicabut"). Yang bertahan adalah versi lebih lemah tapi masih kuat: vektor laten
   nyaris **ortogonal** terhadap setiap embedding token nyata (0,07–0,16),
   sementara `soft`/`gumbel` mencapai 0,87–0,96. "Di luar distribusi" tetap benar;
   "berlawanan arah" tidak.
2. **PERTAHANKAN & pertajam klaim titik tetap.** Di Qwen3-4B jalur laten membeku
   pada **langkah 12** → `latent_steps: 60` memberi ~12 pikiran lalu **48 salinan**.
3. **PERTAHANKAN klaim entropi nol.** 9/9 pasang seed identik di semua varian
   deterministik, di kedua backbone.
4. **TEMUAN BARU, lebih tajam dari ketiganya**: di Qwen3-8B tanpa realignment,
   jalur laten **identik untuk ketiga arah riset yang berbeda** (3/3). Rollout
   laten mentah bukan hanya kehilangan varians antar-seed — ia kehilangan
   **informasi arah**.

---

## 2. G2 — uji cepat `latent_steps` ∈ {5, 10, 20, 40} (+60 pembanding)

Mode `kv`, 2 arah × 3 seed = **6 run per nilai** (AUDIT meminta 2; dinaikkan
karena murah). "ada" = run yang menghasilkan ≥1 ekspresi; "hidup" = ber-IC dan
>2 nilai unik/hari; `unparse` = output construct tak terparse pada attempt pertama.

| latent_steps | run berhasil | ekspresi | lolos gate | cacat semantik | hidup | mean IC | mean \|IC\| | maks \|IC\| | klaster | unparse | KV di construct | detik/run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **5** | **6/6** | 33 | 58% | 21% | 22/33 | −0,0051 | 0,0119 | 0,0473 | **16** | **0/6** | 6 215 | 30 |
| **10** | 5/6 | 26 | 54% | 15% | 13/26 | −0,0114 | 0,0142 | **0,0515** | 6 | 1/6 | 6 574 | 29 |
| 20 | 1/6 | 1 | 100% | 0% | 1/1 | −0,0266 | 0,0266 | 0,0266 | 1 | 6/6 | 7 763 | 50 |
| 40 | **0/6** | 0 | — | — | — | — | — | — | — | 6/6 | 7 764 | 45 |
| **60 (produksi)** | **0/6** | 0 | — | — | — | — | — | — | — | 6/6 | 9 049 | 127 |

Dosis–respons monoton dan bersih. Prediksi AUDIT §8/G2 **terkonfirmasi, dan
efeknya jauh lebih besar dari yang diperkirakan**. Contoh keluaran construct pada
ls=60 (`lab/out/llm_outputs/`):

```
的

的
<tool_response>
的
```

Pola gagal yang sama dengan B10/B13 di MONITORING_NOTES — **dan ia bereproduksi
di model 8B**, jadi bukan soal kapasitas model. Konsekuensinya, B7
("inkonsistensi `latent.steps: 60` vs komentar 'disamakan ke 10'") naik status
dari "perlu disadari" menjadi **cacat konfigurasi yang mematikan mode kv**.

Catatan cakupan: ls=5 menemukan **16 klaster sinyal dari 22 faktor hidup**.
Batch lama (AUDIT §2.4) menemukan **2 klaster dari 39 faktor**. Satu baris config.

---

## 3. G3 — mengganti langkah laten: `raw` vs `gumbel` vs `sample`

Diimplementasikan di `llm/client.py::_CoreEngine._latent_step_vec`, dikendalikan
env `LATENT_STEP_MODE` / `LATENT_STEP_TEMP`, **default tetap `raw`** sehingga
perilaku produksi tidak berubah tanpa diminta:

```python
soft    z = softmax(W_out h / T) @ W_in                 # deterministik, in-distribution
gumbel  z = softmax((W_out h + Gumbel) / T) @ W_in      # + stokastik, knob entropi
sample  z = W_in[i],  i ~ softmax(W_out h / T)          # batas diskret gumbel
```

Semua tetap dinormalkan ke `target_norm`; KV, chat template, dan parser tak
tersentuh. Mode `kv`, ls=10, 2 arah × 3 seed:

| langkah laten | run berhasil | ekspresi | **lolos gate** | cacat | hidup | mean \|IC\| | maks \|IC\| | klaster | unparse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw` (produksi) | 5/6 | 26 | 54% | 15% | 13/26 | 0,0142 | **0,0515** | 6 | 1/6 |
| **`gumbel` T=0,7** | **6/6** | 35 | **91%** | **9%** | **27/35** | 0,0150 | 0,0326 | **9** | **0/6** |
| `sample` T=1,0 | 6/6 | 36 | 56% | 36% | 25/36 | 0,0141 | 0,0336 | 7 | 0/6 |

Dua hal yang harus dipisahkan:

- **Keandalan & cakupan: `gumbel` menang telak.** Lolos gate 54% → **91%**, cacat
  9%, faktor hidup 13 → **27**, klaster 6 → **9**. Ini menyerang B14 tepat di
  sasarannya: lintasan laten berbeda antar-seed (G1: 0/9 identik), sehingga tiga
  agen tak lagi memikirkan hal yang sama.
- **Kekuatan sinyal: tidak membaik.** 0,0142 → 0,0150, Welch t = −0,27 (tidak
  signifikan), dan faktor tunggal terbaik justru ada di `raw`.

`gumbel` memperbaiki **mesin pencarinya**, bukan **mutu sinyal yang tersedia di
data ini** — konsisten dengan AUDIT §2.5. `sample` (batas diskret) menaikkan yield
tapi memperburuk cacat semantik (36%), jadi yang berguna adalah sifat *kontinu +
stokastik*, bukan sekadar stokastik.

---

## 4. G4 — replikasi 3 seed × 2 arah per comm_mode

n = 6 run per mode (sebelumnya n = 1). `text` tidak memakai jalur laten sama
sekali sehingga `latent_steps` tak relevan untuknya.

| comm_mode | run berhasil | ekspresi | lolos gate | hidup | mean \|IC\| | **\|IC\| per-run** | klaster | detik/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `text` | **6/6** | 36 | **83%** | 21/36 | 0,0098 | **0,0084** | 13 | 67 |
| `kv` @ls10 | 5/6 | 26 | 54% | 13/26 | 0,0142 | **0,0152** | 6 | 29 |
| `kv_and_text` @ls10 | 4/6 | 22 | 50% | 9/22 | 0,0189 | **0,0158** | 6 | 99 |
| `kv_and_text` @ls60 | **0/6** | 0 | — | — | — | — | — | 73 |
| *(rujukan)* `kv` @ls5 | 6/6 | 33 | 58% | 22/33 | 0,0119 | 0,0122 | **16** | 30 |

Welch t pada \|IC\| per-run: `kv` vs `text` = **+2,62** (p ≈ 0,066);
`kv_and_text` vs `text` = +1,90; `kv` vs `kv_and_text` = −0,17.

**Ini kesimpulan terpenting untuk Bab 4, dan tidak nyaman:**

1. **Peringkat mode BERBALIK LAGI.** Bab 4 lama: `kv` terburuk. AUDIT §3.3 dengan
   \|IC\|: `text ≈ kv > kv_and_text`. Sekarang, dengan 8B + ls=10 dan n=6:
   `kv_and_text ≈ kv > text`, dan bedanya nyaris signifikan. Peringkat itu berubah
   setiap kali **backbone**, **metrik**, atau **`latent_steps`** berubah.
   Kesimpulan yang tidak tahan terhadap tiga hal itu sekaligus **tidak boleh
   dilaporkan sebagai temuan** — dan justru ITULAH temuan yang bisa dipertahankan.
2. **`text` menang di keandalan, kalah di mutu.** 6/6 run, 83% lolos gate, tanpa
   akumulasi KV sama sekali — tetapi \|IC\| per-run-nya paling rendah (0,0084).
   Mode KV menghasilkan lebih sedikit faktor tetapi lebih kuat. Sumbu "keandalan"
   dan "mutu" harus dilaporkan terpisah; melipatnya jadi satu angka adalah cara
   Bab 4 lama menghasilkan kesimpulan yang tak stabil.
3. **Semua lengan masih di bawah atau setara lantai acak** (mean \|IC\| acak =
   0,0170). Temuan pokok AUDIT §2.5 **bertahan pada model 8B**.

---

## 5. G5 — laju tolak gate semantik pada run nyata

Seluruh korpus front-end (G2+G3+G4+G6): **302 kandidat ekspresi, 94 ditolak (31%)**.

| sebab tolakan | jumlah | % dari tolakan |
|---|---:|---:|
| **SEMANTIK** (gate baru dari AUDIT §7) | **47** | **50%** |
| tak-parsable | 24 | 26% |
| regulator (SL / ER / duplikasi) | 18 | 19% |
| arity | 5 | 5% |

Rincian 47 tolakan semantik: kondisi non-boolean **43**, window degenerate 3,
ambang pada persentil 1.

**Keputusan sesuai kriteria AUDIT §8/G5**: ambangnya "kalau tolak > ~60%, prompt
DSL harus diperbaiki dulu". Terukur **31%** → **gate semantik boleh tetap aktif**;
tak ada ledakan putaran repair (repair hanya terpicu **8 dari 108 run**, karena ia
hanya jalan bila SEMUA ekspresi gagal gate).

**Efektivitas gate**: dari 198 ekspresi yang lolos, **0 masih cacat-semantik**.
Bandingkan batch lama (AUDIT §2.2): 49% ekspresi cacat, 21 di antaranya konstan
atau NaN total. Gate itu bekerja.

**Tiga kelas kebocoran BARU** (lolos gate lalu mati/gagal saat dieksekusi):

1. **Keluaran boolean 2-nilai.** `($volume > TS_ZSCORE($volume,5)) ? (-1) : (1)`
   → hanya 2 nilai unik per hari. Prompt sudah melarang ("must always be rankable,
   never a boolean signal") tetapi tak ada gate yang memeriksanya. **7 dari 198.**
2. **Argumen kuantil di luar [0,1].** `TS_QUANTILE($volume, 20, 5)` → crash
   `Quantile q must be in [0, 1], got 5.0`. `validate_semantics` memeriksa ambang
   *perbandingan* terhadap RANK/TS_RANK (`_PCT_FUNCS`) tetapi tidak memeriksa
   **argumen q** milik `TS_QUANTILE`/`PERCENTILE`. Diperparah urutan argumen yang
   tidak konsisten di prompt: `TS_QUANTILE(A, p, q)` vs `PERCENTILE(A, q, p)`.
3. **Gate kondisional yang menghasilkan kolom kosong.**
   `ZSCORE((TS_ZSCORE($volume,5) > 2) ? TS_PCTCHANGE($close,1) : 0)` → "empty
   after dropna/OOS". Sah secara sintaksis dan semantis, mati secara numerik.
   Hanya **eksekusi percobaan** yang bisa menangkapnya (AUDIT §S1 — belum dipasang;
   lihat RENCANA B12). **11 dari 198.**

**Lubang arity satu arah** (terpisah, prevalensi rendah): `validate_function_arity`
menurunkan `min_args` dari **signature Python**, bukan dari kontrak DSL di prompt.
Karena `TS_MEAN(df, p=5)` punya default, `TS_MEAN($close)` dinyatakan sah, lalu
Python diam-diam mengisi window = 5. Jadi ekspresi yang dievaluasi bukan ekspresi
yang ditulis model. Terukur **1 dari 129 ekspresi unik (1%)** — nyata tapi jarang.

---

## 6. G6 — ablasi `use_realign`

AUDIT §3.1 melarang ablasi ini di Qwen3-4B karena kedua cabang identik, dan
menyarankan Qwen3-8B. Karena user memang memakai 8B, ablasi ini sah.

### 6.1 Probe statis (`lab/realign_probe.py`, langsung dari bobot)

| | Qwen3-4B | Qwen3-8B |
|---|---:|---:|
| `tie_word_embeddings` | **true** | **false** |
| ‖M − I‖_F / ‖I‖_F | **1,4 × 10⁻⁶** (0,0002%) | **1,040** (104%) |
| cos(h, hM) untuk h acak | **1,000000** | **0,011** |
| ‖hM‖/‖h‖ | 1,000 | 0,306 |
| cos(baris W_in, baris W_out) | (tertaut) | **0,004** |

- **Qwen3-4B: konfirmasi penuh AUDIT §3.1.** Realignment = identitas + penskalaan.
- **Qwen3-8B: kebalikannya.** W_in dan W_out praktis ortogonal, sehingga M memutar
  hidden state hampir 90°. Baris kode yang SAMA adalah *no-op* di satu backbone
  dan *transformasi dominan* di backbone lain. Pesan metodologis untuk skripsi:
  hasil ablasi mekanisme LatentMAS **tidak dapat dipindahkan antar-backbone**
  tanpa mengecek `tie_word_embeddings` dulu.

### 6.2 End-to-end (mode `kv`, ls=10, 6 run per lengan)

| | run berhasil | ekspresi | lolos gate | hidup | mean \|IC\| | \|IC\|/run | klaster |
|---|---:|---:|---:|---:|---:|---:|---:|
| `use_realign=True` (produksi) | 5/6 | 26 | 54% | 13/26 | 0,0142 | 0,0152 | 6 |
| **`use_realign=False`** | **6/6** | 35 | **83%** | **30/35** | **0,0171** | 0,0171 | **13** |

Welch t = +0,45 (tidak signifikan) pada \|IC\|/run.

**Bacaan.** Pada `latent_steps=10`, **mematikan realignment sama baiknya atau lebih
baik** di setiap sumbu keandalan. Ini tidak bertentangan dengan G1: di sana `raw`
(tanpa M) runtuh ke titik tetap pada langkah 34 dan kehilangan informasi arah —
tetapi itu pada **ls=60**. Pada ls=10 rollout belum pernah membeku, jadi efek
protektif M tak pernah terpakai.

Rumusan yang bisa dipertahankan: **realignment hanya berguna ketika rollout cukup
panjang untuk membeku — dan rollout sepanjang itu justru yang harus dihindari
(G2).** Dengan `latent_steps` yang benar, kontribusi realignment tidak
terdemonstrasikan. Kombinasi G6.1 + G6.2 + G1 adalah pertanyaan sah untuk sidang
dan untuk penulis paper LatentMAS.

---

## 7. G7 (tambahan) — ruang laten saat BERTUMBUH lintas hop

### 7.1 Kenapa G1–G6 tidak menjawabnya

`lab/latent_dynamics.py` — dan AUDIT §3.2 — mengukur **satu agen dari prompt
segar**. Pipeline tidak begitu; ia menumpuk:

```
proposal : KV = [prompt_p]                            + L vektor laten
design   : KV = [ ...yang di atas ] + [prompt_d]      + L vektor laten
construct: KV = [ ...yang di atas ] + [prompt_c]      + L vektor laten  → EMIT
```

Agen yang menulis ekspresi justru yang berkonteks **terpanjang dan paling
tercemar**. `lab/latent_growth.py` mengukur sumbu itu: komposisi KV per hop,
duplikasi verbatim, dan **massa attention nyata** (dari `output_attentions`,
bukan proksi) saat token pertama di-emit.

### 7.2 Hasil (Qwen3-8B, mode `kv`)

| | ls=60 (produksi) | ls=10 |
|---|---:|---:|
| KV total di construct | **7 875 token** | 5 955 token |
| token laten | 180 (2,3%) | 30 (0,5%) |
| **prompt construct yang duplikat verbatim (8-gram)** | **59,5%** | **59,5%** |
| cos blok laten proposal↔design | **0,826** | 0,563 |
| cos blok laten proposal↔construct | **0,831** | 0,624 |
| cos blok laten design↔construct | **0,855** | 0,551 |

Massa attention token pertama yang di-emit (rata-rata 36 layer;
*enrichment* = massa ÷ porsi panjang, 1,00× = adil):

| segmen | ls=60 massa | ls=60 enrich | ls=10 massa | ls=10 enrich |
|---|---:|---:|---:|---:|
| proposal:prompt | 0,3268 | **2,93×** | 0,3333 | **2,26×** |
| proposal:latent | 0,0103 | 1,36× | 0,0033 | 1,97× |
| design:prompt | 0,1179 | 0,41× | 0,0933 | 0,24× |
| design:latent | 0,0184 | 2,41× | 0,0134 | **7,96×** |
| **construct:prompt** *(instruksi tugasnya sendiri)* | 0,1968 | **0,62×** | 0,1802 | **0,43×** |
| construct:latent | 0,0037 | **0,48×** | 0,0146 | **8,72×** |

### 7.3 Empat hal yang hanya terlihat pada sumbu pertumbuhan

**(a) 59,5% prompt construct adalah pengulangan verbatim.** Pustaka fungsi DSL
(~1,5k token) ditulis ULANG di system prompt `design` DAN `construct`
(`prompts.yaml` memuatnya tiga kali: design, construct, repair). Di
`comm_mode=text` tiap agen mulai dari konteks kosong → pustaka muncul **sekali**.
Di `kv`/`kv_and_text` semuanya menumpuk di SATU KV → blok hampir identik muncul
**2–3 kali dalam satu konteks**. Ini **satu-satunya perbedaan struktural besar**
antara mode yang jalan (`text`) dan yang kolaps — dan ini bukan efek 60 vektor
laten (hanya 2,3% konteks), melainkan efek ~1 500 token teks duplikat (~19%).

**(b) Menaikkan `latent_steps` justru MENGECILKAN pengaruh kanal laten.**
`construct:latent`: pada ls=10, **10 token** menarik massa **0,0146** (8,72×
porsinya); pada ls=60, **60 token** hanya menarik **0,0037** (0,48×). Enam kali
lebih banyak token laten → **empat kali lebih sedikit** attention. Penjelasannya
langsung dari G1: setelah titik tetap, vektor-vektor itu saling menjadi salinan;
softmax membagi massa di antara kunci yang hampir identik, dan peluruhan jarak
RoPE menghukum yang lama. "Berpikir lebih lama" di jalur laten bukan cuma
sia-sia — ia **melemahkan kanal yang jadi inti metode**.

**(c) Blok laten antar-agen saling menjadi salinan.** Kosinus antar-hop naik dari
**0,55–0,62 (ls=10)** menjadi **0,83–0,86 (ls=60)**. Pada `latent_steps: 60`,
"pikiran" proposal, design, dan construct **85% sama**. Ini rumusan mekanis yang
tepat untuk B14: bukan hanya antar-trajectory yang seragam, tetapi **antar-AGEN di
dalam satu trajectory**. Pipeline tiga agen berdegenerasi menjadi satu agen yang
mengulang dirinya.

**(d) Agen yang meng-emit paling sedikit memperhatikan instruksinya sendiri.**
`construct:prompt` mendapat 0,43–0,62× porsi adilnya, sementara `proposal:prompt`
— prompt tertua dan terjauh — mendapat 2,26–2,93×. Konsisten dengan awal konteks
bertindak sebagai *attention sink*, dan menjelaskan kenapa construct "lupa" format
JSON-nya padahal instruksinya ada di konteks.

### 7.4 Ini juga menjelaskan G2

Ketiga efek (a)–(c) tumbuh dengan `latent_steps` DAN dengan jumlah hop. Itulah
sebabnya kurva G2 monoton: pada ls=60 kanal laten kolaps jadi salinan (c),
kehilangan pengaruhnya (b), dan yang tersisa memandu emisi hanyalah konteks yang
19%-nya teks duplikat (a).

---

## 8. Audit persamaan & pembukuan pada jalur pertumbuhan

### 8.1 `kv_max_tokens` tidak pernah diberlakukan — anggaran KV = kode mati

`configs/experiment.yaml` menetapkan `kv_max_tokens: 20480`. Jejaknya:

- `pipeline/loop.py:229` membacanya ke `self._kv_max_tokens`;
- satu-satunya pemakaian berikutnya adalah **string log** di `loop.py:262`;
- `kv_truncate` di-import di `loop.py:15` tetapi **tidak pernah dipanggil**;
- `llm/client.py:1657` akan memanggilnya, tetapi hanya bila `_kv_max_seq_len`
  bukan None — dan `kv_max_seq_len=` **tidak pernah diteruskan oleh pemanggil mana
  pun** (`pipeline/settings.py::build_backend` tidak menyertakannya; terverifikasi
  dengan grep seluruh basis kode).

Ditambah `knn_enabled` yang **dimatikan otomatis** setiap kali `latent_steps > 0`
(`client.py::_CoreEngine.__init__`), maka **tidak ada satu pun kendali ukuran KV
yang aktif di dalam satu trajectory**. KV tumbuh apa adanya sampai 7–9k token
(terukur, §7.2) — konsisten dengan B1 (OOM di crossover) dan B2 (rambling sampai
konteks 40 960 penuh).

Diagram "KV-Cache Flow" di `METODE_QUANTALATENT.md` menuliskan
`kv_truncate(kv_feedback, kv_max_tokens) → _pipeline_kv (next iteration)`.
Kode melakukan sebaliknya: `loop.py:511` menyetel `self._pipeline_kv = None`.
Dokumen itu salah pada dua titik sekaligus. Pertumbuhan yang nyata terjadi
**di dalam** satu trajectory, bukan antar-iterasi.

### 8.2 `kv_truncate` merusak pembukuan posisi RoPE — TERBUKTI

```python
K_l ← K_l[..., -k:, :] ,  V_l ← V_l[..., -k:, :]
```

Key yang disimpan masih membawa **fase RoPE dari posisi absolut aslinya**
`[d, …, N−1]`, `d = N − k`. Tetapi panjang cache yang dilaporkan menjadi `k`, dan
transformers menetapkan posisi token berikutnya dari `get_seq_length()` = `k` —
seolah blok itu bermula di posisi 0. Karena attention RoPE bergantung pada
**selisih** posisi, setiap key lama tampak `d` posisi lebih dekat dari seharusnya.

Basis kode sudah mengakui kelas galat ini di jalur lain: `kv_knn_filter` memanggil
`_rerotate_keys_contiguous` persis untuk memperbaikinya. `kv_truncate` tidak.

**Bukti empiris** (`lab/kv_truncate_probe.py`, Qwen3-8B, konteks 2 761 token →
simpan 512, buang 2 249). Distribusi token berikutnya dibandingkan dengan rujukan
"konteks segar berisi 512 token yang sama":

| cara melanjutkan | KL ke rujukan | top-1 | sesuai rujukan? | entropi |
|---|---:|---|---|---:|
| **A. potong saja (kode sekarang)** | **5,09** | `" is"` | **tidak** | 3,24 |
| **B. potong + re-rotasi** | **0,94** | `" the"` | **ya** | 1,38 |
| C. rujukan (konteks segar) | 0 | `" the"` | — | 2,00 |

Pemotongan tanpa re-rotasi menggeser distribusi **5,4× lebih jauh** dari rujukan
dan mengubah token teratas; re-rotasi memulihkannya. (Sisa KL 0,94 pada B memang
diharapkan: K/V yang disimpan dulu dihitung sambil memperhatikan prefiks yang
dibuang — itu sifat pemotongan, bukan bug.)

**Status: galat nyata tetapi DORMAN** karena §8.1 — jalurnya tak pernah
tereksekusi. Bahayanya muncul kalau anggaran KV "diperbaiki" dengan menyambungkan
`kv_max_tokens`: itu akan **mengaktifkan** galat ini. Perbaiki dulu (RENCANA B8),
baru sambungkan (B9).

### 8.3 `kv_concat` — LatentMAS Eq. 4 tidak ada di jalur hidup

`kv_concat` didokumentasikan sebagai primitive *hierarchical working-memory
transfer* (LatentMAS Eq. 4) dan disebut di METODE §1.3 sebagai cara crossover
menggabungkan KV beberapa parent. Faktanya: **tidak ada satu pun pemanggil** di
seluruh basis kode (hanya muncul di docstring dan `__all__`). Dan kalaupun
dipakai, ia meng-concat sepanjang dim sekuens **tanpa re-rotasi**: key parent A
ter-rotasi di posisi 0…n−1, parent B di 0…m−1, tetapi B ditempatkan di n…n+m−1 —
dua rentang posisi tumpang-tindih dalam satu cache, sekelas §8.2 tapi lebih parah.

Konsekuensi untuk skripsi: klaim bahwa sistem mengimplementasikan transfer
working-memory hierarkis **tidak didukung jalur eksekusi**. Materi parent memang
ditransfer sebagai TEKS (`pipeline.py::run_evolution`), dan METODE §5.1 sudah
menyebut itu — tetapi §1.3 dan tabel operasi KV masih menyiratkan sebaliknya.

### 8.4 Yang BUKAN galat, tapi perlu dinyatakan

- **Realignment ridge diterapkan jauh di luar daerah fitnya.** M menyelesaikan
  `min_M ‖W_out M − W_in‖²_F` — pemetaan yang benar untuk vektor yang *berupa
  baris* `W_out`. Hidden state bukan itu: G1 mengukur kosinusnya ke embedding
  terdekat hanya 0,07–0,16. Jadi M adalah ekstrapolasi linear di wilayah yang tak
  pernah difit. Peta yang secara prinsip benar adalah
  `e = W_inᵀ softmax(W_out h)` — persis varian `soft`/`gumbel` (G3), yang memang
  mendarat di dalam convex hull (cos 0,87–0,96).
- **Normalisasi `z ← z/‖z‖ · target_norm` membuang skala.** Setiap token laten
  dipaksa bernorma sama, padahal norma embedding nyata bervariasi (sd 0,31 di 8B).
  Yang tersisa hanya arah — bagian yang justru paling tidak andal menurut poin
  di atas.
- **Token laten tidak punya kanal istimewa.** Ia token biasa dalam attention,
  tunduk pada peluruhan jarak RoPE seperti teks. Dengan 60 langkah × 3 hop, blok
  laten hop-1 berada ~5 000 posisi di belakang query yang meng-emit. Istilah
  "working memory" menyiratkan persistensi; mekanismenya hanyalah "lebih banyak
  token, yang meluruh dengan jarak".

---

## 9. Pertanyaan tambahan — model terlalu bodoh, atau prompt yang bermasalah?

Desain 2×2 penuh: **{Qwen3-4B, Qwen3-8B} × {prompt v0, prompt v1}**,
`comm_mode=text` (supaya jalur laten tidak ikut jadi variabel), 2 arah × 3 seed =
6 run per sel, 36 ekspresi per sel.

`prompts_v1.yaml` dibangkitkan `lab/make_prompts_v1.py` dan **hanya** menyentuh
kalimat yang oleh AUDIT §2.2/§S6 terbukti menanam kesalahan: RANK/TS_RANK
dinyatakan persentil [0,1]; syarat ternary wajib perbandingan eksplisit; window
≥ 2 (≥ 5 untuk statistik sebaran); larangan ambang absolut `$volume`; satu simbol
satu makna; tanpa markdown. Peran agen, format keluaran, dan alur **identik**.

| model | prompt | lolos gate | **cacat semantik** | hidup | mean \|IC\| | IC>0 | **\|IC\|/run** |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | v0 | 50% | **14%** | 22/36 | 0,0131 | 8/22 | 0,0139 |
| Qwen3-4B | v1 | 67% | **3%** | 21/36 | 0,0119 | 3/21 | 0,0118 |
| Qwen3-8B | v0 | 83% | **11%** | 21/36 | 0,0098 | 4/21 | 0,0084 |
| **Qwen3-8B** | **v1** | **86%** | **3%** | **25/36** | **0,0137** | **10/25** | **0,0137** |

**Jawabannya: keduanya, tetapi pada sumbu yang berbeda — dan tak satu pun
menjelaskan IC yang lemah.** "Ekspresi buruk" ternyata tiga masalah terpisah:

1. **Cacat SEMANTIK → penyebabnya PROMPT, bukan kapasitas model.**
   14% → 3% (4B) dan 11% → 3% (8B). Menaikkan ukuran model hampir tak berpengaruh
   (14% vs 11%); memperbaiki prompt memotongnya ~4× **pada kedua model**. Kelas
   cacat yang selama ini terbaca sebagai "model 4B terlalu bodoh"
   (`TS_RANK(...) < 50`, ternary non-boolean, window degenerate) **kita sendiri
   yang menanamnya lewat dokumentasi DSL yang menyesatkan**. AUDIT §S6 benar, dan
   kini terkuantifikasi.

2. **Legalitas / kemampuan parse → penyebabnya KAPASITAS MODEL.**
   Lolos gate 50% (4B v0) → 83% (8B v0): +33 poin hanya dari ukuran model.
   Perbaikan prompt hanya menambah +17 poin di 4B dan +3 poin di 8B. Menulis DSL
   yang well-formed (arity benar, kurung seimbang, nama fungsi ada) memang butuh
   kapasitas.

3. **Kekuatan sinyal → BUKAN keduanya.** Keempat sel berada di 0,0084–0,0139,
   semuanya **di bawah lantai acak 0,0170**. Yang paling menohok: **4B dengan
   prompt lama (0,0139) menyamai 8B dengan prompt baru (0,0137)**, dan **8B dengan
   prompt lama justru TERBURUK (0,0084)**. Tidak ada efek kapasitas yang monoton
   pada mutu sinyal. Ini menguatkan AUDIT §2.5 pada backbone baru: mutu sinyal
   adalah sifat **data × DSL**, bukan sifat model maupun prompt.

Catatan interaksi: v1 menaikkan \|IC\|/run di 8B (+63%) tetapi sedikit menurunkan
di 4B (−15%). Dengan n=6 ini belum signifikan; jangan diklaim.

**Konsekuensi praktis**: pakai prompt v1 (murah, memotong cacat 4×, tak ada
kerugian teridentifikasi), tetapi **jangan berharap prompt atau model yang lebih
besar menaikkan IC**. Untuk itu, ubah fungsi fitness (AUDIT §S2/S3: pakai \|IC\|
dengan tanda ditetapkan di jendela latih) — jauh lebih murah daripada memperbaiki
generatornya.

---

## 10. Status setup (README)

Seluruh langkah README §0–§6 dijalankan dan lolos verifikasi §6:

| item | status |
|---|---|
| `uv` di `/workspace/.local/bin/uv` | ✅ |
| `.venv` Python 3.10.12 di project root | ✅ |
| `torch 2.6.0+cu124`, `cuda: True`, A40 | ✅ |
| `qlib 0.9.7` + `mlflow 3.14.0` (kompatibel protobuf) | ✅ |
| `cn_data` ter-extract (6 016 instrumen) | ✅ |
| `daily_pv.h5` 380 MB + debug 1,4 MB | ✅ |
| model `Qwen/Qwen3-8B` (16 GB) + `Qwen/Qwen3-4B` (8 GB) di cache `/workspace` | ✅ |
| import `pipeline.settings`, `FrontEndPipeline`, `FactorRegulator` | ✅ |

Perubahan konfigurasi yang diterapkan sesuai permintaan user:
`configs/experiment*.yaml` dan `pipeline/settings.py` → `model_name:
"Qwen/Qwen3-8B"`. **`latent.steps` sengaja TIDAK diubah** (masih 60) meski G2
menunjukkan nilai itu mematikan mode kv — itu keputusan eksperimen milik user;
usulannya ada di `lab/RENCANA_PERBAIKAN.md` B1.

### 10.1 Validasi end-to-end (README §7), termasuk backtest Qlib

`configs/experiment_smoke.yaml` (1 arah, 1 ronde, 1 loop, `comm_mode=kv`,
`latent.steps=10`), dijalankan lewat jalur produksi penuh:

```
QUANTA_RUN_DIR=... PYTHONPATH=backend python launcher.py mine \
    --direction "price-volume momentum factor" --config_path configs/experiment_smoke.yaml
```

**rc=0.** `factor_propose` 28,6 s → 5 ekspresi lolos gate; `factor_backtest`
984 s; trajectory tersimpan di
`backend/runs/e2e_smoke_2026-08-07_05-49-13/trajectory_pool.json`.

**Revalidasi silang harness CPU pada run yang baru saja dibuat** — IC produksi vs
`lab/core.py`:

| faktor | IC produksi | IC lab (CPU) | selisih |
|---|---:|---:|---:|
| `..._0` | −0,022617 | −0,022617 | 0 |
| `..._1` | −0,034904 | −0,034904 | 1,2 × 10⁻¹² |
| `..._2` | **None** | **−0,024613** | — |
| `..._3` | −0,034952 | −0,034952 | 0 |
| `..._4` | −0,022620 | −0,022620 | 2,8 × 10⁻⁸ |

Dua hal: (1) replika CPU **identik dengan jalur produksi** sampai 10⁻⁸ pada run
yang sama sekali baru — jadi semua angka §1–§9 sah; (2) faktor `_2` tercatat
`factor_ic = None` di produksi padahal **punya IC nyata −0,0246** — persis
fenomena AUDIT §2.3 (correlation-gate menghapus nama dari `exp.factor_ic`),
kini terkonfirmasi langsung, bukan dari pool lama.

Dua mode gagal tambahan yang terlihat di run ini dan layak dicatat:
nama fungsi rusak akibat tokenisasi (`TS_PCT CHANGE`, `TSSTD` tanpa garis bawah)
pada attempt pertama construct, dan hipotesis yang kehilangan spasi
("mean-reversionfollowsinnext-periodreturns"). Keduanya adalah kelas kegagalan
yang akan dihapus oleh guided decoding (RENCANA B11).

---

## 11. Perubahan kode di sesi ini

| file | perubahan | mengubah perilaku default? |
|---|---|---|
| `llm/client.py` | `_CoreEngine._latent_step_vec()` + knob `LATENT_STEP_MODE`/`LATENT_STEP_TEMP` (G3) | **tidak** — default `raw` = perilaku lama |
| `lab/latent_dynamics.py` | dukung backbone untied, varian `raw_realign`, deteksi titik tetap, hemat memori untuk 8B | n/a (lab) |
| `lab/realign_probe.py` | `--model`, dukung untied, resolusi cache HF dari env | n/a (lab) |
| `lab/core.py` | tolak hasil eval non-numerik (dulu menjatuhkan seluruh loop skoring) | tidak |
| `lab/frontend_probe.py`, `gpu_suite.py`, `analyze_gpu.py`, `gate_report.py`, `latent_growth.py`, `kv_truncate_probe.py`, `make_prompts_v1.py` | BARU | n/a (lab) |
| `latent_mas/prompts_v1.yaml` | BARU, dibangkitkan; `prompts.yaml` tak disentuh | tidak |
| `configs/experiment*.yaml`, `pipeline/settings.py` | model 4B → 8B | ya (diminta user) |
| `configs/experiment_smoke.yaml` | BARU — 1 arah, 1 ronde, ls=10, untuk validasi end-to-end | tidak |

---

## 12. Reproduksi

```bash
cd /workspace/project/multi-agent-system
source /workspace/runpod_env.sh && unset HF_TOKEN     # token di repo sudah kedaluwarsa

# G1 (± 25 mnt/backbone)
.venv/bin/python lab/latent_dynamics.py --model Qwen/Qwen3-8B --steps 60 --device cuda --tag g1
.venv/bin/python lab/latent_dynamics.py --model Qwen/Qwen3-4B --steps 60 --device cuda --tag g1

# G6 probe statis (± 3 mnt)
.venv/bin/python lab/realign_probe.py --model Qwen/Qwen3-8B
.venv/bin/python lab/realign_probe.py --model Qwen/Qwen3-4B

# G2 / G3 / G4 / G6-e2e / pertanyaan prompt (satu proses per rencana; model dimuat sekali)
.venv/bin/python lab/gpu_suite.py --plan g2 --seeds 0,1,2 --directions d0,d1
.venv/bin/python lab/gpu_suite.py --plan g3 --ls 10 --seeds 0,1,2
.venv/bin/python lab/gpu_suite.py --plan g4 --ls 10 --seeds 0,1,2
.venv/bin/python lab/gpu_suite.py --plan g6 --ls 10 --seeds 0,1,2
.venv/bin/python lab/gpu_suite.py --plan prompt --model Qwen/Qwen3-8B --seeds 0,1,2
.venv/bin/python lab/gpu_suite.py --plan prompt --model Qwen/Qwen3-4B --seeds 0,1,2

# G7 pertumbuhan ruang laten (± 5 mnt/konfigurasi)
.venv/bin/python lab/latent_growth.py --model Qwen/Qwen3-8B --steps 60 --comm-mode kv
.venv/bin/python lab/latent_growth.py --model Qwen/Qwen3-8B --steps 10 --comm-mode kv

# bukti galat RoPE pada kv_truncate (± 5 mnt)
.venv/bin/python lab/kv_truncate_probe.py --model Qwen/Qwen3-8B --keep 512

# analisis (CPU)
.venv/bin/python lab/analyze_gpu.py --glob 'frontend_g2_*.json' --by latent_steps --clusters
.venv/bin/python lab/analyze_gpu.py --glob 'frontend_g4_*.json' --by comm_mode latent_steps
.venv/bin/python lab/analyze_gpu.py --glob 'frontend_px_*.json' --by model _prompts
.venv/bin/python lab/gate_report.py  --glob 'frontend_g*.json'  --by comm_mode latent_steps
```

Keluaran JSON/parquet ada di `lab/out/`. Rencana tindak lanjut:
`lab/RENCANA_PERBAIKAN.md`.
