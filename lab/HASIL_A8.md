# A8 — ablasi agen: apakah rantai `proposal → design → construct` layak dipertahankan?

**Dijalankan**: 2026-08-07, Qwen3-8B, A40 46 GB, branch `exp/rencana-perbaikan`.
**Konfigurasi**: hasil Tahap 1 (`comm_mode=kv`, `latent_steps=10`, `step_mode=gumbel`),
gate Tahap 2 aktif (B15 + B12), prompt Tahap 2 (B4).
**Reproduksi**: `python lab/gpu_suite.py --plan a8 --comm-mode kv --ls 10 --seeds 0,1,2 --directions d0,d1`

Dokumen ini menjawab dua pertanyaan yang selama ini belum pernah diajukan ke sistem:

1. **Apakah agen `design` membayar biayanya?** (A8/B13)
2. **Kalau tidak, slot itu diisi apa?** — usulan user: agen yang tugasnya
   BERINOVASI, bebas dari keharusan sesuai teori ekonomi, asal legal secara DSL
   dan menaikkan IC. (B16)

Kriteria keputusan ditulis di `RENCANA_PERBAIKAN.md §Tahap 3a` **sebelum** data ada.

---

## 0. Kenapa pertanyaan kedua layak diuji — bukan sekadar selera

Keluhan "ekspresi yang dihasilkan monoton dan standar" bisa diukur. Dari korpus
446 ekspresi yang dihasilkan sistem sebelum eksperimen ini (G2+G3+G4+G6+2×2 prompt):

| pengukuran | angka |
|---|---|
| fungsi DSL yang **tak pernah dipakai sekali pun** | **28 dari 55** |
| fungsi yang mendominasi | TS_ZSCORE, TS_PCTCHANGE, TS_STD, RANK |
| ekspresi dengan pembungkus terluar yang sama (RANK/ZSCORE/TS_ZSCORE) | **59%** |
| ekspresi dengan hanya 2–3 pemanggilan fungsi | 78% |
| mean \|IC\| lengan LLM terbaik | 0,0189 |
| **mean \|IC\| ekspresi ACAK dari DSL yang sama** | **0,0170** |

Baris terakhir yang menentukan arah. Ekspresi acak — tanpa hipotesis, tanpa
teori, tanpa agen — berada di kisaran yang sama dengan, dan pada sebagian besar
lengan MENGUNGGULI, keluaran rantai agen yang seluruh promptnya tentang mekanisme
ekonomi. Kalau begitu yang langka bukan pembenaran teoretis, melainkan **cakupan
struktural**. Itu membuat usulan user bukan tebakan, melainkan hipotesis yang
sejalan dengan bukti yang sudah ada di repo ini.

---

## 1. Lengan yang diuji

Lima lengan, 6 run masing-masing (2 arah × 3 seed), satu variabel berubah antar
pasangan lengan:

| lengan | rantai | klem kesetiaan di emitter | yang diisolasi |
|---|---|---|---|
| `full` | proposal→design→construct | ON | rantai produksi (rujukan) |
| `nodesign` | proposal→construct | ON | kontribusi `design` |
| `direct` | construct sendirian | ON | nilai seluruh hulu |
| `innovate` | proposal→**innovate**→construct | **OFF** | usulan pengganti, utuh |
| `innovate_fid` | proposal→**innovate**→construct | ON | memisahkan efek agen dari efek klem |

Lengan `innovate_fid` ada karena mengganti `design` dengan `innovate` mengubah DUA
hal sekaligus: agennya, dan klausa "FIDELITY FIRST … Variety lives inside the
hypothesis, never outside it" di prompt emitter. Menaruh agen yang tugasnya
membelokkan hipotesis di hulu emitter yang diperintahkan setia kepada hipotesis
adalah dua perintah yang saling meniadakan. Tanpa lengan pemisah ini, tak ada
klaim kausal yang bisa dipertahankan.

---

## 2. Catatan pelaksanaan yang harus ikut dilaporkan

Tiga hal ditemukan SAAT menjalankan A8, dan semuanya mengubah angka. Dicatat di
sini supaya angka di §3 dibaca dengan benar.

**(a) Lengan `direct` sempat diberi prompt yang berbohong.** Prompt emitter
membuka dengan "The hypothesis and the candidate-function palette from the
Proposal and Design agents…". Pada lengan `direct` kedua agen itu tidak ada, jadi
emitter disuruh membaca sesuatu yang tak pernah ada. Lengan itu diulang dengan
cabang prompt sendiri (`from_direction`); cabang lengan lain dipastikan merender
IDENTIK sehingga tetap sebanding.

**(b) Jalur retry emitter diam-diam mengembalikan rezim kesetiaan.** Saat output
construct tak terparse, retry dijalankan tanpa meneruskan `free_form` — sehingga
retry di lengan innovate diuji pada rezim yang salah (terlihat dari panjang
prompt: 1598 token pada percobaan pertama, 1627 pada retry, persis panjang lengan
`full`). Sudah diperbaiki.

**(c) Guided decoding (B11) tidak sekadar kode mati — ia tidak bisa jalan.**
Replika integrasi transformers di `llm/guided_decoding.py` ditulis untuk varian
`TokenEnforcerTokenizerData` 5-argumen, sedangkan versi terpasang (0.10.12)
menerima 3 dan `get_allowed_tokens` mengembalikan `List[int]` langsung. Dua lengan
guided pertama karena itu gagal 6/6 run dengan `TypeError`, dan kegagalan itu
sempat terbaca seolah "guided decoding menghasilkan nol ekspresi". Setelah
diperbaiki, constraint terbukti bekerja (dekode terbatas inkremental memaksa
`{"hypothesis" :"","factors" : [ {"name" :"",…`).

Pelajaran umumnya: **kode mati tidak sama dengan kode yang siap dipakai.** Jalur
yang tak pernah dipanggil siapa pun juga tak pernah teruji terhadap versi pustaka
yang benar-benar terpasang.

---

## 3. Hasil

`python lab/decide_a8.py --comm-mode kv`

| lengan | run produktif | ekspr | lolos gate | hidup | **\|IC\|/run** | pustaka | fungsi baru | klaster | dtk/faktor | token/faktor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | **6/6** | 35 | 83% | 25 | 0,0109 | 13 | 3 | **20** | 7,0 | 1095 |
| `nodesign` | **6/6** | 35 | 89% | 33 | 0,0087 | 12 | 0 | 7 | **5,8** | **902** |
| `direct` | 3/6 | 18 | 89% | 16 | 0,0110 | 7 | 0 | 7 | 20,6 | 2265 |
| **`innovate`** | 4/6 | 23 | 87% | 14 | **0,0174** | **23** | **7** | 8 | 9,5 | 1482 |
| `innovate_fid` | **6/6** | 33 | 70% | 25 | 0,0153 | 16 | 2 | 14 | 7,4 | 1272 |
| *(lanjutan)* `full_guided` | 5/6 | 29 | 62% | 17 | 0,0117 | 15 | 4 | 12 | 50,1 | 2354 |

Fungsi yang **belum pernah dipakai sistem sebelum A8**:
- `full` → DELTA, EMA, FILTER
- `nodesign`, `direct` → tidak ada
- **`innovate` → BB_LOWER, BB_UPPER, DECAYLINEAR, FILTER, INV, SEQUENCE, SKEW**
- `innovate_fid` → DECAYLINEAR, FILTER

### 3.1 Terhadap lantai acak — satuan per-ekspresi

Ini pembanding yang paling penting dan paling sering salah dibaca. Lantai acak
(`lab/random_baseline.py`, 271 ekspresi hidup) = mean \|IC\| **0,0170**.

| lengan | mean \|IC\| per-ekspresi hidup | Mann-Whitney vs acak |
|---|---:|---|
| `nodesign` | 0,0085 | **jauh di bawah** (z=−4,30, p<0,001) |
| `direct` | 0,0106 | di bawah (z=−1,93, p=0,054) |
| `full` | 0,0112 | **di bawah** (z=−2,07, p=0,039) |
| `full_guided` | 0,0119 | di bawah (z=−1,85, p=0,064) |
| `innovate_fid` | 0,0157 | setara (z=−1,51, p=0,131) |
| **`innovate`** | **0,0167** | **setara** (z=−1,00, p=0,316) |

**Yang boleh diklaim**: rezim inovasi adalah satu-satunya yang **menutup jarak**
ke lantai acak. Rantai produksi (`full`) secara statistik LEBIH BURUK daripada
mengambil ekspresi acak dari DSL yang sama; rezim inovasi tidak lagi begitu.

**Yang TIDAK boleh diklaim**: bahwa ia MENGALAHKAN lantai acak. Ia hanya menyamai.
Menyamai pencarian acak bukan kemenangan mutlak — tetapi bagi sistem yang selama
ini kalah, itu perubahan tanda, dan itu temuan yang sah.

### 3.2 Tiga hal yang dijawab tabel ini

**(a) `design` menyumbang KERAGAMAN, bukan kuantitas.** `nodesign` menghasilkan
lebih banyak faktor hidup (33 vs 25) tetapi semuanya menumpuk di **7 klaster
sinyal**, sementara `full` menyebar ke **20**. Menghapus `design` menghasilkan
lebih banyak faktor yang pada dasarnya sinyal yang sama. Ini kebalikan dari dugaan
awal saat hanya melihat laju lolos gate.

**(b) Agen inovasi bekerja HANYA sebagai paket.** Bandingkan dua lengan yang
agennya identik dan hanya berbeda pada klem kesetiaan di emitter:

| | pustaka | fungsi baru | \|IC\|/run | lolos gate |
|---|---:|---:|---:|---:|
| `innovate` (klem OFF) | 23 | 7 | 0,0174 | 87% |
| `innovate_fid` (klem ON) | 16 | 2 | 0,0153 | 70% |

Memasang agen yang tugasnya membelokkan hipotesis sambil tetap memerintahkan
emitter setia pada hipotesis menghasilkan yang terburuk dari keduanya: eksplorasi
tercekik DAN laju lolos gate turun. Kalau ide ini diadopsi, ia harus diadopsi utuh.

**(c) Rantai pendek `direct` gagal karena FORMAT, bukan karena pendek.** 3 dari 6
run tak menghasilkan apa pun, dan semuanya karena JSON tak terparse (bukan karena
ekspresinya jelek). Biaya per faktornya paling mahal (20,6 detik) justru karena
run yang gagal tetap membakar waktu.

### 3.3 Kenapa `innovate` hanya 4/6 — penyebabnya diketahui

Dua run yang gagal (keduanya seed=1) bukan menghasilkan ekspresi buruk; emitter
menjawab **`"I understand the instruction."`** lalu berhenti (2–6 token). Prompt
`innovate` padat instruksi-meta ("what you are free from / what you are bound
by"), dan emitter melanjutkan register itu alih-alih menulis faktor.

Ini kegagalan FORMAT, dan penting untuk tidak salah dibaca sebagai "ide inovasi
tidak stabil". Bukti pendukungnya: `innovate_fid` — agen yang sama, prompt hulu
yang sama — mencapai 6/6. Yang berbeda hanya prompt emitter.

### 3.4 B11 (guided decoding) — hasil negatif yang jujur

`full_guided` vs `full`: laju lolos gate **turun** 83% → 62%, cacat semantik
muncul (7 dari 29), biaya melonjak **7×** (7,0 → 50,1 detik per faktor diterima),
dan model menghalusinasikan nama fungsi yang tak ada (`TS_RESIDUAL`).

Grammar JSON menjamin **bentuk**, bukan **isi** — persis peringatan yang ditulis
di RENCANA B11. Kriteria lulus Tahap 2 ("laju tak-terparse → 0") memang tercapai,
tetapi dengan mengorbankan sumbu yang lebih penting. **B11 tidak direkomendasikan
sebagai default**; ia berguna sebagai jaring pengaman pada rantai yang terbukti
sering gagal parse (mis. `direct`), bukan sebagai setelan global.

---

## 4. Keputusan (aturan §Tahap 3a, diterapkan mekanis oleh `lab/decide_a8.py`)

**Gerbang 1 — apakah `design` berkontribusi?**
mean \|IC\|/run: `full` 0,0109 vs `nodesign` 0,0087 (selisih +0,0022, Welch
t=+0,79). Karena |t| < 1, kriteria 1 **TERPENUHI** → secara aturan, `design`
dinyatakan **tidak berpengaruh signifikan terhadap IC**.

**Gerbang 2** — `full` memang lebih mahal, tetapi unggul pada A1 (selisih > 0)
dan A3 (20 vs 7 klaster) → kriteria 2 **tidak** terpenuhi.

**Gerbang 3 — apakah `innovate` layak mengisi slot itu?**
- unggul mean \|IC\|/run: **0,0174 > 0,0109** ✓
- unggul cakupan pustaka: **23 > 13** ✓
- klaster sinyal: 8 < 20 ✗
- keandalan: **4/6 vs 6/6 → GAGAL** (syaratnya maksimal 1 run lebih buruk)

### PUTUSAN

> **BELUM DIGANTI — tetapi bukan karena idenya salah.**
>
> `design` gugur di gerbang 1: pengaruhnya terhadap IC tidak signifikan (t=0,79).
> `innovate` unggul pada dua sumbu terpenting (mutu sinyal dan cakupan pencarian)
> tetapi gagal syarat keandalan, dan penyebab kegagalannya sudah teridentifikasi
> sebagai masalah FORMAT yang bisa diperbaiki (§3.3).
>
> Yang terbukti bukan "ide inovasi salah", melainkan "implementasinya belum
> stabil". Langkah yang sah bukan melonggarkan aturan setelah melihat hasil,
> melainkan **memperbaiki penyebab kolaps, mendaftarkan ulang lengannya, dan
> menjalankan lagi**.

### Kenapa `design` tidak langsung dipangkas sekarang

Aturan mengizinkannya (B13), tetapi memangkas `design` lalu menggantinya lagi
beberapa hari kemudian berarti dua perubahan arsitektur berturut-turut pada objek
studi yang sama — dan Bab 4 harus menjelaskan keduanya. Lebih murah menunggu satu
ronde: `design` dipertahankan sampai pengganti yang stabil ada, lalu diganti
sekali.

Catatan yang wajib ikut dilaporkan: n=6 per lengan. Sumbu \|IC\| pada n ini tidak
punya daya uji; yang benar-benar menopang keputusan adalah sumbu bervarians rendah
— cakupan pustaka (23 vs 13), klaster sinyal (20 vs 7), laju lolos gate, dan biaya
per faktor diterima.

---

## 5. Ronde berikutnya — didaftarkan SEBELUM dijalankan

**Hipotesis**: kolaps `innovate` disebabkan register instruksi-meta di prompt
hulu yang diteruskan lewat KV, bukan oleh mandat inovasinya.

**Intervensi** (satu variabel per lengan):
1. `innovate_lean` — prompt `innovate` dipangkas: buang paragraf "WHAT YOU ARE
   FREE FROM / BOUND BY", sisakan menu sumbu + larangan idiom + swauji. Mandatnya
   tetap, retorikanya hilang.
2. `innovate_guided` — `innovate` + guided decoding HANYA pada emitter. Menguji
   apakah paksaan format menyelamatkan run yang kolaps, dan dengan biaya berapa.
3. `innovate_fid` diulang sebagai rujukan stabil (sudah 6/6).

**Kriteria lulus**: keandalan ≥ 5/6 **dan** cakupan pustaka ≥ 20 **dan** mean
\|IC\| per-ekspresi tidak berbeda dari lantai acak (p > 0,05). Bila terpenuhi,
`design` diganti `innovate_lean` secara permanen (B16) dan Bab 4 melaporkan
pergantian itu sebagai SATU perubahan arsitektur dengan alasan terukur.

**Bila tidak terpenuhi**: jalankan B13 (pangkas `design`), dan laporkan agen
inovasi sebagai arah lanjutan Bab 5 — bukan sebagai komponen sistem.
