# Rencana perbaikan QuantaLatent — dari temuan G1–G7 ke intervensi

> **STATUS PELAKSANAAN** (diperbarui 2026-08-07, branch `exp/rencana-perbaikan`)
>
> | tahap | status | catatan |
> |---|---|---|
> | Tahap 0 — bekukan baseline + sumbu A6/A7 | **SELESAI** | A7 uji-3 menemukan **nol** faktor rank-equivalent ke kolom mentah (≥0,99) — kekhawatiran AUDIT §2.5 tidak bereproduksi di korpus front-end |
> | Tahap 1 — B1 + B2 (+ B8, B9) | **SELESAI** | B8 diverifikasi: KL(perbaikan‖re-rotasi manual) = 0,0 persis. B9 baru dinyalakan SETELAH itu |
> | Tahap 2 — B11, B12, B15, B4 | **SELESAI** | B15 diuji 17 kasus + regresi 322 ekspresi; B12 kini penyebab tolakan terbanyak; B11 NEGATIF pada rantai `design` (HASIL_A8 §3.4) tetapi kemudian dipromosikan default TERIKAT rantai `innovate` (§4b) |
> | Tahap 3 + 3b — A8 (ablasi agen) + ronde pengganti | **SELESAI, LULUS** | hasil & keputusan: `lab/HASIL_A8.md` |
> | A4/A5 (geometri laten & efisiensi konteks) | **SELESAI** (diukur ulang) | A4: mekanisme identik persis dengan G1 lama (sesuai dugaan, B6/B7 belum disentuh). A5: redundansi construct 59,5%→2,5% |
> | Tahap 4 — B6 + B7 | **SELESAI** | `lab/HASIL_TAHAP4.md`. B6 nol-efek di produksi (menyala 9/9 di `raw`, 0/9 di `gumbel`) — nilainya sebagai pengaman salah-setel. B7: default kode `raw`→`gumbel` permanen; ridge M terbukti praktis ortogonal (cos 0,011) dan `use_realign` kini **inert** |
> | A9 kapasitas kanal laten | **SELESAI** (alat baru) | `lab/channel_capacity.py`. Kanal laten murni memulihkan 19–35% muatan pada ls=10, 76–84% pada ls=40; mode `kv` lossless karena **token prompt** yang ikut diwariskan, bukan karena vektor latennya |
> | Tahap 6 (B10) | **DITELITI + DIPROTOTIPEKAN**, tidak diadopsi | `lab/latent_bottleneck.py` + `HASIL_TAHAP4.md` §4 |
> | B5, B14, A10, A11 | **SELESAI** | `lab/KESIMPULAN.md` §4–§7. **Seluruh RENCANA_PERBAIKAN kecuali Tahap 5 kini beres** |
>
> **Keputusan A8 — final**: `design` gugur di gerbang 1 (pengaruhnya terhadap IC
> tidak signifikan, Welch t=0,79). Rekomendasi awal alat ("tunda satu ronde")
> **ditimpa keputusan eksplisit user** (2026-08-07) untuk mengganti `design` →
> `innovate` sekarang. Ronde lanjutan `innovate_guided` (guided decoding di
> emitter) kemudian LULUS ketiga kriteria yang sudah didaftarkan di muka
> (keandalan 6/6, pustaka 22, IC tak beda dari lantai acak p=0,633) — sehingga
> keputusan user dan aturan yang terdaftar bertemu di titik yang sama.
>
> **Konfigurasi produksi sekarang**: rantai `proposal → innovate → construct`
> (`frontend_chain` di settings.py/YAML) + guided decoding aktif di construct
> (`guided_decoding: true`, TERIKAT pada rantai ini — lihat peringatan di
> settings.py bila rantai dikembalikan ke `design`). vs rantai lama: \|IC\|/run
> +67% (0,0182 vs 0,0109), lolos gate 88% vs 83%, 22 vs 13 fungsi DSL disentuh.
> Rincian lengkap: `lab/HASIL_A8.md`.
>
> ---
>
> ### Status per-item — SEMUA sumbu A & katalog B
>
> "Tahap X selesai" tidak berarti semua item bernomor di dalamnya dikerjakan
> (mis. B3/B5/B14 bukan bagian Tahap 2/3 meski nomornya berdekatan). Tabel ini
> memberi status eksplisit untuk SETIAP item, supaya tak ada yang tersamar
> sebagai "selesai" hanya karena tahap induknya selesai.
>
> **Sumbu benchmark (A)**
>
> | item | status | hasil / catatan |
> |---|---|---|
> | A1 mutu sinyal | ada, dipakai terus | alat lama; dipakai di semua run sesi ini |
> | A2 keandalan produksi | ada, dipakai terus | alat lama |
> | A3 cakupan pencarian (klaster) | ada, dipakai terus | dihitung untuk semua lengan A8 |
> | A4 geometri & entropi jalur laten | **SELESAI**, diukur ulang | identik persis dgn G1 lama di 8 varian. **Tetap berlaku setelah B6/B7**: B6 nol-efek pada `gumbel` (0/9 menyala) dan B7 hanya mengubah DEFAULT kode — produksi sudah `gumbel` sejak B2 |
> | A5 efisiensi konteks | **SELESAI**, diukur ulang | redundansi construct 59,5%→2,5%; KV total −22% |
> | A6 biaya per faktor diterima | **SELESAI** (baru sesi ini) | dipakai di seluruh A8 |
> | A7 kesetiaan rantai | **SELESAI** (baru sesi ini) | var_recall 0,79–0,97; 0 faktor rank-equivalent ke kolom mentah |
> | A8 ablasi agen | **SELESAI** (baru sesi ini) | `lab/HASIL_A8.md` — dasar keputusan B16 |
> | A9 kapasitas kanal laten | **SELESAI** (alat baru sesi ini) | `lab/channel_capacity.py`; hasil di `lab/HASIL_TAHAP4.md` §3 |
> | A10 sensitivitas arah | **SELESAI** (alat baru sesi ini) | `lab/direction_sensitivity.py` + pasangan arah berlawanan `opp_mom`/`opp_rev`; hasil di `lab/KESIMPULAN.md` §5 |
> | A11 stabilitas jangka panjang | **SELESAI** (alat baru sesi ini) | `lab/stability_probe.py`; hasil di `lab/KESIMPULAN.md` §7 |
>
> **Katalog intervensi (B)**
>
> | item | status | hasil / catatan |
> |---|---|---|
> | B1 latent.steps 60→10 | **SELESAI** | settings.py + 5 file config |
> | B2 step_mode=gumbel | **SELESAI** | dipromosikan ke kunci YAML; A4 mengonfirmasi mekanismenya (47/60 token unik vs 2/60 pada raw) |
> | B3 kontrak DSL (prompts_v1.yaml) | ada SEBELUM sesi ini | bukan hasil sesi ini; produksi tetap pakai prompts.yaml, bukan v1 |
> | B4 prompt ringkas (lib_in_kv) | **SELESAI** | construct −37% token; dikonfirmasi ulang lewat A5 |
> | B5 ringkas format keluaran proposal | **SELESAI** — dgn koreksi | format 3-bagian ternyata SUDAH ada sejak `fe42127`; yang dikerjakan: perbaiki rantai usang pasca-B16 di prompt proposal + pangkas (866→788 tok). KESIMPULAN §4 |
> | B6 early-stop rollout laten | **SELESAI**, terverifikasi | nol-efek di produksi (`gumbel` 0/9); menyala 9/9 & hemat 47% di `raw`. Nilainya sebagai pengaman salah-setel — HASIL_TAHAP4 §1 |
> | B7 ganti persamaan realignment permanen | **SELESAI** | default kode `raw`→`gumbel`. Temuan ikutan: `use_realign` INERT di produksi → ablasi G6 hanya berlaku utk `raw` — HASIL_TAHAP4 §2 |
> | B8 perbaiki kv_truncate (RoPE) | **SELESAI**, terverifikasi | KL(perbaikan‖manual)=0,0 persis |
> | B9 hidupkan anggaran KV | **SELESAI** | dinyalakan setelah B8 terverifikasi |
> | B10 latent bottleneck | **DITELITI + DIPROTOTIPEKAN**, **tidak diadopsi** | prototipe training-free `lab/latent_bottleneck.py`; keputusan & alasan di HASIL_TAHAP4 §4 |
> | B11 guided decoding construct | **SELESAI**, dipromosikan bersyarat | negatif di rantai `design`; positif & default TERIKAT rantai `innovate` |
> | B12 execution gate | **SELESAI** | `execution_gate.py` baru |
> | B13 pangkas rantai bila design tak kontribusi | **SUPERSEDED oleh B16** | design DIGANTI, bukan dipangkas |
> | B14 medium "konteks segar + ringkasan" | **DIIMPLEMENTASI + DIUKUR**, bukan default | `comm_mode="summary"` di pipeline.py + 8 unit test; hasil & keputusan di KESIMPULAN §6 |
> | B15 tiga lubang gate (a/b/c) | **SELESAI** | 17 unit test + regresi 322 ekspresi |
> | B16 ganti design→innovate | **SELESAI, LULUS** | keputusan user + kriteria Tahap 3b terpenuhi |
>
> **Tahap (rencana bertahap §C)**
>
> | tahap | status |
> |---|---|
> | Tahap 0 | SELESAI |
> | Tahap 1 | SELESAI |
> | Tahap 2 | SELESAI |
> | Tahap 3 + 3b | SELESAI, LULUS |
> | Tahap 4 (B6, B7) | **SELESAI** — `lab/HASIL_TAHAP4.md` |
> | Tahap 5 (mutation/crossover/feedback) | **BELUM DISENTUH SAMA SEKALI** — bukan cuma sesi ini, sejak awal proyek |
> | Tahap 6 (B10) | **diteliti lebih awal** (A9 memberi premisnya); tidak diadopsi — HASIL_TAHAP4 §4 |

**Dibuat**: 2026-08-07, setelah §8 AUDIT_KRITIS dijalankan di GPU (Qwen3-8B).
Angka pendukung ada di `lab/HASIL_GPU.md`; dokumen ini hanya soal **apa yang
harus diukur berikutnya** dan **apa yang harus diubah, dengan urutan dan alasan**.

Ditulis sebagai RENCANA, bukan sebagai perubahan yang sudah diterapkan. Alasannya
eksplisit: dari temuan yang ada, ada tiga lapisan yang bisa diubah (matematika
laten, prompt, arsitektur agen) dan sebagian intervensinya saling meniadakan
kalau dijalankan bersamaan. Menjalankan semuanya sekaligus akan menghasilkan
sistem yang mungkin lebih baik tetapi **tanpa satu pun klaim kausal yang bisa
dipertahankan di sidang** — padahal justru itu aset skripsi ini.

---

## 0. Batas berlaku temuan sekarang (dan kenapa itu bukan kelemahan)

Pengujian G1–G7 hanya menyentuh **proposal → design → construct**. Agen
`mutation`, `crossover`, `feedback`, `repair`, dan seluruh siklus evolusi
BELUM diuji.

Itu keputusan yang disengaja dan sebaiknya dipertahankan, karena front-end adalah
**satu-satunya jalur yang menghasilkan ekspresi**. Mutation dan crossover tidak
menulis ekspresi — mereka hanya menetapkan arah lalu **memanggil ulang front-end
yang sama** (`FrontEndPipeline.run_evolution` → `self.run(seed_kv=guidance_kv)`).
Jadi:

> Apa pun batas atas mutu yang berlaku untuk front-end **juga berlaku untuk
> seluruh sistem evolusioner**, karena evolusi tidak punya jalur lain untuk
> menghasilkan faktor.

Konsekuensinya, hasil G1–G7 adalah **batas bawah yang sah** untuk mengukur
perbaikan agen lain: kalau front-end pada `latent.steps=60` menghasilkan 0
ekspresi dalam 6 run, maka mutation/crossover di atasnya **tidak mungkin**
menghasilkan apa pun — persis yang terlihat di B13 (trajectory kosong).

Yang HARUS diuji terpisah nanti (dan tidak bisa disimpulkan dari front-end):
apakah arah dari mutation/crossover benar-benar mengubah apa yang dibangun
front-end (§B7 di bawah), dan apakah feedback memengaruhi ronde berikutnya.

---

## A. Sumbu benchmark yang diusulkan

Sumbu 1–4 sudah punya alat dan sebagian angkanya sudah ada. Sumbu 5–11 adalah
tambahan yang diusulkan; masing-masing menyertakan cara ukurnya supaya tidak
berhenti jadi wacana.

### A1. Mutu sinyal (sudah ada)
`|IC|` per-faktor hidup, `|IC|` per-run, ICIR, t-stat, **selalu** dibandingkan
dengan lantai acak `lab/random_baseline.py`. Unit analisis = run, bukan ekspresi.
*Alat*: `lab/analyze_gpu.py`.

### A2. Keandalan produksi (sudah ada)
Fraksi run yang menghasilkan ≥1 ekspresi; laju output tak-terparse; laju
faktor mati (≤2 nilai unik/hari); laju tolak gate per sebab.
*Alat*: `lab/analyze_gpu.py`, `lab/gate_report.py`.
*Kenapa penting*: G2 menunjukkan sumbu ini bisa runtuh total (0/6) sementara
sumbu A1 diam-diam terlihat "baik" karena dihitung dari sisa yang sedikit.

### A3. Cakupan pencarian (sudah ada)
Jumlah klaster sinyal (|Spearman deret IC| > 0,7) per lengan.
*Kenapa penting*: inilah B14 yang benar — `kv` lama menemukan 2 klaster dari 39
faktor. Kekuatan sinyal bisa sama sementara cakupannya beda 4×.

### A4. Geometri & entropi jalur laten (SELESAI — diukur ulang 2026-08-07)
cos ke embedding terdekat, langkah titik tetap, identik-antar-seed,
identik-antar-arah. *Alat*: `lab/latent_dynamics.py`.

**HASIL**: sweep penuh 8 varian (Qwen3-8B, 60 langkah, 3 arah × 3 seed)
diulang setelah seluruh Tahap 0–3b. **Identik persis** dengan baseline G1 lama
di semua metrik dan semua varian — tabel penuh:

| varian | H_akhir | cos ke embedding terdekat | langkah titik tetap | token unik/60 |
|---|---:|---:|---:|---:|
| raw (T=0) | 0,10 | 0,074 | 34 | 2 |
| raw_realign (T=0) | 5,99 | 0,275 | tak pernah | 6 |
| raw_noise (T=0,1) | 0,11 | 0,074 | 34 | 2 |
| soft (T=1,0) | 1,10 | 0,898 | tak pernah | 20 |
| soft (T=2,0) | 0,03 | 0,811 | 18 | 8 |
| **gumbel (T=0,7)** *(default produksi, B2)* | 0,00 | 0,940 | tak pernah | **47** |
| gumbel (T=1,0) | 0,00 | 0,895 | tak pernah | 38 |
| sample (T=1,0) | 0,00 | 1,000 | tak pernah | 42 |

Kesamaan sampai 2–3 desimal ini **BUKAN kegagalan pengukuran** — ia bukti
langsung bahwa mekanisme rollout laten itu sendiri (matriks realignment,
embedding, titik tetap) tidak tersentuh sepanjang Tahap 0–3b, persis sesuai
prediksi: hanya B6 (early-stop) dan B7 (ganti persamaan realignment) yang akan
mengubah angka-angka ini, dan keduanya **belum dikerjakan**. Baris `gumbel@0,7`
dicetak tebal karena itulah mode yang sekarang berjalan di produksi (B2):
47/60 token unik dan cos-ke-embedding 0,940 (selalu in-distribution) — kontras
dengan `raw` (mode lama) yang cuma 2/60 token unik dan mencapai titik tetap di
langkah 34. Ini menjelaskan MEKANISME di balik hasil G3 (gumbel menaikkan
klaster sinyal 6→9, lolos gate 54%→91%): jalur laten `raw` runtuh jadi
salinan berulang, sedangkan `gumbel` tetap bergerak dan tetap di dalam convex
hull embedding sepanjang rollout.
File: `lab/out/latent_dynamics_Qwen_Qwen3-8B_postB16.json`.

### A5. **Efisiensi konteks** (SELESAI — diukur ulang 2026-08-07 setelah B4/B16)
Tiga metrik, dihitung `lab/latent_growth.py`:
- **redundansi**: fraksi n-gram prompt hop-k yang sudah ada verbatim di KV;
- **panjang konteks efektif**: massa attention per segmen pada langkah emisi;
- **massa attention per segmen** dengan *enrichment* = massa ÷ porsi panjang.
*Kenapa penting*: ini menerjemahkan "penumpukan KV" dari keluhan menjadi angka,
dan langsung memberi target optimasi (turunkan redundansi, naikkan enrichment
segmen instruksi aktif).

**HASIL** (Qwen3-8B, `comm_mode=kv`, `latent_steps=10`, sebelum vs sesudah
B4+B16; skrip disesuaikan untuk mengukur rantai produksi SAAT INI —
`proposal→innovate→construct` dengan `lib_in_kv=True`, bukan rantai `design`
lama, kalau tidak angka yang keluar adalah sistem yang sudah tak dipakai):

| metrik | LAMA (`design`, pra-B4) | BARU (`innovate`, pasca-B4/B16) |
|---|---:|---:|
| KV total di construct | 5 955 token | **4 624 token** (−22%) |
| token prompt construct | 2 486 | **1 591** (−36%, cocok klaim B4 §Tahap 2) |
| **redundansi verbatim (8-gram) construct** | **59,5%** | **2,5%** |
| cos blok laten proposal↔hulu | 0,563 | 0,084 |
| cos blok laten proposal↔construct | 0,624 | 0,108 |
| cos blok laten hulu↔construct | 0,551 | 0,078 |
| massa attention construct:latent | 0,0146 (8,72×) | 0,0184 (8,50×) |

**Baca hati-hati**: B4 **bukan** satu-satunya penyebab turunnya redundansi dari
59,5%→2,5% — perbandingan ini menggabungkan TIGA perubahan sekaligus (B4 prompt
ringkas, B2 gumbel menggantikan raw, B16 innovate menggantikan design), karena
tujuannya di sini adalah memotret KEADAAN SISTEM SAAT INI, bukan mengisolasi
kontribusi B4 sendirian (itu sudah diverifikasi terpisah di Tahap 2: 2579→1624
token pada prompt construct, murni dari `lib_in_kv`). Turunnya cos antar-hop
(0,55–0,62 → 0,08–0,11) konsisten dengan G3 (gumbel menaikkan entropi/keragaman
jalur laten dibanding raw) — bukan berarti "pikiran" antar-agen makin tak
berkaitan, melainkan makin tak sekadar salinan satu sama lain.
**Massa attention pada blok laten construct tetap tinggi** (8,5×) di kedua
kondisi — kanal laten construct konsisten dominan, tak terpengaruh perubahan
rantai/prompt di hulunya. File: `lab/out/latent_growth_Qwen3-8B_kv_ls10_postB16.json`.

### A6. **Biaya per faktor diterima** (BARU)
`token diproses ÷ faktor lolos gate` dan `detik ÷ faktor lolos gate`.
*Kenapa penting*: satu-satunya metrik yang menghukum arsitektur boros secara
adil. Contoh dari G4: `text` = 67 s/run untuk 30 faktor lolos gate;
`kv_and_text@ls10` = 99 s/run untuk 11. Perbandingan "per run" menyembunyikan ini.

### A7. **Kesetiaan rantai** (BARU) — apakah agen benar-benar berkomunikasi?
Tiga uji deterministik, tanpa LLM tambahan:
- **fidelity hipotesis→ekspresi**: variabel yang disebut hipotesis vs variabel
  yang muncul di ekspresi; horizon yang disebut vs window yang dipakai.
  (AUDIT §2.1 menemukan hipotesis "small-cap … high volume" yang berakhir jadi
  `-RANK($volume)` — sumbu ini menangkapnya otomatis.)
- **kepatuhan palette**: % fungsi di ekspresi yang berasal dari palette design.
  Kalau rendah, agen `design` adalah hiasan.
- **rank-equivalence ke kolom mentah**: |Spearman| faktor terhadap
  `$volume`,`$close`,`$high−$low`,… Kalau > 0,99, faktor itu kolom mentah yang
  disamarkan. Ini menjadikan temuan tertajam AUDIT §2.5 sebagai gate rutin,
  bukan penemuan sekali jalan.

### A8. **Ablasi agen** (BARU) — benchmark arsitektur yang paling murah
Jalankan front-end dengan rantai dipotong dan bandingkan A1–A3:
`proposal→design→construct` (sekarang) · `proposal→construct` ·
`direction→construct` · `construct` sendirian dengan direction sebagai teks.
*Kenapa penting*: kalau memotong `design` tidak menurunkan apa pun, maka
sepertiga biaya konteks (2 273 token prompt design) adalah beban murni. Ini
pertanyaan arsitektur yang belum pernah diajukan, dan biayanya ~15 menit GPU.

### A9. **Kapasitas kanal laten** (BARU) — berapa bit yang benar-benar lewat?
Titipkan payload yang DIKETAHUI ke agen hulu (mis. 5 nama fungsi acak dari
library), lalu minta agen hilir merekonstruksinya, sekali lewat KV saja dan
sekali lewat teks. Skor = akurasi rekonstruksi.
*Kenapa penting*: ini mengubah "apakah laten lossy untuk muatan simbolik" dari
tafsiran hasil mining menjadi **pengukuran langsung**, dan hasilnya tidak
tercemar mutu faktor. Ini juga uji yang paling bisa dipertahankan di sidang
untuk klaim inti skripsi.

### A10. **Sensitivitas terhadap arah** (BARU)
Beri dua arah riset yang berlawanan, ukur jarak antara himpunan ekspresi yang
dihasilkan (Jaccard fungsi + korelasi deret IC). Kalau outputnya sama, sistem
mengabaikan masukannya. G1 sudah menunjukkan gejalanya di level vektor laten
(identik-antar-arah 3/3 pada 8B `raw`); sumbu ini mengukurnya di level keluaran.

### A11. **Stabilitas jangka panjang** (BARU, untuk klaim "berkelanjutan")
VRAM puncak per trajectory, pertumbuhan KV per hop, dan laju kebocoran memori
lintas task. Sudah ada catatannya (B1, T6) tapi belum pernah jadi metrik rutin.

---

## B. Katalog intervensi

Tiap butir: **[bukti yang mendasarinya] → perubahan → risiko → cara verifikasi**.
Kolom "biaya" = perkiraan jam GPU untuk memverifikasi, bukan waktu koding.

### Lapisan 1 — konfigurasi (sudah terbukti, risiko ~nol)

**B1. `latent.steps: 60 → 5` (atau 10).** Biaya: 0 (sudah diukur).
[G2: 0/6 vs 6/6 run berhasil; G1: titik tetap di langkah 12 pada 4B]
Risiko: tak ada yang teridentifikasi — mutu |IC| tidak turun di ls rendah.
Verifikasi: sudah, 30 run.

**B2. `LATENT_STEP_MODE=gumbel`, T=0,7.** Biaya: 0 (sudah diukur).
[G3: lolos gate 54% → 91%, faktor hidup 13 → 27, klaster 6 → 9]
Risiko: kekuatan sinyal TIDAK membaik (t = −0,27); jangan diklaim sebagai
perbaikan mutu. Verifikasi: sudah, 18 run.

### Lapisan 2 — prompt

**B3. Perbaiki kontrak DSL di prompt** (sudah dibuat: `prompts_v1.yaml`).
[AUDIT §2.2/§S6 + G5: 46% tolakan gate adalah "kondisi non-boolean"]
Perubahan: RANK/TS_RANK dinyatakan persentil [0,1]; syarat ternary wajib
perbandingan; window ≥ 2 (≥ 5 untuk statistik sebaran); larangan ambang absolut
`$volume`; satu simbol satu makna; tanpa markdown.
Risiko: rendah. Verifikasi: 2×2 model × prompt (§HASIL_GPU §Q).

**B4. Prompt RINGKAS untuk agen hilir saat comm_mode = kv/kv_and_text.**
Biaya ~1 jam GPU. [G7: 59,5% prompt construct adalah duplikat verbatim; konteks
7,9k token yang 19%-nya pengulangan]
Perubahan: design & construct tidak mengirim ulang seluruh pustaka fungsi ketika
`handoff='kv'` — cukup aturan sintaks + daftar nama fungsi. Polanya sudah ada di
repo ini untuk jalur rdagent (`QlibAlphaAgentScenario.get_compact_desc(step)`,
lihat `LATENT_INTEGRATION_LOG.md` §1) tetapi agen `latent_mas` tak memakainya.
Risiko: **bisa memperburuk** kalau model ternyata perlu membaca ulang pustaka.
Karena itu wajib jadi lengan terkontrol, bukan diterapkan langsung.
Verifikasi: A5 (redundansi harus turun dari 59,5%), A2, A1.

**B5. Ringkas format keluaran proposal.** [catatan.txt user]
Gabungkan `KNOWLEDGE/OBSERVATION/JUSTIFICATION/SPECIFICATION/SUMMARY` menjadi
`A/B/Final Hypothesis`. Efek langsung: KV lebih pendek untuk hop berikutnya.
Risiko: rendah. Verifikasi: A5 + A2.

### Lapisan 3 — matematika laten

**B6. Early-stop adaptif pada rollout laten.** Biaya ~1 jam. [G1: titik tetap
langkah 12–34; G7: pengaruh kanal laten TURUN 4× saat ls 10→60]
Perubahan (± 5 baris di `latent_pass`): berhenti saat
`cos(h_k, h_{k−1}) > 0,999`. `latent_steps` berubah makna dari target menjadi
batas atas. Ini menghapus salinan tanpa menghapus "pikiran", dan membuat sistem
tahan terhadap salah-setel seperti B7 di MONITORING_NOTES.
Risiko: rendah; perilaku pada ls kecil praktis tak berubah.

**B7. Ganti persamaan realignment (permanen, bukan hanya env).**
[G6: M = identitas di 4B tapi memutar ~90° di 8B; G1: `raw` di 8B kehilangan
informasi arah 3/3] Peta yang secara prinsip benar adalah
`e = W_inᵀ softmax(W_out h / T)` — proyeksi ke convex hull embedding nyata —
sedangkan ridge `M` adalah ekstrapolasi linear di daerah yang tak pernah difit
(cos hidden-state ke embedding terdekat hanya 0,07–0,16).
Risiko: mengubah klaim Bab 4 tentang "mekanisme inti"; harus dilaporkan sebagai
**temuan**, bukan disembunyikan sebagai perbaikan.

**B8. Perbaiki `kv_truncate` sebelum menyambungkan anggaran KV.**
[§9.2 HASIL_GPU] Panggil `_rerotate_keys_contiguous` dengan
`orig_positions = arange(N−k, N)`. **Prasyarat** untuk B9 — menyambungkan
anggaran tanpa ini menukar satu bug dengan bug lain.

**B9. Hidupkan anggaran KV yang selama ini mati.**
[§9.1 HASIL_GPU: `kv_max_tokens: 20480` tidak pernah diberlakukan; `knn_enabled`
otomatis mati saat `latent_steps>0` → tidak ada kendali ukuran KV sama sekali]
Perubahan: teruskan `kv_max_seq_len` di `pipeline/settings.py::build_backend`.
Risiko: mengaktifkan B8 kalau B8 belum dipasang.

**B10. *Latent bottleneck* (paling ambisius di lapisan ini).**
Alih-alih mewariskan SELURUH KV agen hulu, ringkas jadi `m ≪ L` vektor
(attention pooling atas blok laten + jawaban), lalu itu saja yang diwariskan.
Ini menyerang akar pertumbuhan: konteks emitter berhenti tumbuh linear terhadap
jumlah hop. Ini juga lebih dekat ke klaim "working memory" di paper daripada
implementasi sekarang.
Risiko: TINGGI, ini riset. Jangan dikerjakan sebelum B1–B6 selesai dan A8
menunjukkan rantai agennya memang layak dipertahankan.

### Lapisan 4 — arsitektur multi-agen

**B11. Guided decoding untuk `construct`.** Biaya ~1 jam. **Rasio manfaat/risiko
terbaik di lapisan ini.**
[G5: 38% tolakan gate adalah output tak-terparse; seluruh kolaps ls≥20
bermanifestasi sebagai output tak-terparse]
Infrastrukturnya **sudah ada dan sudah tersambung** —
`llm/guided_decoding.py` + `LocalLLMBackend.run(json_schema=...)` — tetapi
`AgentSpec` tidak punya field `json_schema` sehingga jalur itu tak pernah
terpakai oleh agen mana pun. Menambahkannya = satu field di `AgentSpec`,
satu baris di `LatentAgent.run`, satu kunci di `prompts.yaml`.
Risiko: overhead latensi 10–20%; grammar JSON tidak menjamin ISI yang bermakna.
Verifikasi: laju tak-terparse harus → 0; lihat apakah kolaps ls tinggi berubah
dari "0 ekspresi" jadi "ekspresi buruk" (itu pun kemajuan diagnostik).

**B12. Eksekusi percobaan sebelum gate (execution gate).** Biaya ~0 GPU.
[AUDIT §S1: sembilan gate semuanya struktural; G5: 1 dari 34 ekspresi lolos gate
ternyata konstan] Jalankan ekspresi pada sampel kecil (± 1 detik CPU), tolak
kolom NaN-total/konstan, dan kirim pesannya ke agen repair.
Risiko: sangat rendah. Ini gate termurah yang masih hilang.

**B15. Tambal tiga lubang gate yang ditemukan G5.** Biaya ~0 GPU, semuanya
deterministik dan bisa diuji dengan unit test:
- **keluaran boolean 2-nilai** (`(A > B) ? (-1) : (1)`, 7 dari 198 yang lolos):
  tolak ekspresi yang seluruh cabang ternary-nya konstan — promptnya sudah
  melarang, gate-nya belum memeriksa;
- **argumen kuantil di luar [0,1]** (`TS_QUANTILE($volume, 20, 5)` → crash):
  `validate_semantics` memeriksa ambang *perbandingan* terhadap `_PCT_FUNCS`
  tetapi tidak memeriksa argumen `q` milik `TS_QUANTILE`/`PERCENTILE`. Sekalian
  seragamkan urutan argumennya di prompt — sekarang `TS_QUANTILE(A, p, q)` vs
  `PERCENTILE(A, q, p)`, dan itu jebakan yang kita pasang sendiri;
- **arity satu arah**: `_build_arity_map` menurunkan `min_args` dari signature
  Python, sehingga `TS_MEAN($close)` lolos dan Python diam-diam mengisi
  window = 5 — ekspresi yang dievaluasi bukan yang ditulis model. Ambil `min_args`
  dari kontrak DSL di prompt, bukan dari default Python. (Prevalensi cuma 1%,
  jadi prioritas rendah, tetapi perbaikannya sepele.)

**B13. Pangkas rantai agen sesuai hasil A8.** Biaya: A8 itu sendiri (~15 menit).
Kalau `design` tak berkontribusi, hapus atau gabungkan ke `construct`; itu
memotong 2 273 token prompt dan satu hop pertumbuhan KV sekaligus.

**B16. GANTI `design` dengan agen inovasi (`innovate`) — bukan sekadar hapus.**
[permintaan user 2026-08-07 + tiga angka dari korpus 446 ekspresi]
Ini melengkapi B13: kalau A8 menunjukkan `design` tak berkontribusi, slot itu
tidak harus kosong — ia bisa diisi agen dengan mandat yang berlawanan.

*Bukti yang mendasarinya.* Keluhan "ekspresi monoton dan standar" ternyata
terukur, dan angkanya lebih tajam dari dugaan:
- **28 dari 55 fungsi DSL TIDAK PERNAH dipakai sekali pun** (BB_*, DECAYLINEAR,
  EMA, WMA, SMA, SUMAC, PROD, COUNT, SUMIF, FILTER, DELTA, SIGN, LOG, SQRT,
  POW, INV, FLOOR, SKEW, KURT, MEDIAN, STD, MACD, SEQUENCE, …);
- empat fungsi (TS_ZSCORE, TS_PCTCHANGE, TS_STD, RANK) mendominasi keluaran;
- **59% ekspresi memakai pembungkus terluar yang sama** (RANK/ZSCORE/TS_ZSCORE),
  dan 78% hanya berisi 2–3 pemanggilan fungsi;
- sementara itu ekspresi **ACAK** dari DSL yang sama mencapai mean |IC| 0,0170 —
  **di atas setiap lengan LLM** (0,0084–0,0189).

Baris terakhir itu yang menentukan arah. Kalau pencarian acak yang tak punya
teori sama sekali mengungguli rantai agen yang seluruh promptnya tentang
mekanisme ekonomi, maka yang langka bukan pembenaran teoretis — melainkan
**cakupan struktural**. `design` menyempitkan (memilih palette yang dibenarkan
teori); `innovate` melebarkan.

*Kenapa bukan sekadar "suruh model lebih kreatif".* Model kecil yang disuruh
berkreasi akan kembali ke idiom yang sama; itu justru yang terjadi sekarang
(prompt construct SUDAH memuat "FOUR WAYS TO VARY"). Karena itu `innovate`
tidak memakai kata sifat, melainkan **mesin**: menu 11 sumbu struktural yang
konkret (kedalaman, pasangan tak lazim, kontras skala waktu, normalisasi ganjil,
statistik urutan, bentuk-bukan-level, residual, asimetri, silang-famili,
inversi, operator terlantar), **daftar idiom jenuh yang dibatasi maksimal satu**,
dan swauji cakupan sebelum menutup jawaban.

*Perubahan menyertainya yang tak bisa dihindari.* Prompt `construct` sekarang
membuka dengan "FIDELITY FIRST … Variety lives inside the hypothesis, never
outside it." Menaruh agen yang tugasnya MEMBELOKKAN hipotesis di hulu emitter
yang diperintahkan SETIA pada hipotesis adalah dua perintah yang saling
meniadakan. Karena itu lengan `innovate` menyalakan `free_form`: klausa
kesetiaan diganti klausa cakupan. Supaya keduanya tidak berubah bersamaan tanpa
kendali, A8 menjalankan **dua** lengan innovate — dengan dan tanpa klem
kesetiaan.

Risiko: ekspresi "bebas" bisa jadi omong kosong yang mahal. Mitigasinya bukan
teori melainkan gate: B15 + B12 menolak yang tak rankable, mati, atau tak legal,
dan A6 menghukum lengan yang boros per faktor diterima. Kebebasan ada pada
BENTUK rumus, bukan pada aturan DSL.

Risiko kedua, khusus pengukuran: operator lambat (`REGBETA`/`REGRESI`, rolling
quantile) bisa memakan belasan menit per ekspresi, sehingga lengan innovate
akan tampak buruk karena **timeout**, bukan karena mutunya. Karena itu skoring
CPU diberi anggaran waktu per-ekspresi, dan REGBETA/REGRESI dikeluarkan dari
daftar operator yang dianjurkan (tetap legal, hanya tidak dipromosikan) —
sejalan dengan lantai acak yang juga mengecualikannya.

**B14. Ganti medium menjadi "konteks segar + ringkasan terstruktur".**
[G4: `text` = 6/6 run berhasil, lolos gate 83%, tanpa akumulasi KV sama sekali]
Ini pada dasarnya mengakui bahwa `text` menang di sumbu keandalan, lalu
memperbaikinya di sumbu mutu (yang justru lebih lemah: |IC|/run 0,0084 vs
0,0152). Bentuk konkretnya: tiap agen mulai dari konteks bersih, menerima
ringkasan JSON dari agen sebelumnya, bukan KV.
Risiko: ini mengubah objek studi skripsi. Boleh diusulkan sebagai *arah lanjutan*
di Bab 5, bukan sebagai perubahan sebelum Bab 4 ditulis.

---

## C. Rencana bertahap dengan gerbang keputusan

Prinsipnya: **satu variabel per tahap**, dan setiap tahap punya kriteria lulus
yang ditulis SEBELUM dijalankan.

### Tahap 0 — bekukan baseline (½ jam GPU)
Simpan hasil G1–G7 sekarang sebagai baseline resmi. Tambahkan A6 dan A7 ke
`analyze_gpu.py` supaya semua lengan berikutnya terukur pada sumbu yang sama.
*Lulus bila*: tabel baseline lengkap untuk ketiga comm_mode.

### Tahap 1 — kembalikan sistem ke keadaan "hidup" (0 jam GPU tambahan)
Terapkan **B1** (ls 5–10) dan **B2** (gumbel). Keduanya sudah diverifikasi.
*Lulus bila*: ≥5/6 run menghasilkan ekspresi di ketiga mode. (Sudah tercapai
untuk kv; kv_and_text perlu dicek ulang pada ls=5.)

### Tahap 2 — murahkan konteks (≈2 jam GPU)
**B11** (guided decoding) → **B12** (execution gate) + **B15** (tambal lubang
gate) → **B4** (prompt ringkas). Urutannya penting: B11/B12/B15 menghapus
kelas-kelas kegagalan sehingga efek B4 bisa diukur bersih.
*Lulus bila*: laju tak-terparse → 0, redundansi konteks turun dari 59,5% ke
< 20%, dan A2 tidak memburuk.
*Gerbang*: kalau B4 justru menurunkan A2, batalkan B4 dan catat — itu bukti
bahwa model memang perlu membaca ulang pustaka, dan itu temuan yang layak
dilaporkan.

### Tahap 3 — uji arsitekturnya, bukan cuma parameternya (≈1 jam GPU)
**A8** (ablasi agen) + **A9** (kapasitas kanal) + **A10** (sensitivitas arah).
*Gerbang*: kalau A8 menunjukkan rantai 3-agen tidak mengungguli
`direction→construct`, maka seluruh pertanyaan "medium komunikasi" berpindah
konteks — yang perlu dijelaskan bukan lagi KV vs teks, melainkan **kenapa
kolaborasi multi-agen tidak menambah nilai di skala model ini**. Itu kesimpulan
yang sah dan kuat, dan A9 memberi mekanismenya.

#### Tahap 3a — nasib agen `design`: kriteria keputusan, ditulis SEBELUM dijalankan

A8 dijalankan sebagai lima lengan, 6 run per lengan (2 arah × 3 seed), pada
konfigurasi Tahap 1 (`kv`, ls=10, gumbel):

| lengan | rantai | klem kesetiaan | yang diisolasi |
|---|---|---|---|
| `full` | proposal→design→construct | ON | rantai produksi (rujukan) |
| `nodesign` | proposal→construct | ON | kontribusi `design` |
| `direct` | construct sendirian | ON | nilai seluruh hulu |
| `innovate` | proposal→innovate→construct | **OFF** | usulan pengganti, utuh |
| `innovate_fid` | proposal→innovate→construct | ON | memisahkan efek agen dari efek klem |

**Aturan keputusan** (unit analisis = run; `design` dinyatakan tak berkontribusi
bila SALAH SATU terpenuhi):
1. mean |IC| per-run `full` **tidak** melebihi `nodesign` secara terarah
   (selisih ≤ 0 atau Welch |t| < 1 dengan n=6 — dengan n sekecil ini kita hanya
   bisa menolak klaim "jelas lebih baik", bukan membuktikan setara); **atau**
2. `full` lebih mahal pada A6 (detik & token per faktor diterima) tanpa unggul
   pada A1 maupun A3 (klaster sinyal).

**Bila `design` dinyatakan tak berkontribusi**, slot itu diisi `innovate` —
dengan syarat lengan `innovate` melampaui `full` pada minimal satu dari:
mean |IC| per-run, jumlah klaster sinyal (A3), atau cakupan pustaka (jumlah
fungsi DSL berbeda yang terpakai), **dan** tidak lebih buruk pada A2
(fraksi run yang menghasilkan ≥1 ekspresi) lebih dari 1 run dari 6.

Kalau `innovate` juga tidak melampaui apa pun, keputusannya adalah **B13**
(pangkas jadi `proposal→construct`), bukan mempertahankan `design` — karena
gerbang 1/2 sudah menyatakan slot itu tidak membayar biayanya.

*Catatan kejujuran yang harus ikut dilaporkan*: n=6 per lengan terlalu kecil
untuk uji beda yang meyakinkan pada |IC|. Sumbu yang benar-benar bisa diputuskan
pada n ini adalah yang variansnya rendah dan efeknya besar — cakupan pustaka,
klaster sinyal, laju lolos gate, dan biaya A6. Keputusan mengganti `design`
karena itu digantung pada sumbu-sumbu tersebut, dan |IC| dilaporkan sebagai
sumbu yang **tidak** membedakan, bila memang begitu hasilnya.

### Tahap 3b — RONDE PENGGANTI `design` — SELESAI, LULUS

*Hipotesis yang diuji*: kolaps `innovate` (2 dari 6 run; emitter menjawab "I
understand the instruction." lalu berhenti) disebabkan register instruksi-meta
di prompt hulu yang diteruskan lewat KV, bukan oleh mandat inovasinya sendiri.

*Lengan yang dijalankan*: `innovate_guided` (`innovate` apa adanya + guided
decoding HANYA di emitter). `innovate_lean` (versi prompt dipangkas) tidak jadi
dijalankan karena lengan guided sudah lulus ketiga kriteria sekaligus.

*Hasil vs kriteria yang didaftarkan di muka*:

| kriteria | syarat | hasil |
|---|---|---:|
| keandalan | ≥ 5/6 | **6/6** |
| cakupan pustaka | ≥ 20 | **22** |
| \|IC\| vs lantai acak (Mann-Whitney) | p > 0,05 | **p = 0,633** |

**LULUS ketiganya.** `design` diganti `innovate` secara permanen (B16); rantai
produksi kini `proposal → innovate → construct` dengan guided decoding aktif di
construct. Detail lengkap & angka pembanding: `lab/HASIL_A8.md` §4b.

*Catatan metodologis*: user memutuskan mengganti `design` SEBELUM angka
`innovate_guided` ini selesai dihitung — menimpa rekomendasi awal alat ("tunda
satu ronde"). Keputusan itu dihormati lebih dulu; hasil ronde lanjutan
kebetulan memenuhi kriteria yang sudah didaftarkan, sehingga tak ada
pelonggaran aturan yang terjadi di sini.

### Tahap 4 — matematika laten (≈2 jam GPU)
**B6** (early-stop) → **B8**+**B9** (anggaran KV yang benar) → baru **B7**
(ganti persamaan realignment secara permanen).
*Gerbang*: B7 hanya dikerjakan kalau Tahap 3 menyimpulkan rantai agennya layak
dipertahankan. **Terpenuhi** — Tahap 3/3b menyimpulkan rantai front-end
(`proposal→innovate→construct`) layak dipertahankan (mengungguli rujukan lama
pada mutu sinyal & cakupan pencarian, dan menyamai lantai acak). B8+B9 sudah
selesai di Tahap 1 (di luar urutan awal, karena keduanya konfigurasi berisiko
rendah); B6 dan B7 masih menunggu Tahap 4.

### Tahap 5 — agen yang belum diuji (≈3 jam GPU)
Baru di sini `mutation`, `crossover`, `feedback` diuji, dengan front-end yang
sudah stabil sebagai fondasi. Pertanyaan spesifiknya: apakah `guidance_kv`
benar-benar mengubah keluaran front-end (bandingkan dengan seed_kv=None pada
parent yang sama), dan apakah feedback mengubah ronde berikutnya.

### Tahap 6 — hanya bila Tahap 3 lulus: **B10** (latent bottleneck)
Riset, bukan perbaikan. Perlakukan sebagai kontribusi terpisah.

---

## D. Apa yang TIDAK saya sarankan, dan alasannya

- **Jangan naikkan ukuran model lagi.** Kolaps mode `kv` bereproduksi di 8B
  persis seperti di 4B (HASIL_GPU §3). Menaikkan ke 14B akan memperbesar biaya
  tanpa menyentuh penyebabnya.
- **Jangan mengejar |IC| lebih dulu.** Lantai acak (0,0170) belum terlampaui
  oleh lengan mana pun; dan AUDIT §2.6 menunjukkan sinyalnya sebenarnya stabil
  di holdout — yang bermasalah adalah fungsi fitness-nya (IC bertanda), bukan
  penemuannya. Memperbaiki fitness (S2/S3 di AUDIT) lebih murah daripada
  memperbaiki generator.
- **Jangan mengubah beberapa lapisan sekaligus.** Nilai skripsi ini ada pada
  rantai sebab yang bisa dipertahankan; sistem yang lebih baik tanpa penjelasan
  justru menurunkan nilainya.
