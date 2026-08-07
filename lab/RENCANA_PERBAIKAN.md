# Rencana perbaikan QuantaLatent — dari temuan G1–G7 ke intervensi

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

### A4. Geometri & entropi jalur laten (sudah ada)
cos ke embedding terdekat, langkah titik tetap, identik-antar-seed,
identik-antar-arah. *Alat*: `lab/latent_dynamics.py`.

### A5. **Efisiensi konteks** (BARU — sebagian sudah diukur di G7)
Tiga metrik, semuanya sudah bisa dihitung `lab/latent_growth.py`:
- **redundansi**: fraksi n-gram prompt hop-k yang sudah ada verbatim di KV
  (terukur: **59,5%** untuk construct);
- **panjang konteks efektif**: `exp(H(attention))` pada langkah emisi — berapa
  token yang *sebenarnya* dipakai dari sekian ribu yang disimpan;
- **massa attention per segmen** dengan *enrichment* = massa ÷ porsi panjang.
*Kenapa penting*: ini menerjemahkan "penumpukan KV" dari keluhan menjadi angka,
dan langsung memberi target optimasi (turunkan redundansi, naikkan enrichment
segmen instruksi aktif).

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

### Tahap 4 — matematika laten (≈2 jam GPU)
**B6** (early-stop) → **B8**+**B9** (anggaran KV yang benar) → baru **B7**
(ganti persamaan realignment secara permanen).
*Gerbang*: B7 hanya dikerjakan kalau Tahap 3 menyimpulkan rantai agennya layak
dipertahankan. Kalau tidak, memperbaiki matematika kanal yang tak dipakai adalah
pekerjaan sia-sia.

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
