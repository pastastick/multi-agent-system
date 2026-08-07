# Kesimpulan akhir — apa yang berubah di QuantaLatent, dan apa efeknya

> Cakupan: **seluruh `lab/RENCANA_PERBAIKAN.md` KECUALI Tahap 5** (agen
> `mutation`/`crossover`/`feedback`, yang memang belum pernah diuji sejak awal
> proyek). Backbone **Qwen3-8B**, A40 46 GB, branch `exp/rencana-perbaikan`,
> dikerjakan 2026-08-07.
>
> Dokumen sumber per tahap: `lab/HASIL_GPU.md` (G1–G7, baseline),
> `lab/HASIL_A8.md` (ablasi agen), `lab/HASIL_TAHAP4.md` (B6/B7/A9/B10),
> dan §4–§7 di bawah untuk B5/A10/B14/A11.
>
> **Status artefak**: seluruh angka di dokumen ini terverifikasi terhadap
> `lab/out/*.json` yang di-commit. Satu berkas turunan CPU-only
> (`lab/out/icseries_b14_summary.parquet` — dipakai HANYA untuk deret IC per
> hari, bukan untuk tabel §6) belum sempat diregenerasi ulang saat sesi ini
> dihentikan; `*.parquet` memang gitignored di repo ini. Regenerasi (murni
> CPU, tanpa GPU): `python lab/frontend_probe.py --score-only --tag
> b14_summary`. Tidak ada klaim di §6 yang bergantung padanya — tabel B14
> memakai `lab/analyze_gpu.py`, yang bekerja langsung dari `frontend_*.json`.

---

## 1. Jawaban singkat atas empat pertanyaan

**Apakah IC meningkat?** Untuk sistem secara keseluruhan: **ya, tetapi hanya
relatif terhadap dirinya sendiri, dan TIDAK melampaui lantai acak.** |IC| per-run
rantai produksi naik dari 0,0109 (rantai lama `design`) ke 0,0182
(`innovate` + guided decoding), +67%. Tetapi ekspresi ACAK dari DSL yang sama
mencapai mean |IC| 0,0170, dan lengan produksi sekarang **tidak berbeda secara
statistik dari lantai acak itu** (Mann-Whitney p = 0,633). Membaca ini sebagai
"sistem jadi pandai memilih faktor" tidak dibenarkan datanya.

**Apakah lebih efisien?** **Ya pada token, TIDAK pada waktu.** Redundansi konteks
`construct` turun 59,5% → 2,5%; token prompt `construct` −36%; KV total −22%;
`latent_steps` 60 → 10 tanpa penurunan |IC|; token per faktor diterima 1 095 →
997. Tetapi **detik per faktor diterima justru NAIK 7,0 → 15,8** — guided
decoding (B11) dan agen `innovate` yang menulis lebih panjang membayar
ketepatan format dengan waktu. Menyebut sistem ini "lebih efisien" tanpa
menyebut sumbu mana adalah klaim yang tidak jujur.

**Apakah lebih kredibel?** **Ya, dan ini nilai terbesar sesi ini.** Sebelumnya
sembilan gate semuanya struktural dan 49% ekspresi lolos dalam keadaan cacat;
sekarang 0 dari 198 ekspresi lolos-gate yang masih cacat-semantik, dan tiga
kelas kebocoran baru ditambal (B15) plus gate eksekusi (B12). Lebih penting
lagi, tiga klaim mekanisme yang selama ini dipegang ternyata **salah atau tak
berlaku**, dan sekarang terukur — lihat §2.

**Mode mana yang optimal?** Lihat §6. Jawaban singkatnya: **`kv` — dan bukan
"tergantung sumbu".** Pada konfigurasi produksi sekarang, `kv` menang atau
menyamai `summary` dan `text` di SETIAP sumbu yang diukur (mutu sinyal, laju
lolos gate, biaya waktu, biaya token, cakupan pustaka). Ini berbeda dari
temuan lama (G4): pasca-B16/B11, `kv` tidak lagi harus dipertukarkan antara
keandalan dan mutu — ia menang keduanya sekaligus.

---

## 2. Tiga klaim lama yang gugur (nilai ilmiah terbesar sesi ini)

Ini bagian yang paling penting untuk sidang, karena semuanya adalah koreksi
terhadap dokumen sendiri, bukan angka yang menyenangkan.

**(a) "Realignment ridge adalah mekanisme inti."** Salah untuk backbone yang
dipakai. Pada Qwen3-8B matriks `M` memutar hidden state sampai `cos(h, hM) =
0,011` — praktis ortogonal terhadap masukannya — dan menghasilkan vektor di luar
manifold embedding (cos ke embedding terdekat 0,312 vs 0,985 pada `gumbel`).
Sejak B7 ia **tidak lagi dipakai produksi**, dan flag `use_realign` **inert**
(dibuktikan identik bit-per-bit). Konsekuensi: hasil ablasi G6 hanya berlaku
untuk mode `raw`.

**(b) "Kanal laten membawa 'pikiran' antar-agen."** Tidak untuk muatan simbolik.
A9 mengukurnya langsung: kanal laten murni memulihkan **0,350** muatan
(`latent_steps=10`) sementara token prompt yang ikut diwariskan memulihkan
**1,000**. Jadi mode `kv` lossless **karena token prompt-nya**, bukan karena
vektor latennya ekspresif. Kanal laten meluruh sepanjang urutan
(0,90/0,60/0,25/0,00/0,00) dan gagal sebagai konfabulasi, bukan sebagai keluaran
kosong.

**(c) "`latent_steps` besar = penalaran lebih dalam."** Terbalik. Pada ls=60
mode `kv` menghasilkan **0 ekspresi dari 6 run**; pada ls=5/10 menghasilkan 6/6
tanpa penurunan |IC|. Jalur laten `raw` mencapai titik tetap di langkah 12–34,
jadi langkah setelah itu hanya menyalin vektor yang sama sambil mendesak
konteks.

Satu nuansa yang harus ikut dilaporkan supaya (c) tidak dibaca berlebihan: A9
menunjukkan kapasitas kanal laten **naik** dari 0,350 (m=10) ke 0,840 (m=40).
Jadi B1 membeli keandalan dengan menyempitkan kanal laten. Kedua fakta benar;
melaporkan salah satunya saja menyesatkan.

---

## 3. Perubahan yang BENAR-BENAR diterapkan ke kode produksi

Hanya butir di tabel ini yang mengubah perilaku sistem yang berjalan. Selebihnya
(sumbu A*, prototipe B10) adalah alat ukur atau riset.

| # | perubahan | berkas produksi | efek terukur |
|---|---|---|---|
| **B1** | `latent.steps` 60 → 10 | `settings.py`, 5 × `configs/*.yaml` | keandalan 0/6 → 6/6 run; |IC| tak turun |
| **B2** | `step_mode` `raw` → `gumbel` (T=0,7) | `settings.py`, `configs/*` | lolos gate 54% → 91%; faktor hidup 13 → 27; klaster 6 → 9. **Kekuatan sinyal TIDAK membaik** (t = −0,27) |
| **B4** | prompt ringkas saat `lib_in_kv` | `prompts.yaml` | token prompt construct 2 579 → 1 624 (−37%) |
| **B8** | `kv_truncate` membukukan RoPE | `llm/_shared.py` | KL(perbaikan‖re-rotasi manual) = 0,0 persis |
| **B9** | anggaran KV benar-benar diberlakukan | `settings.py` | `kv_max_tokens` tidak lagi kode mati |
| **B11** | guided decoding di `construct` | `agent.py`, `prompts.yaml` | laju tak-terparse → 0; **TERIKAT rantai `innovate`** (di rantai `design` MERUGIKAN: gate 83% → 62%) |
| **B12** | execution gate | `execution_gate.py` (baru) | menangkap faktor konstan/NaN-total yang lolos gate struktural |
| **B15** | tiga lubang gate (boolean 2-nilai, kuantil di luar [0,1], arity) | `factor_regulator.py` | 17 unit test + regresi 322 ekspresi |
| **B16** | `design` → `innovate` | `pipeline.py`, `prompts.yaml`, `settings.py` | \|IC\|/run +67%; pustaka DSL 13 → 22 fungsi; lolos gate 83% → 88%; **tetapi klaster sinyal TURUN 20 → 9 dan detik/faktor NAIK 7,0 → 15,8** |
| **B6** | early-stop rollout laten | `client.py`, `settings.py`, `configs/*` | **nol-efek di produksi** (0/9 menyala pada `gumbel`); hemat 47% bila `step_mode` kembali ke `raw` |
| **B7** | default persamaan laten `raw` → `gumbel` permanen | `client.py` | menutup celah "default kode ≠ default produksi" |
| **B5** | prompt `proposal`: rantai usang diperbaiki + dipangkas | `prompts.yaml` | §4 |
| **B14** | medium baru `comm_mode="summary"` | `pipeline.py`, `settings.py` | §6 — **tersedia, bukan default; `kv` tetap menang** |

**Yang TIDAK diterapkan, dengan alasan**: B3 (`prompts_v1.yaml` ada tetapi
produksi tetap `prompts.yaml`), B13 (digantikan B16 — slot diisi, bukan
dipangkas), **B10** (ditolak berdasarkan bukti — §4 `HASIL_TAHAP4.md`).

---

## 4. B5 — prompt `proposal`: yang diminta ternyata sudah ada, yang perlu ternyata lain

**Temuan audit.** RENCANA §B5 meminta format keluaran `proposal` diringkas dari
`KNOWLEDGE / OBSERVATION / JUSTIFICATION / SPECIFICATION / SUMMARY` menjadi
`A / B / Final Hypothesis`. Perubahan itu **sudah diterapkan sejak commit
`fe42127`**, jauh sebelum RENCANA ditulis — kontraknya sekarang
`observation / driver / HYPOTHESIS:`. Jadi §B5 adalah **dokumentasi basi, bukan
pekerjaan tertunda**. Format 5-bidang yang dikeluhkan `catatan.txt` masih hidup,
tetapi hanya di `mutation`/`crossover` — agen Tahap 5, di luar cakupan.

**Kenapa mekanisme B5 tak mungkin bekerja di medium produksi.** B5 beralasan
"keluaran lebih pendek → KV lebih pendek untuk hop berikutnya". Tetapi pada
`comm_mode=kv`, `proposal` berjalan `kv_only` dan **tidak men-decode teks sama
sekali** (terukur: `out_tok = 0`). Kontribusinya ke KV adalah **prompt**-nya,
bukan keluarannya. Memendekkan format keluaran karena itu nol-efek di produksi,
dan hanya berpengaruh di `text` / `kv_and_text`.

**Yang justru ditemukan.** Prompt `proposal` masih menjelaskan rantai **lama**:
*"Your hypothesis passes to a Design agent that decides which functions could
express it"* — padahal B16 sudah mengganti `design` dengan `innovate` (Explorer)
yang tugasnya berlawanan: MELEBARKAN hipotesis, bukan menyempitkannya ke palette.
Agen hulu diberi tahu pipeline yang tidak lagi ada. Ini drift pasca-B16.

**Yang dikerjakan**: (i) perbaiki deskripsi rantai, (ii) pangkas redundansi
(definisi "apa itu alpha factor" yang juga sudah ada di prompt `construct`).

**Hasil A/B terkontrol** (6 run/lengan, 2 arah × 3 seed, `kv` ls=10 gumbel;
satu-satunya yang berbeda adalah berkas prompt):

| | prompt SEBELUM B5 | prompt SESUDAH B5 |
|---|---:|---:|
| prompt `proposal` | 866 tok | **788 tok** (−9,0%) |
| KV setelah `proposal` | 889,5 | **811,5** (−78) |
| KV setelah `innovate` | 2 468,5 | **2 390,5** (−78) |
| prompt `construct` | 1 598 | 1 598 (tak berubah) |
| run produktif (A2) | 4/6 | **5/6** |

Penghematan 78 token **merambat persis** ke hilir — bukti bahwa penghematan
prompt hulu tidak "dimakan" balik oleh agen berikutnya. Prompt `construct` tak
tersentuh, sesuai rancangan (B5 hanya menyentuh `proposal`).

*Kejujuran yang harus ikut*: 78 token dari KV construct ±4 900 adalah **1,6%** —
nyata tetapi kecil. Perbaikan A2 dari 4/6 ke 5/6 pada n=6 **tidak bisa diklaim
sebagai efek**; itu satu run, persis di dalam rentang derau. Nilai B5 yang
benar-benar bisa dipertahankan adalah **koreksi drift**, bukan penghematannya.

---

## 5. A10 — apakah sistem membaca arah risetnya?

**Desain**: dua arah yang benar-benar berlawanan pada tiga sumbu sekaligus —
tanda efek (`opp_mom`: lanjutkan tren 30-60 hari) vs (`opp_rev`: balik tajam
1-3 hari) — dijalankan 3 seed/arah, `comm_mode=kv`, ls=10. Dibandingkan jarak
**ANTAR**-arah dengan jarak **DALAM**-arah (seed berbeda, arah sama) sebagai
kontrol derau; tanpa kontrol ini angka kemiripan tak bisa ditafsirkan.

| ukuran | DALAM-arah (kontrol derau) | ANTAR-arah |
|---|---:|---:|
| jarak Jaccard fungsi (↑ = lebih beda) | 0,845 (n=6) | **0,738** (n=8) |
| \|Spearman\| deret IC (↓ = lebih beda) | 0,317 (n=2) | 0,230 (n=4) |

**VONIS: tidak terbukti membaca arah.** Jarak Jaccard ANTAR-arah (0,738) justru
**lebih rendah** (lebih mirip) daripada jarak DALAM-arah (0,845) — dua run
dengan arah yang **sama** tapi seed berbeda menghasilkan himpunan fungsi yang
LEBIH berbeda daripada dua run dengan arah **berlawanan**. Selisihnya (−0,108)
berada jauh di dalam sebaran derau dalam-arah (0,220). Korelasi deret IC
bergerak searah dugaan (antar-arah 0,230 < dalam-arah 0,317, sesuai harapan
kalau arah membedakan sinyal) tetapi n=2 vs n=4 terlalu kecil untuk berarti
apa pun sendirian.

**Satu sinyal kualitatif yang tetap layak dicatat.** Fungsi yang HANYA muncul
di satu arah cukup masuk akal secara domain:

| arah | fungsi khas |
|---|---|
| `opp_mom` (momentum panjang) | `REGRESI`, `SEQUENCE` — cocok untuk menangkap tren |
| `opp_rev` (balik pendek) | `TS_KURT`, `TS_MAD`, `TS_MEDIAN`, `TS_SKEW` — statistik sebaran/ketahanan, cocok untuk range spike |

Jadi vokabuler fungsi **bergeser ke arah yang masuk akal**, tetapi pergeseran
itu tidak cukup besar untuk membuat KESELURUHAN himpunan fungsi satu run lebih
mirip ke arah yang sama daripada ke seed yang sama. Bacaan paling jujur: sistem
ini **membaca arah SEBAGIAN** — cukup untuk memiringkan pilihan fungsi individual,
tidak cukup untuk mendominasi varian seed-ke-seed pada n sekecil ini.

*Batas berlaku*: n=3 seed/arah adalah eksperimen kecil untuk klaim negatif yang
kuat. "Tidak terbukti" bukan "terbukti tidak" — ada kemungkinan efeknya nyata
tapi lebih kecil dari derau pada n ini. Berkas:
`lab/out/direction_sensitivity_a10.json`.

---

## 6. B14 — medium `summary`: mana medium yang optimal?

**Desain.** `comm_mode="summary"` = konteks bersih tiap agen (seperti `text`)
tetapi materi handoff diringkas DETERMINISTIK ke kontrak yang sudah ditegakkan
prompt (baris `HYPOTHESIS:` dari proposal; blok JSON dari innovate) — tanpa
panggilan LLM tambahan. Dijalankan 6 run (2 arah × 3 seed) untuk `summary` dan
`text`, rantai `innovate` + guided decoding aktif (sama seperti produksi).
Dibandingkan dengan `kv` produksi memakai rujukan A8 `innovate_guided`
(HASIL_A8 §4b) — konfigurasi identik (chain, guided decoding, seed & arah)
kecuali medium.

| ukuran | `kv` (produksi) | `summary` (B14) | `text` |
|---|---:|---:|---:|
| run produktif | 6/6 | 6/6 | 6/6 |
| lolos gate | **88%** | 69% | 44% |
| \|IC\|/run | **0,0182** | 0,0127 | 0,0180 |
| pustaka DSL | **22** | 20 | 20 |
| detik/faktor diterima | **15,8** | 24,3 | 32,8 |
| token/faktor diterima | **997** | 1 899 | 2 602 |
| cacat semantik | 0% | 11% | 0% |

**`kv` menang atau menyamai di SETIAP sumbu.** Ini jawaban langsung untuk
"mode mana yang optimal" pada konfigurasi sekarang (Qwen3-8B, chain `innovate`,
guided decoding aktif): **`kv` adalah medium terbaik di sistem ini sekarang**,
bukan cuma tercepat.

**`summary` vs `text` — nilai B14 yang sebenarnya.** `summary` mengalahkan
`text` pada TIGA dari empat sumbu: efisiensi (24,3 vs 32,8 detik/faktor;
1 899 vs 2 602 token/faktor — sekitar **27% lebih hemat** di kedua sumbu) DAN
lolos gate (69% vs 44%), tanpa kehilangan run produktif. Satu-satunya sumbu
yang lebih baik pada `text` adalah cacat semantik (0% vs 11% pada `summary`)
— indikasi bahwa meringkas keluaran hulu kadang membuang nuansa yang
dibutuhkan hilir untuk tetap presisi semantik, meski efeknya kecil (4 dari 36
ekspresi). Welch t pada \|IC\|/run: `summary` vs `text` t=−1,17 (p tak
signifikan pada n=6) — **tak ada beda mutu sinyal yang bisa diklaim** antara
keduanya pada n ini; `summary` menang pada keandalan-format dan biaya, bukan
pada mutu.

**Keputusan: `summary` DIIMPLEMENTASI dan diukur, TIDAK dijadikan default.**
Tiga alasan: (i) `kv` tetap unggul pada sumbu yang paling penting (mutu sinyal
DAN biaya sekaligus) — tak ada alasan berpindah dari medium yang menang; (ii)
motivasi asli B14 ("`text` menang di keandalan, `kv` menang di mutu" — G4 lama)
sudah tidak berlaku pasca-B16/B11: `kv` sekarang menang di keduanya sekaligus;
(iii) `summary` tetap bernilai sebagai **medium cadangan** — kalau backbone
atau chain berubah lagi dan `kv` kembali menunjukkan pola G2 (kolaps pada
konfigurasi tertentu), `summary` adalah titik tengah yang sudah siap pakai
antara keandalan `text` dan efisiensi `kv`, tanpa perlu menulis medium baru
dari nol.

*Kejujuran yang harus ikut*: G4 lama (HASIL_GPU §4) memperingatkan bahwa
**peringkat medium berubah setiap kali backbone, metrik, atau `latent_steps`
berubah**. Tabel di atas berlaku untuk (Qwen3-8B, chain `innovate`, guided
decoding ON, ls=10). Ia BUKAN klaim umum "KV selalu menang" — ia klaim
"KV menang PADA KONFIGURASI PRODUKSI SEKARANG", dan itu klaim yang lebih
sempit tapi jujur.

---

## 7. A11 — stabilitas jangka panjang: VRAM puncak, KV/hop, kebocoran

**Desain.** 6 trajectory berturut-turut dalam SATU proses (`comm_mode=kv`,
ls=10), karena kebocoran hanya terlihat lintas-run. VRAM puncak diukur per
run (`reset_peak_memory_stats`); residu diukur SETELAH `gc.collect()` +
`torch.cuda.empty_cache()` — apa yang tersisa itulah yang benar-benar tak
terlepas.

| ukuran | hasil |
|---|---|
| VRAM puncak di atas bobot | 2 154,7 MB rata-rata [1 483–2 953] |
| residu setelah run (vs sebelum run pertama) | +640,6 MB rata-rata [+630,6 – +657,7] |
| kemiringan residu lintas-run | **+5,71 MB/run** |

**KV kumulatif per hop** (rata-rata 6 run): `proposal` 811,5 tok →
`innovate` 2 390,5 tok (+1 579) → `construct` 4 725,6 tok (+2 335). Angka ini
**cocok persis** dengan pengukuran independen B5 (811,5 dan 2 390,5 — sampai
satu desimal) dan konsisten dengan A5 (KV construct produksi 4 624 tok) —
saling menguatkan bahwa metodologi pengukuran token/KV di seluruh sesi ini
konsisten satu sama lain.

**Kebocoran: sinyal lemah, bukan tanpa sinyal.** Residu per run: 630,6 →
631,3 → 631,3 → 639,4 → 657,7 → 653,1 MB. Ini **bukan** garis lurus naik —
run terakhir justru turun dari run sebelumnya — tetapi juga bukan derau murni
di sekitar konstanta: run 4-5 (657,7 dan 653,1 MB) jelas di atas run 0-2
(~631 MB). Pertumbuhan total ≈ **27 MB dalam 6 run** (≈4,3%). Pada 46 GB VRAM
ini jauh dari mengkhawatirkan dalam horizon pendek, tetapi **kemiringan
positif pada n=6 tidak bisa diabaikan begitu saja** sebagai temuan jangka
panjang — 100 run dengan laju yang sama (dengan asumsi linear, yang TIDAK
terbukti dari 6 titik ini) akan menambah ±450 MB.

**Sumber residu yang paling mungkin, bukan tafsiran tunggal**: `LatentRealigner`
menyimpan matriks `M` dan `target_norm` dalam `_cache` yang di-keying per
`(id(model), device)` — ini **cache yang disengaja**, bukan kebocoran, dan
menjelaskan sebagian besar dari 630 MB dasar (bertahan sejak run pertama,
tidak bertambah). Yang bertambah pelan-pelan (630→657 MB) kemungkinan
fragmentasi CUDA caching allocator atau buffer log (`TensorConvManager`) —
**belum diisolasi mana yang mana**; itu pekerjaan lanjutan, bukan kesimpulan
sesi ini.

**Peringatan n**: 6 run adalah jumlah yang sangat kecil untuk klaim kebocoran.
Verdict "ADA indikasi" di skrip memakai ambang mekanis (>5 MB/run) yang belum
divalidasi terhadap horizon panjang sungguhan — ini **sinyal untuk diselidiki**,
bukan kesimpulan yang bisa dipertahankan sebagai "sistem ini bocor". Berkas:
`lab/out/stability_a11.json`.

---

## 8. Batas berlaku SELURUH kesimpulan ini

Lima batasan yang harus ikut disebut di sidang, karena tanpanya angka-angka di
atas terdengar lebih kuat daripada yang sebenarnya.

1. **n = 6 run per lengan.** Terlalu kecil untuk uji beda pada |IC| ATAU pada
   kebocoran memori. Sumbu yang benar-benar bisa diputuskan pada n ini adalah
   yang variansnya rendah dan efeknya besar: cakupan pustaka, laju lolos gate,
   laju run produktif, dan biaya per faktor. Setiap kali |IC| dipakai untuk
   memutuskan, itu disebutkan sebagai sumbu yang **tidak** membedakan.
2. **Satu backbone, satu pasar, satu periode.** Semua pada Qwen3-8B dan
   CSI300. HASIL_GPU §4 sudah menunjukkan peringkat medium **berbalik** ketika
   backbone / metrik / `latent_steps` berubah — jadi peringkat apa pun di sini
   (termasuk "`kv` menang", §6) berlaku untuk konfigurasi ini, bukan untuk
   "KV vs teks" secara umum.
3. **Tahap 5 belum tersentuh.** `mutation`, `crossover`, `feedback`, dan seluruh
   siklus evolusi belum pernah diuji. Yang berlaku untuk front-end adalah
   **batas atas** bagi sistem evolusioner (karena evolusi memanggil ulang
   front-end yang sama), tetapi apakah `guidance_kv` benar-benar memindahkan
   arah dari Director ke front-end **belum diukur sama sekali**.
4. **Lantai acak belum terlampaui.** Tidak satu pun lengan mengungguli mean
   |IC| ekspresi acak (0,0170) secara signifikan. Semua klaim "membaik" di
   dokumen ini adalah membaik **relatif terhadap konfigurasi sebelumnya**, bukan
   membaik terhadap pencarian tanpa teori.
5. **A10 dan A11 keduanya berakhir pada verdict "sinyal lemah, n terlalu kecil
   untuk klaim kuat".** Ini pola yang konsisten dan harus dilaporkan sebagai
   pola, bukan disembunyikan sebagai dua kegagalan terpisah: n=6 adalah
   anggaran GPU yang wajar untuk *keputusan arsitektur* (A8, B5) tetapi bukan
   untuk *deteksi statistik halus* (sensitivitas arah, kebocoran memori). Kalau
   sumbu itu perlu diputuskan tegas, anggarannya harus dinaikkan ke n=20-30,
   bukan diberi kesimpulan tegas dari n=6.

---
