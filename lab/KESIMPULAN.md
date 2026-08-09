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
> **Status artefak** (diperbarui 2026-08-08, sesi CPU lanjutan). Seluruh angka
> di dokumen ini terverifikasi terhadap `lab/out/*.json` yang di-commit. Catatan
> versi sebelumnya menyebut **satu** berkas turunan yang belum diregenerasi
> (`icseries_b14_summary.parquet`); itu keliru — pada checkout bersih **tidak
> satu pun** dari 31 `icseries_*.parquet` ada, karena `*.parquet` gitignored.
> Akibatnya seluruh **sumbu A3 (klaster sinyal)** berhenti bisa direproduksi di
> mesin lain, dan A3 adalah sumbu yang menopang keputusan A8/B16. Regenerasi +
> verifikasi silang terhadap angka mesin GPU: **§11**. Bagian §9–§12 ditulis di
> sesi CPU tersebut.

---

## 1. Jawaban singkat atas empat pertanyaan

> **BACA §11.3 LEBIH DULU.** Sesi CPU 2026-08-08 memasang lantai acak pada sumbu
> **cakupan pencarian** — sumbu yang selama ini dipakai memutuskan arsitektur
> tetapi tak pernah punya pembanding. Hasilnya membalik sebagian isi §1–§6 di
> bawah: rantai lama `proposal→design→construct` ternyata **mengungguli
> pencarian acak secara telak pada sumbu itu** (21 klaster vs sebaran nol
> 10 [7–15], p<0,002), sementara rantai produksi sekarang hanya menyamainya
> (p=0,462) dan rantai tanpa `design` berada **di bawahnya** (p=0,988). Karena
> keputusan B16 diambil pada sumbu \|IC\| — yang kini terbukti tidak
> membedakan lengan mana pun dari acak — kalimat "+67%" di bawah benar tetapi
> **tidak mengukur hal yang penting**.

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
4. **Lantai acak belum terlampaui — dan sejak §11.3, itu berlaku pada DUA
   sumbu.** Pada mutu sinyal, tak satu pun lengan mengungguli mean |IC| ekspresi
   acak (0,0170) secara signifikan. Pada **cakupan pencarian**, yang selama ini
   menjadi pembelaan terakhir rantai multi-agen, lantainya baru dihitung di
   sesi CPU 2026-08-08 (kolam acak: 271 ekspresi → 44 klaster) dan tiga lengan
   yang sudah terukur mendarat **di atau di bawah median sebaran nol
   n-tercocok**. Lengan A8 belum ikut terukur — lihat batas berlaku di §11.3.
   Semua klaim "membaik" di dokumen ini adalah membaik **relatif terhadap
   konfigurasi sebelumnya**, bukan membaik terhadap pencarian tanpa teori.
5. **A10 dan A11 keduanya berakhir pada verdict "sinyal lemah, n terlalu kecil
   untuk klaim kuat".** Ini pola yang konsisten dan harus dilaporkan sebagai
   pola, bukan disembunyikan sebagai dua kegagalan terpisah: n=6 adalah
   anggaran GPU yang wajar untuk *keputusan arsitektur* (A8, B5) tetapi bukan
   untuk *deteksi statistik halus* (sensitivitas arah, kebocoran memori). Kalau
   sumbu itu perlu diputuskan tegas, anggarannya harus dinaikkan ke n=20-30,
   bukan diberi kesimpulan tegas dari n=6.

---

## 9. Untuk skripsi: temuan mana menopang klaim apa

Bagian ini memetakan hasil `AUDIT_KRITIS.md` + `RENCANA_PERBAIKAN.md` ke peran
yang bisa mereka mainkan di naskah. Disusun menurut **peran bukti**, bukan
menurut urutan pengerjaan, karena satu angka bisa menopang klaim yang berbeda
di bab yang berbeda.

### 9.1 Kontrol metodologis yang tidak dimiliki korpus pembanding

`skripsi/ANALISIS_SKRIPSI_REFERENSI.md` §3 mendaftar tiga kelemahan **sistemik**
genre skripsi studi-banding ini: (a) tanpa uji signifikansi perbandingan, (b)
satu split tanpa replikasi/CV, (c) kesimpulan yang kadang bertentangan dengan
badan naskah. Dokumen itu menyarankan pembedanya adalah "N rep per mode + uji
beda + CI bootstrap". Yang sudah ada di tangan sekarang menutup ketiganya —
jadi enam hal berikut bukan sekadar kelengkapan, mereka **pembeda yang sudah
terukur**:

| kontrol | isi | dipakai untuk |
|---|---|---|
| **lantai acak (null model)** | 300 ekspresi acak dari DSL yang sama, 271 hidup, mean \|IC\| **0,0170** | menolak klaim "sistem menemukan faktor bagus" tanpa pembanding |
| **holdout sejati** | 2022-01-01…2025-12-26, terpisah dari jendela seleksi 2021 | memisahkan penemuan dari overfitting |
| **unit analisis = run** | bukan per-ekspresi; alasannya terukur (70% faktor batch lama ada dalam SATU klaster sinyal → bukan pengamatan independen) | mencegah nilai-p yang terlalu optimistis |
| **kriteria didaftarkan di muka** | aturan keputusan Tahap 3a/3b ditulis SEBELUM data ada, lalu diterapkan mekanis (`lab/decide_a8.py`) | menutup ruang untuk melonggarkan aturan setelah melihat hasil |
| **replikasi n=6 per lengan** | 2 arah × 3 seed, bukan n=1 seperti batch lama | menjawab kelemahan (b) "satu split tanpa replikasi" |
| **uji beda dilaporkan termasuk saat null** | Welch t & Mann-Whitney dengan n dan arah, juga ketika hasilnya "tidak berbeda" | menjawab kelemahan (a) |

### 9.1b Sumbu penyajian yang lebih kuat daripada "KV vs TEXT"

Judul yang sudah dikunci ("Perbandingan Medium Komunikasi Laten dan Teks…")
tetap muat, karena perbandingan itu memang dilakukan. Tetapi kalau fokus
pembahasan masih boleh digeser, data yang ada sekarang menopang sumbu yang
**lebih tahan sidang** — bukan temuan baru, melainkan penyajian berbeda atas
eksperimen yang sama.

Masalahnya dengan sumbu "KV vs TEXT" sudah terukur: peringkat medium **berbalik
empat kali** ketika backbone, metrik, atau `latent_steps` berubah (§9.4). Sebuah
skripsi yang jantungnya peringkat itu harus mempertahankan kesimpulan yang
datanya sendiri tunjukkan tidak stabil.

Yang stabil di seluruh eksperimen justru ini: **setiap kali sebuah keunggulan
sistem multi-agen diberi lantai pembanding acak, hasilnya berubah makna.**

| sumbu | klaim tanpa lantai | setelah diberi lantai acak |
|---|---|---|
| mutu sinyal (\|IC\|) | "faktor kami ber-IC 0,045" | tak berbeda dari acak (p=0,70); rantai `design` bahkan **lebih buruk** (p=0,011) |
| cakupan pencarian (klaster) | "`full` 21 klaster vs `nodesign` 7" | **benar-benar unggul** — 0 dari 500 sampel acak menyamainya (§11.3) |
| kestabilan tanda IC ke holdout | "sinyal kami bertahan 96%" | acak bertahan 83%, dan **8/8 pada lapis kuat di keduanya** (§11.4) |
| kapasitas kanal laten | "handoff KV lossless" | lossless karena **token prompt**, bukan vektor laten (A9) |

Empat baris itu punya bentuk yang sama, dan itulah yang membuatnya layak jadi
sumbu pembahasan: **klaim tentang sistem LLM hanya bermakna relatif terhadap
lantai yang dirancang untuk mengalahkannya.** Tiga dari empat sumbu runtuh
ketika lantainya dipasang; satu bertahan — dan yang bertahan itu **bukan** yang
selama ini dipromosikan.

Rumusan yang bisa dipakai sebagai fokus: *sistem multi-agen LLM pada skala 8B
tidak menemukan faktor yang lebih KUAT daripada pencarian acak di DSL yang sama,
tetapi menemukan portofolio faktor yang lebih BERAGAM — dan keragaman itu
seluruhnya berasal dari satu agen yang, karena dinilai pada sumbu yang salah,
justru dihapus.*

Struktur naratifnya lengkap dan seluruh datanya sudah ada:

1. **Pasang lantai.** Sampling acak dari DSL yang sama (§AUDIT 2.5, §11.3).
2. **Tiga sumbu runtuh.** Mutu sinyal, kestabilan tanda, kapasitas kanal laten —
   semuanya setara acak atau lebih buruk.
3. **Satu sumbu bertahan, dan sebabnya bisa diisolasi.** Ablasi terkontrol
   `full` vs `nodesign` (beda tunggal = agen `design`): p<0,002 vs p=0,988.
4. **Ironi metodologis sebagai penutup.** Agen itu dihapus oleh prosedur
   keputusan yang sah dan didaftarkan di muka — tetapi yang mengukur pada sumbu
   \|IC\|, sumbu yang buta. *Ablasi yang benar pada metrik yang salah
   menghasilkan keputusan yang salah dengan penuh keyakinan.*

Butir 4 itu pelajaran yang berlaku jauh di luar skripsi ini, dan ia lahir dari
kesalahan yang didokumentasikan sendiri, bukan dari kemenangan. Untuk sidang
statistika, itu bahan yang lebih kuat daripada peringkat medium mana pun.

### 9.2 Bukti mekanistik untuk klaim inti (medium laten vs teks)

Ini yang paling langsung menopang judul. Nilainya: klaim tentang kanal laten
tidak lagi **ditafsirkan dari mutu faktor**, melainkan **diukur langsung**.

- **A9 kapasitas kanal** — muatan simbolik diketahui (k=5), diminta dipulihkan
  agen hilir: `kv_full` 1,000 · `kv_prompt_only` 1,000 · **`kv_latent_only`
  0,350** · `none` 0,000. Mode `kv` lossless **karena token prompt hulu yang ikut
  diwariskan**, bukan karena vektor latennya ekspresif. Kanal laten meluruh
  sepanjang urutan (0,90/0,60/0,25/0,00/0,00) dan gagal sebagai **konfabulasi**
  (`TS_COVARIANCE` → `TS_COVARIABILIDAD`), bukan sebagai keluaran kosong.
- **G1/A4 geometri** — cos vektor laten ke embedding TERDEKAT: 0,074 (8B) /
  0,162 (4B) pada `raw`, versus 0,94 pada `gumbel`. Titik tetap pada langkah
  12–34. Entropi nol (9/9 pasang seed identik). Pada 8B tanpa realignment,
  jalur laten **identik untuk ketiga arah riset** (3/3) — kehilangan informasi
  arah, bukan cuma varians.
- **G7 pertumbuhan lintas hop** — pada `ls=60` blok laten antar-agen 83–86%
  saling mirip: pipeline tiga agen berdegenerasi jadi satu agen yang mengulang
  dirinya. Dan menaikkan `latent_steps` justru **mengecilkan** pengaruh kanal
  laten (enrichment attention 8,72× → 0,48×).
- **G2 dosis–respons** — 6/6 → 5/6 → 1/6 → 0/6 → 0/6 run produktif untuk
  `ls` ∈ {5,10,20,40,60}. Monoton dan bersih; ablasi nyata, bukan anekdot.

### 9.3 Koreksi terhadap paper yang dirujuk

Ini bagian yang paling bisa dipertahankan di sidang, karena semuanya turun dari
teks paper itu sendiri plus pengukuran:

- **Teorema 3.3 LatentMAS ("lossless")** adalah pernyataan tentang benarnya
  KV-caching, bukan tentang laten versus teks. Konsekuensinya dua arah:
  (i) kalimat "transfer KV lossless" **tidak boleh** dipakai membela mode `kv`
  murni; (ii) hasil `kv_and_text ≈ text` **bukan hasil nol** — itu justru satu-
  satunya prediksi yang bisa diturunkan dari teorema itu, dan data kita
  konsisten dengannya. A9 mengonfirmasi teoremanya sekaligus membantah tafsiran
  populernya.
- **Teorema 3.1 (ekspresivitas)** menghitung himpunan yang bisa
  **direpresentasikan**; yang relevan untuk sistem *training-free* adalah
  himpunan yang bisa **dicapai**. Karena `h_{k+1} = f(h_k)` deterministik, dari
  satu prompt himpunan terjangkaunya adalah satu lintasan — dan lintasan itu
  konvergen. Ekspresivitas per-langkah dan entropi per-langkah adalah dua
  besaran berbeda; benchmark LatentMAS semuanya bertipe jawaban-tunggal (yang
  butuh besaran pertama), pencarian evolusioner butuh yang kedua.
- **Realignment tidak portabel antar-backbone.** `‖M−I‖/‖I‖` = **1,4×10⁻⁶** di
  Qwen3-4B (*tied*) versus **1,040** di Qwen3-8B (*untied*), dengan
  `cos(h, hM)` = 0,011. Baris kode yang SAMA adalah *no-op* di satu backbone dan
  transformasi dominan di backbone lain. Konsekuensi untuk paper: kenaikan
  akurasi yang LatentMAS laporkan pada 4B **tidak dapat** diatribusikan ke
  realignment.
- **QuantaAlpha dibandingkan pada kategori yang salah.** IC 0,1501 di Tabel 1
  adalah IC **prediksi model**, bukan IC per-faktor; IC per-faktor mereka ada di
  Tabel 3 (0,0465–0,0793, dan ada faktor ber-Rank IC **−0,072**). Premis
  "faktor kita jauh tertinggal" runtuh — tanpa berbalik jadi "kita setara",
  karena universe dan periode berbeda, dan karena faktor terbaik kita
  rank-ekuivalen dengan kolom mentah sementara faktor mereka benar-benar
  tersusun.

### 9.4 Hasil negatif yang dirancang untuk bisa gagal

Genre "menunjukkan kelemahan suatu metode" hanya kuat kalau eksperimennya
punya cara untuk membuktikan sebaliknya. Lima yang memenuhi syarat itu:

1. **Sistem multi-agen tidak mengungguli sampling acak.** Batch lama p=0,70;
   pasca-perbaikan rantai `full` justru **lebih buruk** dari acak (p=0,039),
   dan `innovate_guided` **menyamai** (p=0,633). Perubahan tanda, bukan
   kemenangan — dan dilaporkan begitu.
2. **Faktor terbaik seluruh sistem rank-ekuivalen dengan kolom masukan mentah.**
   `RANK($volume)·(TS_RANK($return,1)?−1:1)` ≡ `−RANK($volume)`, identik pada
   setiap statistik (IC +0,04493, t +6,72). Hipotesis yang menyertainya
   ("small-cap … high volume") tidak terimplementasi sama sekali.
3. **B10 (latent bottleneck) ditolak oleh datanya sendiri.** 4 keluarga
   training-free × 3 anggaran × 2 posisi muatan → pooling memulihkan **0,000**
   di semua sel. Pembedaan yang menentukan: cache-nya **tidak rusak, ia kosong**
   (keluaran hilir tetap koheren dan patuh-format). Dan `select_recent` yang
   "menang" ternyata cuma bias resensi (0,920 saat muatan di akhir, 0,060 saat
   di tengah) — yang ketahuan hanya karena posisi muatan dijadikan variabel.
4. **B11 (guided decoding) berefek berlawanan tergantung rantai.** Merugikan di
   rantai `design` (lolos gate 83%→62%, biaya 7×, halusinasi nama fungsi
   `TS_RESIDUAL`), menguntungkan di rantai `innovate`. Intervensi yang sama,
   tanda yang berbeda.
5. **A10: tidak terbukti membaca arah riset**, dengan kontrol derau yang benar —
   jarak ANTAR-arah (0,738) justru lebih kecil daripada jarak DALAM-arah
   (0,845). Tanpa kontrol dalam-arah, angka 0,738 bisa dilaporkan sebagai
   "sistem responsif terhadap arah".

Tambahan dari sesi CPU 2026-08-08, dengan watak yang sama: **anggaran waktu
skoring ternyata membuang 26–30% ekspresi lengan `innovate` tetapi hanya 0–6%
ekspresi lengan `design`**, sehingga perbandingan headline antara keduanya
berdiri di atas himpunan yang tak setara (§11.2). Temuan ini melawan
kepentingan konfigurasi yang sedang dipakai produksi, dan ditemukan hanya
karena korpus diskor ulang di mesin lain — persis jenis pemeriksaan yang
membuat angka bisa dipertahankan.

Dan satu temuan yang formatnya tidak biasa tetapi paling jujur: **peringkat
medium berbalik setiap kali backbone, metrik, atau `latent_steps` berubah**
(Bab 4 lama: `kv` terburuk → AUDIT §3.3: `text ≈ kv > kv_and_text` → G4:
`kv_and_text ≈ kv > text` → KESIMPULAN §6: `kv` menang di semua sumbu).
Kesimpulan yang tidak tahan terhadap tiga hal itu tidak boleh dilaporkan
sebagai temuan — dan **justru itulah temuan yang bisa dipertahankan**.

### 9.5 Koreksi yang WAJIB masuk naskah, bukan opsional

Empat hal di bawah membuat kalimat yang sekarang ada di naskah menjadi tidak
akurat terhadap kode. Memperbaikinya bukan pilihan.

- **Jendela OOS (M3).** Bab 3 menyebut split test 2022–2025;
  `conf_combined_factors.yaml` berisi `test: [2021-01-01, 2021-12-31]`. Seluruh
  `FactorIC_mean` di Bab 4 dihitung pada **2021 saja, 243 hari** — dan 2021 juga
  jendela yang dipakai evolusi untuk **memilih** induk. Menyebutnya "OOS"
  menyesatkan; itu metrik **seleksi**.
- **Universe (M4).** IC per-faktor dihitung lintas **±4.370 emiten per hari**
  (terukur), yaitu hampir seluruh pasar A-share — bukan 300 emiten CSI 300.
  Hanya backtest portofolio LightGBM yang memakai CSI 300. Dua metrik utama
  skripsi berjalan di dua universe berbeda.
- **Bab 4 §Operasi KV-Cache — tiga dari empat submekanisme tidak berjalan.**
  Terverifikasi ulang pada kode saat ini:

  | subbab | status jalur eksekusi |
  |---|---|
  | Penggabungan Hierarkis (`kv_concat`, dinyatakan transkripsi LatentMAS Eq. 4) | **nol pemanggil** di seluruh basis kode; dan tanpa re-rotasi ia menumpuk dua rentang posisi RoPE |
  | Pemotongan (`kv_truncate`) | persamaan yang ditulis naskah = versi yang **terbukti salah** (KL 5,09 vs 0,94 terhadap rujukan konteks segar); sudah diperbaiki di kode (B8) |
  | Penyaringan KNN | `knn_enabled` **otomatis dimatikan** setiap `latent_steps > 0` (`llm/client.py:683`) — yaitu selalu, di produksi |
  | Isolasi (deepcopy) | berjalan |

  Konsekuensinya harus dinyatakan: klaim bahwa sistem mengimplementasikan
  *transfer working-memory hierarkis* **tidak didukung jalur eksekusi**. Materi
  induk memang ditransfer pada crossover — tetapi sebagai **teks**.
- **Bab 4 §Realignment Laten** harus ditulis ulang sebagai **sejarah
  mekanisme** ("ridge M adalah rancangan awal; diukur, ditemukan ortogonal,
  diganti"), bukan sebagai deskripsi sistem yang berjalan: sejak B7 produksi
  memakai `gumbel`, dan `use_realign` **inert** (dibuktikan identik bit-per-bit).

---

## 10. Untuk sistem: peluang yang tersisa, diurutkan menurut bukti/biaya

§3 mendaftar apa yang sudah diterapkan. Bagian ini mendaftar apa yang **belum**,
dan seberapa kuat buktinya.

### 10.1 Sudah didukung data yang ada, belum diterapkan — CPU, murah

**(a) Fungsi fitness: IC bertanda → \|IC\| dengan tanda ditetapkan di jendela
latih.** [AUDIT §S3 + §2.6] Ini lever tunggal terbesar yang tersisa, dan ia
tidak menyentuh generator sama sekali.

Keadaan kode sekarang (diverifikasi ulang di sesi ini):
`pipeline/evolution/trajectory.py:100` `get_primary_metric()` mengembalikan
`FactorIC_mean` **bertanda**, dan `:124 is_successful()` menuntut `ic > thr_ic`
(default 0,0). Maka faktor ber-IC −0,043 dengan t = −11,56 di holdout — alfa
stabil yang tinggal dibalik tandanya — dinyatakan **GAGAL**.

Premisnya — apakah tanda IC memang bertahan di luar jendela seleksi — bisa
diuji **tanpa GPU**, dan alatnya dibuat di sesi ini: `lab/sign_persistence.py`
(§11.4). Ia sengaja memakai kelompok **berlapis** kuat/tengah/lemah, bukan 12
faktor terkuat seperti AUDIT §2.6, karena memilih yang terkuat saja menguji
premis pada kelompok yang paling menguntungkannya. Ia juga menjalankan lantai
acak pada sumbu yang sama: kalau ekspresi acak sama stabilnya, stabilitas tanda
itu sifat data, bukan prestasi sistem.

Risiko yang harus dijaga, dan harus dijaga
dengan presisi: tanda WAJIB ditetapkan pada `train` (2016-01-01…2019-12-31) atau
`valid` (2020) — **bukan** pada 2021. Jendela 2021 adalah jendela **seleksi**
in-loop (`conf_combined_factors.yaml` `segments.test`), jadi menetapkan tanda di
sana lalu menilai |IC| di sana juga adalah look-ahead terhadap metrik seleksi
itu sendiri. Ini persis jenis kesalahan yang §S4 keluhkan, dan mudah terulang
saat memperbaiki §S3.

**(b) Correlation store yang simetris.** `factors/runner.py::_update_corr_store`
hanya menyimpan faktor ber-`factor_ic > 0` ke store (diverifikasi ulang).
Akibatnya memori redundansi buta terhadap separuh ruang faktor — dan itu
mendorong sistem menemukan ulang duplikat yang sama. Perbaikannya: kriteria
`|IC|`, bukan `IC`.

**(c) `factor_ic = None` yang mencampur dua hal.** Correlation gate menghapus
nama yang di-drop dari `exp.factor_ic`, sehingga faktor ber-IC nyata tercatat
`None` (22 dari 41 di batch lama; terkonfirmasi ulang pada run baru di
HASIL_GPU §10.1 — faktor `_2`, IC nyata −0,0246). Ini bukan cuma soal
pelaporan: bila `factor_ic` kosong, `get_primary_metric()` jatuh ke fallback
`RankIC` gabungan yang tercemar baseline.

**(d) Pakai `prompts_v1.yaml` (B3) di produksi.** Terukur pada desain 2×2:
cacat semantik **14%→3%** (4B) dan **11%→3%** (8B) — 4× pada KEDUA model,
sementara menaikkan ukuran model hampir tak berpengaruh (14% vs 11%). Produksi
masih memuat `prompts.yaml` (`latent_mas/agent.py:46`). Gate B12/B15 memang
sudah menolak cacat itu, tetapi **mencegah lebih murah daripada menolak**:
setiap tolakan membakar putaran repair.
*Kejujuran*: v1 **tidak** menaikkan \|IC\| (4B −15%, 8B +63%, n=6, tak
signifikan). Manfaatnya keandalan dan biaya, bukan mutu sinyal.

### 10.2 Sumbu yang MEMBURUK dan belum ditangani

> ⚠️ **Dibaca ulang oleh §11.3 — dan rekomendasinya berbalik.** Sejak sumbu ini
> punya lantai acak, butir (e) bukan lagi "satu sumbu yang memburuk" melainkan
> **satu-satunya sumbu tempat sistem terbukti mengungguli pencarian acak, dan
> sumbu itulah yang dihapus B16**. Ablasi terkontrol `full` vs `nodesign`
> (satu-satunya perbedaan = agen `design`) memberi p<0,002 vs p=0,988 pada
> sumbu yang sama. Karena itu prioritas nomor satu untuk sistem sekarang
> **bukan** menambal klaster, melainkan:
>
> **(e0) Kembalikan `design`, atau jalankan rantai `proposal → design →
> innovate → construct`.** Keduanya belum pernah diuji bersama. Hipotesis yang
> bisa diuji dan bisa gagal: `design` menyumbang keragaman sinyal (p<0,002),
> `innovate` menyumbang mutu sinyal (satu-satunya lengan yang menyamai lantai
> acak pada \|IC\|) — kalau keduanya aditif, rantai gabungan mengungguli lantai
> acak pada KEDUA sumbu sekaligus, yang belum pernah dicapai lengan mana pun.
> Biayanya satu ronde GPU 6 run, dan kriteria lulusnya bisa didaftarkan di muka
> persis seperti Tahap 3a/3b: klaster > p95 sebaran nol **dan** \|IC\| tak beda
> dari lantai acak.

**(e) Klaster sinyal turun 20 → 9 saat `design` diganti `innovate`.** Ini satu-
satunya sumbu di mana konfigurasi produksi sekarang **lebih buruk** daripada
rantai lama, dan §3 sudah mencatatnya tetapi belum ada intervensi. Bacaan
mekanisnya: `innovate` melebarkan **pustaka fungsi** (13→22) tanpa melebarkan
**ruang sinyal** — fungsi baru dipakai untuk menghasilkan sinyal yang tetap
saling berkorelasi.

Kandidat yang konsisten dengan bukti: seleksi/gate yang menghukum kemiripan
**deret IC**, bukan kemiripan sintaksis. Correlation gate yang ada bekerja pada
nilai faktor dalam satu ronde; sumbu klaster (`analyze_gpu.signal_clusters`)
bekerja pada deret IC lintas-ronde dan sudah tersedia sebagai alat. §11.2
memberi lantai acak untuk sumbu ini, yang selama ini belum ada.

### 10.3 Butuh GPU — ronde berikutnya

**(f) `latent_steps` optimal antara 10 dan 40.** A9 menunjukkan kapasitas kanal
naik tajam (0,350 → 0,840) sementara B1 menurunkannya ke 10 demi keandalan
(G2: 0/6 pada ls=60). Titik yang memaksimalkan kapasitas DENGAN keandalan 6/6
belum dicari — dan sekarang bisa dicari, karena kedua sumbunya sudah terukur
dan B6 membuat biayanya tak lagi linear terhadap anggaran.

**(g) Tahap 5** — `mutation`, `crossover`, `feedback`. Satu-satunya bagian
RENCANA yang belum tersentuh. Pertanyaan intinya sudah punya alat: A9 bisa
menguji apakah `guidance_kv` benar-benar memindahkan arah dari Director ke
front-end, atau hanya tampak begitu.

**(h) `innovate_lean`** — didaftarkan di HASIL_A8 §5 tetapi tak jadi dijalankan
karena lengan guided sudah lulus. Guided decoding membayar keandalan dengan
waktu (7,0 → 15,8 detik per faktor diterima). Kalau kolapsnya memang disebabkan
register instruksi-meta, prompt yang dipangkas bisa memberi keandalan yang sama
dengan biaya lebih rendah.

### 10.4 Yang tetap TIDAK disarankan

- **Jangan naikkan ukuran model.** Kolaps `kv` bereproduksi di 8B persis seperti
  di 4B; dan pada mutu sinyal, 4B dengan prompt lama (0,0139) **menyamai** 8B
  dengan prompt baru (0,0137), sementara 8B dengan prompt lama justru terburuk
  (0,0084). Tidak ada efek kapasitas yang monoton pada mutu sinyal.
- **Jangan mengejar \|IC\| lewat generator.** Keempat sel 2×2 model×prompt
  berada di 0,0084–0,0139, semuanya di bawah lantai acak 0,0170. Mutu sinyal
  adalah sifat **data × DSL**.
- **Konsekuensi yang jarang disebut**: kalau batasnya memang data × DSL, lever
  yang belum pernah dicoba bukan di sisi LLM melainkan di sisi **masukan** —
  hipotesis yang dihasilkan sistem didominasi tema *small-cap* padahal kolom
  kapitalisasi **tidak ada** di `daily_pv.h5` (kolomnya `$open $close $high
  $low $volume $factor`, `$return` diturunkan). Agen diminta berteori tentang
  besaran yang tak bisa ia hitung. Menambah kolom fundamental akan memperbesar
  ruang yang bisa dicari; memperbaiki prompt tidak.

---

## 11. Sesi CPU lanjutan (2026-08-08) — verifikasi artefak & tiga temuan baru

Dikerjakan di mesin **lokal tanpa GPU**, branch `exp/rencana-perbaikan`.
Tujuannya menuntaskan sisa pekerjaan yang tidak butuh GPU dan memverifikasi
bahwa angka-angka di dokumen ini bisa dilahirkan ulang di mesin lain.

### 11.1 Deret IC (`icseries_*.parquet`) — bukan satu berkas, tapi semuanya

Catatan status di kepala dokumen ini menyebut **satu** berkas turunan yang belum
diregenerasi (`icseries_b14_summary.parquet`). Itu terlalu optimistis: pada
checkout bersih **tidak satu pun** dari 31 berkas itu ada, karena `*.parquet`
gitignored. Konsekuensinya lebih besar daripada satu tabel — **seluruh sumbu A3
(klaster sinyal) berhenti bisa direproduksi**, padahal A3-lah sumbu bervarians
rendah yang menopang keputusan A8/B13/B16 (|IC| justru sumbu yang TIDAK
membedakan pada n=6).

Alat yang dibuat untuk menutup ini:

| berkas | fungsi |
|---|---|
| `lab/rescore_all.py` | skor ulang korpus dengan cache ekspresi **global lintas-tag** (805 ekspresi → 631 unik) + bandingkan tiap IC dengan angka yang tersimpan dari mesin GPU |
| `lab/rescore_all.sh` | jalankan satu tag per proses — proses panjang tumbuh terus dan **dua kali dibunuh OOM** di mesin 8 GB; cache di disk yang membawa hasil antar-proses |

Dua perbaikan efisiensi yang menyertainya, keduanya tidak mengubah satu angka
pun:

- `lab/core.py::ic_full` dulu menghitung deret IC harian **dua kali** (sekali di
  `ic_of_values`, sekali lagi untuk deretnya). Digabung jadi `_ic_core` —
  groupby-spearman atas ±1 juta baris adalah biaya dominan skoring, bukan
  evaluasi ekspresinya.
- `REGBETA`/`REGRESI`/`BB_*` dipanggil dengan `joblib Parallel(n_jobs=-1)`, jadi
  satu ekspresi men-spawn satu worker per core dan tiap worker mem-fork data
  pasar (**16 worker × 328 MB = 5,2 GB terukur**). Di A40 46 GB itu tak terasa;
  di mesin CPU ia penyebab OOM. `lab/core.py` kini membatasinya lewat
  `LOKY_MAX_CPU_COUNT` (default 3, naikkan dengan `LAB_MAX_WORKERS`).
  Catatan operasional: worker loky **selamat dari kematian induknya** dan tetap
  memegang 5 GB sampai dibunuh manual.

**Verifikasi silang.** Untuk tiap tag, IC yang baru dihitung dibandingkan
dengan IC yang tersimpan di `frontend_*.json` (hasil mesin GPU):

Sampai dokumen ini ditulis: **21 dari 31 tag** selesai diskor ulang, 21
`icseries_*.parquet` dilahirkan kembali, dan **12 selisih** di atas toleransi
1×10⁻⁶. Angka "12" itu terdengar buruk sampai isinya dilihat:

| arah selisih | jumlah | artinya |
|---|---:|---|
| `None → nilai` | **11** | ekspresi yang di mesin GPU kehabisan anggaran waktu, di sini selesai |
| `nilai → None` | **1** | kebalikannya (`g6_realignOFF`, satu `REGRESI`) |
| **nilai → nilai berbeda** | **0** | — |

**Tidak ada satu pun IC yang dihitung di kedua mesin dan menghasilkan angka
berbeda.** Seluruh selisih adalah anggaran waktu skoring yang mengikat berbeda
pada perangkat keras berbeda — dan ia mengikat ke **dua arah**, yang justru
membuktikan itu artefak komputasi, bukan bias sistematis satu mesin. Ini
sekaligus bukti kuantitatif untuk §11.2.

Tiga tag pertama yang diperiksa (`a10`, `b14_summary`, `b14_text`) bahkan nol
selisih sama sekali. Replika CPU di mesin ini
melahirkan ulang angka yang dilaporkan dokumen, pada perangkat keras dan
instalasi yang sama sekali berbeda dari tempat angka itu dibuat. Empat angka
rujukan AUDIT §2.1 dan §2.6 juga direproduksi persis: `−RANK($volume)` →
IC +0,04493 (t +6,72), `TS_MEAN($volume,10)` → −0,03868,
`RANK(TS_STD($high,10))` → −0,03113.

Ini menjawab pertanyaan sidang yang paling mudah diajukan dan paling sulit
dijawab tanpa persiapan: *"kalau saya jalankan ulang, angkanya sama?"*

### 11.2 TEMUAN BARU — anggaran waktu skoring membebani lengan `innovate` secara asimetris

Ini tidak dicari; ia muncul karena korpus diskor ulang di mesin dengan
karakteristik berbeda.

Skoring CPU memberi **anggaran 90 detik per ekspresi** supaya satu operator
lambat tak menyandera sweep. Anggaran itu memang mengikat, dan ekspresi yang
melewatinya tercatat tanpa IC — jadi ia **keluar dari statistik \|IC\|**.
HASIL_TAHAP4 §1 sudah menyebutnya sebagai artefak yang diketahui, tetapi hanya
untuk satu faktor di satu uji asap. Ketika dihitung untuk seluruh korpus, pola
yang muncul tidak acak:

| lengan | ekspresi | kena anggaran waktu | rantai |
|---|---:|---:|---|
| `b5_post` | 30 | **30,0%** | innovate |
| **`a8_kv_innovate_guided`** *(= konfigurasi produksi)* | 32 | **28,1%** | innovate |
| `a8_kv_innovate` | 23 | **26,1%** | innovate |
| `b5_pre` | 23 | **26,1%** | innovate |
| `b14_text` | 36 | 13,9% | innovate |
| `b14_summary` | 36 | 11,1% | innovate |
| **`a8_kv_full`** *(= rujukan pembanding)* | 35 | **5,7%** | design |
| `a8_kv_direct` | 18 | 5,6% | construct saja |
| `a8_kv_innovate_fid` | 33 | **0,0%** | innovate + klem kesetiaan |
| `a8_kv_nodesign` | 35 | 0,0% | proposal→construct |
| seluruh 14 lengan seri G / px | 26–36 | **0,0%** | design |

**Mekanismenya jelas dan konsisten dengan alasan `innovate` diadopsi.** Dari 39
ekspresi unik yang kena anggaran: **21 memakai `REGRESI`/`REGBETA`** (joblib
per-instrumen) dan **18 memakai statistik momen bergulir** (`TS_MAD`, `TS_KURT`,
`TS_SKEW`, `TS_MEDIAN`). Keduanya persis kelompok fungsi yang `innovate` buka
dan `design` tidak pernah sentuh. Buktinya paling tajam ada di baris
`innovate_fid`: agen yang sama, tetapi dengan klem kesetiaan yang menekan
eksplorasi (16 vs 23 fungsi pustaka) — dan tingkat timeout-nya **0%**.

> Jadi keunggulan cakupan `innovate` dibayar dengan biaya komputasi yang tidak
> pernah masuk sumbu A6 — A6 menghitung token dan detik **LLM**, bukan detik
> **skoring** — dan sebagian dibayar dengan hilangnya faktor dari statistik mutu.

**Apa yang ini ubah, dan apa yang TIDAK.**

Yang berubah: klaim headline "\|IC\|/run `innovate_guided` 0,0182 vs `full`
0,0109 (+67%)" dihitung pada dua lengan yang **tidak setara secara pengukuran**
— 28% ekspresi lengan pertama tak pernah dinilai, versus 6% pada pembandingnya.
Angka itu harus dilaporkan dengan kualifikasi tersebut sampai selisihnya
ditutup. Arah biasnya **belum diketahui**: faktor yang hilang bisa lemah
(sehingga +67% terlalu optimistis) atau kuat (sehingga terlalu pesimistis).

Yang **tidak** berubah: keputusan B16 tidak bergantung pada sumbu itu. Ketiga
kriteria Tahap 3b yang didaftarkan di muka adalah keandalan (6/6), cakupan
pustaka (22), dan \|IC\| **tidak berbeda dari lantai acak** (p=0,633) — dua yang
pertama tak tersentuh anggaran waktu, dan yang ketiga adalah uji *tidak-beda*
yang tidak dimenangkan oleh faktor yang hilang.

**Arah biasnya kemudian TERUKUR, bukan ditebak.** Saat `a8_kv_innovate_guided`
— lengan konfigurasi produksi — diskor ulang di mesin ini, **3 ekspresi yang di
mesin GPU tercatat `ic=None` ternyata selesai dan punya IC nyata**. Ketiganya
persis kelompok yang diramalkan: dua `REGRESI`, satu `TS_MAD`/`TS_ARGMAX`.
Jadi arm produksi "mendapat" 3 faktor hanya karena diskor di perangkat keras
lain — demonstrasi paling langsung bahwa statistik \|IC\| lengan ini
**bergantung pada mesin**, bukan hanya pada mutu faktornya.

| | n faktor hidup | mean \|IC\| |
|---|---:|---:|
| `a8_kv_innovate_guided` sebelum pemulihan | 20 | 0,01802 |
| **sesudah 3 faktor dipulihkan** | **23** | **0,01794** |
| lantai acak | 271 | 0,01698 |

Faktor yang dipulihkan: \|IC\| = 0,0364 · 0,0131 · 0,0027 — satu jauh di atas
lantai acak, satu di sekitar, satu jauh di bawah.

**Kesimpulannya melegakan dan harus dikatakan sejelas temuannya**: karena
kelompok yang hilang tidak seragam, mean \|IC\| lengan ini **nyaris tak bergeser**
(−0,00008). Jadi klaim "+67%" **tidak** terbukti sebagai artefak anggaran waktu.

**Tetapi ketidaksetaraannya BELUM tertutup, dan ini harus dinyatakan tepat.**
Skoring ulang di mesin ini memulihkan sebagian saja — dan justru memulihkan
lebih banyak pada lengan yang tadinya lebih sedikit kehilangan:

| lengan | timeout semula | dipulihkan | **masih hilang** |
|---|---:|---:|---:|
| `a8_kv_full` | 2 | 2 | **0** |
| `a8_kv_nodesign` | 0 | — | **0** |
| `a8_kv_innovate_guided` | 9 | 3 | **6 dari 32 (19%)** |
| `a8_kv_innovate` | 6 | 3 | **3** |

Jadi \|IC\| = 0,0105 pada `full` kini dihitung atas **seluruh** ekspresinya,
sementara `innovate_guided` masih kehilangan sebagian.

**Ronde pemulihan lanjutan** (`lab/rescore_timeouts.py --budget 900`, anggaran
10× lipat, 8 worker joblib) menutup sebagian besar sisanya dan memberi jawaban
akhir atas pertanyaan "apakah +67% itu artefak":

| lengan | n faktor sebelum → sesudah | mean \|IC\| sebelum → sesudah |
|---|---:|---:|
| `a8_kv_full` | 27 → 27 *(sudah tuntas)* | 0,01050 → 0,01050 |
| `a8_kv_innovate_guided` | 23 → **25** | 0,01794 → **0,01782** |
| `a8_kv_innovate` | 16 → **19** | 0,01739 → **0,01641** |
| `b5_pre` | 14 → 17 | 0,01675 → 0,01577 |

**Setiap faktor yang dipulihkan sedikit MENURUNKAN mean lengannya** — karena
ekspresi ber-operator mahal ternyata rata-rata bermutu di bawah lengannya
sendiri (\|IC\| pulih: 0,0067 · 0,0022 · 0,0027 · 0,0135 · 0,0174 · 0,0232).
Tetapi geserannya kecil (−0,0001 sampai −0,001), sehingga jarak `full` vs
`innovate_guided` bertahan di kisaran +70%.

**Vonis: klaim "+67%" bukan artefak anggaran waktu, dan sekarang itu terukur,
bukan diasumsikan.** Yang tersisa: satu ekspresi (`TS_KURT($high−$low,5) −
TS_SKEW(...)`) tetap gagal bahkan pada anggaran 900 detik — rolling momen
tinggi memang di luar jangkauan harness CPU ini, dan itu batasan alat, bukan
sifat faktornya.

*Tindak lanjut yang benar* (CPU, tanpa GPU): skor ulang ke-39 ekspresi itu
dengan anggaran besar dan worker joblib lebih banyak, lalu hitung ulang
\|IC\|/run kedua lengan pada himpunan yang setara. Alatnya sudah ada
(`lab/rescore_timeouts.py`); perintahnya di §11.4.

### 11.3 Lantai acak untuk sumbu CAKUPAN — lubang yang belum pernah ditutup

AUDIT §2.5 memberi lantai acak untuk **mutu sinyal**, dan sejak itu setiap
lengan diadu dengan mean \|IC\| = 0,0170. Tetapi seluruh rantai keputusan
arsitektur — A8, B13, B16, dan pembelaan terhadap agen `design` — bertumpu pada
sumbu yang **berbeda**: klaster sinyal (A3), justru karena A3 bervarians rendah
pada n=6 sementara \|IC\| tidak. Sumbu itu **tidak pernah punya lantai acak**.
Kalimat seperti "`full` menyebar ke 20 klaster, `nodesign` hanya 7" karena itu
selama ini dibaca sebagai keunggulan tanpa ada yang mengatakan berapa klaster
yang didapat **tanpa agen sama sekali**.

`lab/random_clusters.py` menutupnya. Perbandingan mentah tidak cukup karena
jumlah klaster tumbuh dengan jumlah ekspresi — lengan dengan 33 faktor hidup
hampir pasti mengalahkan lengan dengan 14, tanpa kaitan apa pun dengan mutu
pencarian. Karena itu pembandingnya **bootstrap n-tercocok**: untuk lengan
ber-k faktor hidup, ambil k ekspresi acak dari kolam acak, hitung klasternya,
ulang B kali → sebaran nol; yang dilaporkan adalah posisi lengan di dalamnya.

**Hasil.** Kolam acak: **271 ekspresi hidup → 44 klaster sinyal**. Bootstrap 500
ulangan, ambang |Spearman| > 0,7 (definisi A3 apa adanya). Tabel di bawah memuat
lengan **pasca-Tahap-2** (A8 + B14), yaitu yang menopang keputusan B16; trio
medium G4 yang berkonfigurasi lain dibahas terpisah di bawahnya supaya tidak
tersilang. Lengan ber-n terlalu kecil dan lengan yang parquet-nya belum
diregenerasi tidak ditampilkan.

Seluruh 31 tag selesai; 500 ulangan bootstrap. Lengan diurutkan menurut posisi
terhadap lantai (rantai `design` ditandai **D**, `innovate` **I**, tanpa keduanya **—**):

| lengan | rantai | hidup | klaster | acak n-tercocok | p(acak ≥ LLM) |
|---|:-:|---:|---:|---:|---:|
| **`a8_kv_full`** | **D** | 27 | **21** | 10 (7–15) | **<0,002** |
| `px_8B_v1` *(text, prompt v1)* | **D** | 25 | 16 | 10 (6–14) | **0,004** |
| `g2_kv_ls5` *(`latent_steps`=5)* | **D** | 22 | 16 | 9 (6–13) | **0,006** |
| `a8_kv_full_guided` | **D** | 17 | 12 | 8 (4–11) | **0,032** |
| `px_8B_v0` *(text)* | **D** | 21 | 13 | 9 (6–12) | **0,048** |
| `g4_text` | **D** | 21 | 13 | 9 (6–13) | 0,052 |
| `a8_kv_innovate_fid` *(+klem kesetiaan)* | I | 24 | 14 | 10 (6–14) | 0,054 |
| `g6_kv_ls10_realignOFF` | **D** | 29 | 13 | 11 (7–15) | 0,280 |
| `b5_post` | I | 22 | 11 | 9 (6–13) | 0,306 |
| `a8_kv_innovate_guided` *(produksi)* | I | 22 | 10 | 9 (6–13) | 0,462 |
| `b14_summary` | I | 24 | 10 | 10 (6–14) | 0,548 |
| `g4_kv_ls10` | **D** | 13 | 6 | 7 (4–10) | 0,744 |
| `g3_kv_ls10_sampleT1.0` | **D** | 24 | 7 | 10 (6–13) | 0,922 |
| **`a8_kv_nodesign`** | **—** | **33** | **7** | 12 (8–16) | **0,988** |
| **`b14_text`** | I | 17 | **4** | 8 (5–11) | **0,994** |

Tiga pola terbaca langsung:

1. **Enam dari tujuh lengan teratas memakai rantai `design`.** Lengan yang
   mengungguli lantai hampir seluruhnya berantai `design`; lengan `innovate`
   berkumpul di sekitar median (p ≈ 0,3–0,6), dan yang tanpa agen hulu sama
   sekali (`nodesign`) berada di dasar (p=0,988).
2. **`latent_steps`=5 mengungguli lantai (p=0,006), `latent_steps`=10 tidak**
   (`g2_kv_ls10` p=0,718; `g4_kv_ls10` p=0,744). B1 memilih 10 demi keandalan,
   dan HASIL_GPU §2 sudah mencatat ls=5 menemukan "16 klaster dari 22 faktor" —
   kini diketahui 16 klaster itu **signifikan di atas lantai acak**. Ini
   menambah satu kandidat lagi ke §10.3(f): titik optimal `latent_steps`
   mungkin **lebih rendah** dari 10, bukan lebih tinggi.
3. **Di dalam keluarga `innovate`, yang berklem kesetiaan justru terbaik**
   (`innovate_fid` p=0,054 vs `innovate_guided` p=0,462). Klem yang di HASIL_A8
   §3.2(b) dinilai "mencekik eksplorasi" ternyata mempertahankan keragaman
   **sinyal** — konsisten dengan §11.2 (lengan itu juga satu-satunya lengan
   `innovate` tanpa kehilangan ekspresi karena anggaran waktu).

*Multiplisitas*: 24 lengan diuji, jadi pada α=0,05 sekitar 1 temuan positif
palsu diharapkan muncul secara kebetulan. Dengan koreksi Bonferroni
(0,05/24 ≈ 0,002) hanya **`a8_kv_full`** yang lolos tanpa syarat; `px_8B_v1`
(0,004) dan `g2_kv_ls5` (0,006) berada tepat di sekitarnya. Karena itu klaim
yang ditegakkan dokumen ini hanya yang pertama — dan ia ditegakkan bukan lewat
ambang, melainkan lewat **ablasi berpasangan** `full` vs `nodesign` di bawah,
yang tidak menuntut koreksi multiplisitas karena ia satu perbandingan yang
didaftarkan sendiri.

**Ini temuan terpenting sesi ini, dan ia membalik sebagian bacaan B16.**

#### Ablasi yang terkontrol sempurna: `full` vs `nodesign`

Dua baris pertama dan terakhir tabel di atas adalah pasangan yang paling
bersih yang dimiliki proyek ini. LLM sama, DSL sama, prompt sama, seed dan arah
sama, jumlah run sama — **satu-satunya perbedaan adalah ada/tidaknya agen
`design`**:

| | `full` | `nodesign` |
|---|---:|---:|
| faktor hidup | 27 | **33** (lebih banyak) |
| klaster sinyal | **21** | **7** |
| lantai acak n-tercocok | 10 (7–15) | 12 (8–16) |
| **posisi terhadap lantai** | **p<0,002 — mengungguli** | **p=0,988 — di bawah** |

`nodesign` menghasilkan **lebih banyak** faktor tetapi menumpuknya di **sepertiga
jumlah klaster**, dan mendarat di tempat yang 98,8% sampel acak berukuran sama
mengalahkannya. Menambahkan satu agen membalik lengan yang sama dari
"lebih buruk dari melempar dadu" menjadi "tak tersamai oleh 300 lemparan dadu".

Ini juga **menghapus confound palet** yang dibahas di bawah: kedua lengan punya
akses DSL yang identik, jadi selisihnya tak mungkin berasal dari tata bahasa.

**Konsekuensi yang tidak nyaman untuk B16.** Gerbang 1 Tahap 3a menyatakan
`design` tidak berkontribusi karena pengaruhnya pada \|IC\| tidak signifikan
(Welch t=0,79) — dan atas dasar itu slotnya diganti `innovate`. Tetapi \|IC\|
adalah sumbu yang, kini terbukti, **tidak membedakan apa pun**: setiap lengan
setara dengan lantai acak di sana. Keputusan diambil pada sumbu yang buta, dan
yang dihapus ternyata satu-satunya komponen yang menyebabkan keunggulan sistem
pada sumbu yang melihat.

HASIL_A8 §3.2(a) sebenarnya sudah menduga ini — "*`design` menyumbang KERAGAMAN,
bukan kuantitas*" — tetapi tanpa lantai acak, dugaan itu tak bisa dibedakan dari
kebetulan. Sekarang bisa: bukan hanya `design` menaikkan keragaman, ia adalah
**satu-satunya alasan sistem ini mengungguli pencarian acak sama sekali**.

Digabung dengan sumbu mutu (Mann-Whitney \|IC\| per-ekspresi hidup terhadap
lantai acak yang sama), muncul **disosiasi ganda** yang bersih:

| lengan | mutu sinyal (\|IC\|) | cakupan pencarian (klaster) |
|---|---|---|
| **`a8_kv_full`** (`design`) | 0,0105 — **LEBIH BURUK dari acak** (z=−2,53, p=0,011) | 21 vs 10 — **JAUH LEBIH BAIK dari acak** (p<0,002) |
| **`a8_kv_innovate_guided`** | 0,0179 — **menyamai acak** (z=−0,42, p=0,673) | 10 vs 9 — **menyamai acak** (p=0,462) |
| **`a8_kv_nodesign`** | 0,0085 — **LEBIH BURUK dari acak** (p<0,001) | 7 vs 12 — **di bawah acak** (p=0,988) |

Bacaannya: **sumbangan nyata rantai multi-agen bukan menemukan sinyal yang
KUAT, melainkan menemukan banyak sinyal yang BERBEDA.** Pada sumbu itu — dan
hanya pada sumbu itu — rantai `design` mengalahkan pencarian acak secara telak:
**nol dari 500** sampel acak berukuran sama menemukan sebanyak 21 klaster.
Selama ini sumbu itu tidak punya lantai, jadi keunggulan tersebut tak pernah
bisa diklaim; sekarang bisa.

Dan konsekuensinya untuk B16 tidak nyaman: mengganti `design` → `innovate`
menaikkan \|IC\| dari "lebih buruk dari acak" menjadi "menyamai acak", tetapi
**menghapus satu-satunya sumbu tempat sistem ini benar-benar mengungguli
pencarian tanpa teori** (p<0,002 → p=0,462). §10.2(e) mencatat penurunan
klaster 20 → 9 sebagai "sumbu yang memburuk"; dengan lantai acak di tangan,
kalimat yang tepat lebih keras dari itu: **yang hilang bukan sekadar angka,
melainkan satu-satunya keunggulan yang terbukti.**

Perlu ditegaskan supaya tidak salah dipakai: ini **tidak** berarti B16 keliru.
Ketiga kriteria Tahap 3b yang didaftarkan di muka tetap terpenuhi, dan
"cakupan pustaka DSL" (fungsi apa saja yang dipakai, 13 → 20) memang naik.
Yang ditunjukkan angka di atas adalah bahwa **cakupan pustaka bukan cakupan
sinyal**: `innovate` memakai lebih banyak jenis fungsi untuk menghasilkan
sinyal yang justru lebih saling mirip. Itu perbedaan yang sebelumnya tidak
terlihat karena kedua hal disebut "cakupan".

#### Perbandingan medium pada sumbu cakupan — peringkatnya berbalik untuk KETIGA kalinya

Sumbu ini juga bisa dipakai untuk pertanyaan asli judul skripsi. Dua trio yang
masing-masing **internal-konsisten** (satu ronde, satu konfigurasi, hanya medium
yang berbeda):

| konfigurasi | medium | hidup | klaster | acak n-tercocok | p |
|---|---|---:|---:|---:|---:|
| **G4** (rantai `design`, pra-Tahap-2) | **`text`** | 21 | **13** | 9 (6–13) | **0,052** |
| | `kv` | 13 | 6 | 7 (4–10) | 0,744 |
| | `kv_and_text` | 9 | 6 | 5 (3–8) | 0,446 |
| **sekarang** (rantai `innovate` + guided) | `kv` | 22 | 10 | 9 (6–13) | 0,462 |
| | `summary` | 24 | 10 | 10 (6–14) | 0,548 |
| | **`text`** | 17 | **4** | 8 (5–11) | **0,994 — di bawah** |

Pada konfigurasi G4, **`text` adalah medium yang paling dekat mengungguli lantai
acak** — p=0,052, tepat di ambang dan karena itu **tidak** boleh disebut
signifikan; pada konfigurasi sekarang, `text` justru medium terburuk (p=0,994)
dan `kv` memimpin. Arah peringkat medium karena itu **berbalik lagi** — sekarang
pada sumbu ketiga, setelah sebelumnya berbalik pada IC bertanda dan pada
\|IC\| (§9.4). Yang ditegakkan di sini adalah **pembalikan arahnya**, bukan
klaim bahwa `text` unggul: untuk itu p=0,052 tidak cukup.

Ini menutup §6 ("`kv` menang di SETIAP sumbu") dengan tepat: kesimpulan itu
benar **untuk konfigurasi sekarang**, dan §6 sudah menyatakan batas berlakunya —
tetapi kini terlihat bahwa pada konfigurasi lain, sumbu yang sama memberi
pemenang yang berbeda. Yang stabil bukan pemenangnya, melainkan **ketidak-
stabilan peringkatnya**.

*Peringatan konfound yang wajib*: kedua trio **tidak boleh disilangkan**. Lengan
G4 memakai prompt & gate pra-Tahap-2 (sebelum B4/B12/B15) sementara lengan
"sekarang" memakai pasca-Tahap-2 + rantai `innovate` + guided decoding. Jadi
`g4_kv` (6 klaster) dan `a8_kv_full` (21 klaster) **bukan** dua pengukuran atas
sistem yang sama meski keduanya `kv` + rantai `design`. Perbandingan yang sah
hanya di dalam masing-masing baris berkonfigurasi sama.

**Keberatan paling wajar, dan mengapa ia gugur.** Metrik klaster memakai
korelasi antar **deret IC harian**. Faktor yang lemah/berisik cenderung tidak
berkorelasi dengan apa pun, jadi ia menjadi klaster tunggal — sehingga lengan
yang menghasilkan faktor lebih BURUK bisa tampak lebih BERAGAM secara artifisial.
Ini keberatan yang harus dijawab sebelum klaim mana pun di atas dipakai.

Datanya menjawabnya sendiri:

| lengan | mean \|IC\| | klaster | faktor hidup |
|---|---:|---:|---:|
| `nodesign` | **0,0085** *(paling lemah)* | **7** *(paling sedikit)* | 33 |
| `full` | 0,0105 | **21** *(terbanyak)* | 27 |
| `innovate_guided` | **0,0179** *(paling kuat)* | 10 | 22 |

Kalau artefak derau yang bekerja, lengan **terlemah** (`nodesign`, 0,0085)
seharusnya punya klaster **terbanyak**. Ia punya paling sedikit. Hubungan
\|IC\| ↔ klaster juga **tidak monoton** ke arah mana pun (0,0085→7, 0,0105→21,
0,0179→10) — dan justru itulah yang membuat cakupan menjadi sumbu yang
independen, bukan proksi terselubung bagi kekuatan sinyal.

**Confound kedua yang wajib diperiksa, dan hasilnya.** Lantai acak dibangkitkan dari
tata bahasa yang **lebih sempit** daripada yang tersedia bagi LLM: generator
`lab/random_baseline.py` memakai **29 dari 55** fungsi (REGBETA/REGRESI sengaja
dikeluarkan karena biayanya, lihat docstring-nya). Palet yang lebih sempit bisa
menghasilkan sinyal yang lebih seragam, sehingga lantainya **terlalu rendah**
dan p<0,002 pada `full` jadi tampak lebih mengesankan daripada seharusnya.

Diperiksa: `a8_kv_full` menghasilkan 21 klaster itu dengan **hanya 13 dari 55
fungsi** — lebih sedikit daripada 29 fungsi yang tersedia bagi generator acak.
Jadi keunggulannya **tidak** datang dari palet yang lebih luas; ia datang dari
cara fungsi-fungsi itu disusun. Confound-nya nyata dan harus disebut, tetapi
pada kasus ini ia berjalan **melawan** LLM, bukan mendukungnya — sehingga
kesimpulannya bertahan. (Untuk lengan `innovate_guided` yang memakai 20 fungsi,
argumen ini lebih lemah dan tak perlu dipakai: lengan itu memang hanya menyamai
lantai.)

*Konvensi nilai-p*: `p` = fraksi dari 500 ulangan acak yang klasternya ≥ lengan
LLM. Untuk `a8_kv_full` fraksinya **0/300**, yang tidak boleh ditulis "p = 0"
— resolusi terkecil yang bisa dinyatakan 500 ulangan adalah 1/500, jadi
dilaporkan **p < 0,002**. Ini uji satu-sisi berbasis permutasi, bukan uji
parametrik; tidak ada asumsi distribusi yang dipakai.

*Batas berlaku*: tiap lengan menyumbang **satu** hitungan klaster (statistik
tingkat-korpus), jadi bootstrap di atas memberi sebaran nol tetapi bukan
sebaran sampling milik lengan itu sendiri. Untuk `a8_kv_full`, jarak ke sebaran
nol sangat besar (21 vs p95=15) sehingga kesimpulannya kokoh; untuk lengan yang
mendarat di sekitar median, "menyamai" adalah bacaan yang tepat, bukan "setara
secara terbukti". Lengan lain menyusul seiring `bash lab/rescore_all.sh`
selesai; baris untuk lengan yang parquet-nya belum ada sengaja tidak
ditampilkan karena skrip hanya akan melihat sebagian ekspresinya.

### 11.4 Premis fitness §S3 diuji — didukung, tetapi dengan catatan yang menohok

§10.1(a) mengusulkan mengganti fitness dari IC bertanda ke \|IC\| dengan tanda
ditetapkan di luar jendela evaluasi. Premisnya: tanda IC stabil di luar jendela
seleksi. AUDIT §2.6 sudah mengujinya dan menemukan 100% — tetapi pada **12
faktor TERKUAT** dari batch lama, yaitu kelompok yang paling mungkin stabil
karena alasan sepele. `lab/sign_persistence.py` mengulangnya pada korpus
pasca-perbaikan (423 ekspresi hidup) dengan kelompok **berlapis** kuat/tengah/
lemah, **dan** menjalankan lantai acak pada sumbu yang sama.

Jendela seleksi 2021 → holdout 2022-01-01…2025-12-26, n=24 per sumber:

| | tanda bertahan | Spearman seleksi↔holdout | mean \|IC\| seleksi → holdout |
|---|---:|---:|---:|
| **LLM** (korpus front-end) | **23/24 = 95,8%** | **+0,949** | 0,0198 → 0,0210 |
| **ACAK** (lantai) | **20/24 = 83,3%** | **+0,935** | 0,0194 → 0,0228 |

Per lapis kekuatan:

| lapis | mean \|IC\| seleksi | LLM tanda bertahan | ACAK tanda bertahan |
|---|---:|---:|---:|
| kuat | 0,0496 / 0,0460 | **8/8** | **8/8** |
| tengah | 0,0097 / 0,0119 | **8/8** | **8/8** |
| lemah | 0,0002 / 0,0003 | 7/8 | 4/8 |

**Dua kesimpulan, dan yang kedua lebih penting.**

**(a) Premis §S3 didukung, dan lebih kuat dari yang AUDIT klaim.** Untuk faktor
yang benar-benar punya sinyal — lapis kuat dan tengah — tanda bertahan
**16/16 di kedua sumber**. Ketidaksepakatan seluruhnya berada di lapis lemah,
tempat \|IC\| ≈ 0,0002, yaitu faktor yang tandanya memang tak bermakna. Jadi
mengganti fitness ke \|IC\| dengan tanda dari jendela latih **tidak** berisiko
membalik tanda faktor yang salah: faktor yang tandanya labil justru faktor yang
akan ditolak ambang mana pun. Ditambah \|IC\| holdout yang **naik**, bukan
meluruh (0,0198 → 0,0210), rekomendasi §10.1(a) berdiri di atas dasar yang jauh
lebih kokoh daripada n=12 di AUDIT.

**(b) Tetapi lantai acak nyaris sama stabilnya** — 83,3% vs 95,8%, Spearman
0,935 vs 0,949, dan **8/8 vs 8/8 pada lapis kuat**. Artinya kestabilan tanda
adalah **sifat data**, bukan prestasi sistem. Ini pola yang sama yang sudah
muncul pada mutu sinyal (§AUDIT 2.5) dan pada cakupan (§11.3), kini pada sumbu
ketiga: **setiap kali sebuah keunggulan sistem diberi lantai acak, lantainya
menyamai.** Itu bukan tiga kegagalan terpisah; itu satu temuan yang muncul tiga
kali.

### 11.5 Cara mereproduksi §11

```bash
cd quantalatent
export PYTHONPATH=backend

# 11.1 — regenerasi deret IC + verifikasi silang thd angka mesin GPU.
#        Satu proses per tag (proses panjang tumbuh & kena OOM di mesin 8 GB);
#        aman diulang, melanjutkan dari cache di lab/out/.rescore_cache.json
bash lab/rescore_all.sh
LAB_MAX_WORKERS=8 bash lab/rescore_all.sh      # mesin ber-RAM lega

# satu tag saja
.venv/bin/python lab/rescore_all.py --tags b14_summary
.venv/bin/python lab/rescore_all.py --dry-run  # verifikasi tanpa menulis

# 11.2 — pulihkan faktor yang hilang karena anggaran waktu
LAB_MAX_WORKERS=8 .venv/bin/python lab/rescore_timeouts.py --budget 1800
LAB_MAX_WORKERS=8 .venv/bin/python lab/rescore_timeouts.py --budget 1800 --apply

# 11.3 — lantai acak untuk sumbu cakupan
.venv/bin/python lab/random_clusters.py --score-only    # bagian mahal, dicache
.venv/bin/python lab/random_clusters.py --boot 300

# premis fitness §10.1(a): apakah tanda IC bertahan ke holdout 2022-2025?
.venv/bin/python lab/sign_persistence.py --top 24

# analisis rutin setelah parquet ada kembali
.venv/bin/python lab/analyze_gpu.py --glob 'frontend_a8_*.json' --by _tag --clusters
```

---

## 12. Yang belum dikerjakan sesudah dokumen ini

**Butuh GPU** — tidak bisa dikerjakan di mesin lokal:

1. **Tahap 5** (`mutation`, `crossover`, `feedback`) — satu-satunya bagian
   RENCANA_PERBAIKAN yang belum tersentuh sejak awal proyek. Pertanyaan intinya
   sudah punya alat: A9 (`lab/channel_capacity.py`) bisa menguji apakah
   `guidance_kv` benar-benar memindahkan arah dari Director ke front-end.
2. **`latent_steps` optimal antara 10 dan 40** — §10.3(f).
3. **`innovate_lean`** — §10.3(h).
4. **Ronde ulang A8 pada himpunan yang setara**, setelah §11.2 diselesaikan.

**Bisa CPU, sengaja TIDAK dikerjakan sepihak di sesi ini:**

5. **Penerapan S2/S3** (fitness \|IC\| + correlation store simetris, §10.1a–c).
   Alasannya bukan teknis melainkan metodologis: memverifikasi efeknya menuntut
   satu putaran evolusi penuh — yaitu GPU — dan RENCANA §D sendiri melarang
   mengubah beberapa lapisan sekaligus tanpa bisa mengukurnya
   ("nilai skripsi ini ada pada rantai sebab yang bisa dipertahankan"). Yang
   bisa dikerjakan tanpa GPU adalah **menguji premisnya**, dan itulah
   `lab/sign_persistence.py`. Menerapkan perubahan fitness sebelum premisnya
   diuji akan mengulang persis kesalahan yang §S3 keluhkan.
6. **Intervensi untuk klaster sinyal 20 → 9** (§10.2e) — menunggu §11.3, karena
   tanpa lantai acak untuk sumbu cakupan, "9 klaster" belum bisa dinilai buruk
   atau wajar.

---
