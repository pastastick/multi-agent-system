# Kesimpulan akhir — apa yang berubah di QuantaLatent, dan apa efeknya

> Cakupan: **seluruh `lab/RENCANA_PERBAIKAN.md` KECUALI Tahap 5** (agen
> `mutation`/`crossover`/`feedback`, yang memang belum pernah diuji sejak awal
> proyek). Backbone **Qwen3-8B**, A40 46 GB, branch `exp/rencana-perbaikan`,
> dikerjakan 2026-08-07.
>
> Dokumen sumber per tahap: `lab/HASIL_GPU.md` (G1–G7, baseline),
> `lab/HASIL_A8.md` (ablasi agen), `lab/HASIL_TAHAP4.md` (B6/B7/A9/B10),
> dan §3–§5 di bawah untuk B5/A10/B14.

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

**Mode mana yang optimal?** Lihat §5. Jawaban singkatnya: **tergantung sumbu,
dan itu sendiri temuannya** — tidak ada mode yang menang di semua sumbu.

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
| **B14** | medium baru `comm_mode="summary"` | `pipeline.py`, `settings.py` | §5 — **tersedia, bukan default** |

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

*(diisi setelah run selesai — lihat `lab/out/direction_sensitivity_a10.json`)*

---

## 6. B14 — medium `summary`: mana medium yang optimal?

*(diisi setelah run selesai)*

---

## 7. Batas berlaku SELURUH kesimpulan ini

Empat batasan yang harus ikut disebut di sidang, karena tanpanya angka-angka di
atas terdengar lebih kuat daripada yang sebenarnya.

1. **n = 6 run per lengan.** Terlalu kecil untuk uji beda pada |IC|. Sumbu yang
   benar-benar bisa diputuskan pada n ini adalah yang variansnya rendah dan
   efeknya besar: cakupan pustaka, laju lolos gate, laju run produktif, dan
   biaya per faktor. Setiap kali |IC| dipakai untuk memutuskan, itu disebutkan
   sebagai sumbu yang **tidak** membedakan.
2. **Satu backbone, satu pasar, satu periode.** Semua pada Qwen3-8B dan
   CSI300. HASIL_GPU §4 sudah menunjukkan peringkat medium **berbalik** ketika
   backbone / metrik / `latent_steps` berubah — jadi peringkat apa pun di sini
   berlaku untuk konfigurasi ini, bukan untuk "KV vs teks" secara umum.
3. **Tahap 5 belum tersentuh.** `mutation`, `crossover`, `feedback`, dan seluruh
   siklus evolusi belum pernah diuji. Yang berlaku untuk front-end adalah
   **batas atas** bagi sistem evolusioner (karena evolusi memanggil ulang
   front-end yang sama), tetapi apakah `guidance_kv` benar-benar memindahkan
   arah dari Director ke front-end **belum diukur sama sekali**.
4. **Lantai acak belum terlampaui.** Tidak satu pun lengan mengungguli mean
   |IC| ekspresi acak (0,0170) secara signifikan. Semua klaim "membaik" di
   dokumen ini adalah membaik **relatif terhadap konfigurasi sebelumnya**, bukan
   membaik terhadap pencarian tanpa teori.

---
