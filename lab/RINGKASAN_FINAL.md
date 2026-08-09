# Ringkasan Final — Status QuantaLatent

> **Dokumen ini menggantikan `KESIMPULAN.md` sebagai rujukan baca-cepat.**
> Setiap klaim di bawah adalah versi FINAL — nilai yang bertahan setelah
> seluruh koreksi dan skoring-ulang — ditulis sekali, tanpa riwayat
> bolak-balik "dulu dikira X, ternyata Y". `KESIMPULAN.md` tetap ada sebagai
> jejak audit lengkap (kronologi, derivasi tiap angka, reproduksi); dirujuk
> di sini sebagai `K§n`. Sumber angka: `HASIL_A8.md` (§3), `HASIL_GPU.md`
> (G4), `AUDIT_KRITIS.md` (§2.5), `KESIMPULAN.md §11` (sesi CPU 2026-08-08),
> `lab/out/*.json`.
>
> **Cakupan**: Tahap 1-4 `RENCANA_PERBAIKAN.md` + sesi CPU 2026-08-08.
> Backbone Qwen3-8B, A40 46 GB kecuali disebut lain. Tahap 5
> (`mutation`/`crossover`/`feedback`) belum pernah diuji — lihat §7.
>
> **Catatan presisi**: beberapa angka "hidup"/"\|IC\|" berbeda ±1 unit atau
> ±0,0001–0,001 antar tabel sumber, karena (a) sesi CPU memulihkan sebagian
> ekspresi yang sebelumnya kehabisan anggaran waktu skoring (K§11.1–11.2),
> dan (b) dua skrip analisis menghitung ekspresi mati/duplikat sedikit
> berbeda. **Tak satu pun kasus di mana IC yang dihitung ulang di mesin
> berbeda menghasilkan angka yang benar-benar berbeda** (K§11.1) — seluruh
> selisih berasal dari cakupan korpus, bukan galat hitung. Tabel di bawah
> memakai angka **paling akhir** yang tersedia; selisih kecil itu tidak
> mengubah urutan atau kesimpulan lengan mana pun.

---

## 1. Delapan fakta inti

1. **Tidak ada satu pun konfigurasi yang diuji (7 rantai agen × 4 medium
   komunikasi) yang mengalahkan pencarian acak pada mutu sinyal (\|IC\|).**
   Terbaik = menyamai lantai acak (konfigurasi produksi sekarang).
2. **Satu konfigurasi mengalahkan pencarian acak secara telak** — bukan pada
   mutu, tapi pada **cakupan/keragaman sinyal**: rantai lama
   `proposal→design→construct` menghasilkan 21 klaster sinyal berbeda vs
   lantai acak 10 [7–15], p<0,002. Ini satu-satunya hasil dari 24 lengan yang
   diuji yang lolos koreksi Bonferroni.
3. **Agen `design` — yang menghasilkan keunggulan #2 — sudah dihapus dari
   produksi**, diganti agen `innovate`, berdasarkan kriteria \|IC\| (Welch
   t=0,79, dianggap "tidak berpengaruh"). \|IC\| terbukti belakangan adalah
   sumbu yang tidak membedakan lengan mana pun dari lantai acak, sehingga
   keputusan itu diambil pada sumbu yang buta terhadap satu-satunya
   keunggulan sistem yang terbukti nyata.
4. **Konfigurasi produksi sekarang** (`proposal→innovate→construct`, guided
   decoding di `construct`, medium `kv`) adalah yang **tercepat dan termurah**
   di antara lengan yang reliabel (6/6 run produktif), dan satu-satunya yang
   **menyamai** lantai acak pada mutu sinyal — tetapi **terburuk** di antara
   lengan-lengan reliabel itu pada cakupan sinyal (10 klaster, vs 21 milik
   rantai `design`).
5. **Lengan tercepat/termurah secara mutlak** (termasuk lengan tak-reliabel)
   = `nodesign` (`proposal→construct` langsung, tanpa agen hulu): 5,8
   detik & 902 token per faktor diterima — tetapi juga **terburuk** pada
   mutu MAUPUN cakupan sekaligus.
6. **Lengan termahal** = `full_guided` (guided decoding dipasang di rantai
   `design`): 50,1 detik/faktor — 7× lengan tercepat — dengan laju lolos
   gate terendah kedua (62%). Guided decoding menolong rantai `innovate`
   tetapi merugikan rantai `design`; efeknya bukan universal.
7. **Peringkat medium komunikasi (`kv`/`text`/`kv_and_text`/`summary`)
   terbalik total tergantung rantai agen yang menyertainya** (§4). Tidak ada
   "medium terbaik" yang berlaku umum — hanya "medium terbaik untuk
   konfigurasi X".
8. **Kombinasi yang secara teori paling menjanjikan — menggabungkan `design`
   (pemilik keunggulan cakupan) dengan `innovate` (pemilik keunggulan
   mutu/biaya) dalam satu rantai — belum pernah dijalankan.**

---

## 2. Tiga lantai pembanding (dipakai di seluruh dokumen ini)

Semua diambil dari kolam yang sama: 300 ekspresi disusun **acak** dari DSL
yang sama yang tersedia bagi LLM (fungsi/parameter dipilih random, bukan
dari hipotesis), 271 di antaranya hidup (tidak NaN/konstan). Generator acak
sengaja memakai **29 dari 55 fungsi** (REGBETA/REGRESI dikeluarkan karena
mahal) — palet yang lebih SEMPIT daripada yang tersedia bagi LLM, sehingga
perbandingan ini konservatif *terhadap* lantai, bukan menguntungkannya.

| lantai | definisi | nilai |
|---|---|---:|
| **Mutu** | mean \|IC\| dari 271 ekspresi acak hidup | **0,0170** |
| **Cakupan** | jumlah klaster sinyal (\|Spearman deret IC harian\|>0,7) dari sampel-ulang sebanyak k ekspresi acak (k = jumlah faktor hidup lengan yang diuji), bootstrap 500× | bergantung k (lihat §3) |
| **Kestabilan tanda** | % dari 24 ekspresi acak yang tanda IC-nya bertahan dari jendela seleksi (2021) ke holdout sejati (2022–2025) | **83,3%** (Spearman seleksi↔holdout +0,935) |

Tiga lantai ini independen tapi berulang polanya: **setiap kali sebuah
keunggulan sistem diuji terhadap lantai acak, hasilnya menyamai** — kecuali
satu sumbu (cakupan, lengan `design`). Itu bukan tiga kegagalan terpisah;
itu satu temuan yang muncul tiga kali, dan satu pengecualian yang karena itu
patut diperhatikan serius.

**Kenapa dua uji statistik dipakai berbeda**: n=6 run/lengan cukup untuk
sumbu bervarians rendah (klaster per-lengan, laju lolos gate, biaya) tapi
tidak bertenaga untuk \|IC\| yang variansnya tinggi. Karena itu uji mutu
memakai satuan **per-ekspresi hidup** (n = puluhan, jauh lebih bertenaga),
sedangkan uji cakupan memakai satuan **per-lengan** (n=1 angka klaster per
lengan, dibandingkan lewat bootstrap 500 lengan-acak-tiruan, bukan lewat
n=6).

---

## 3. Kombinasi rantai agen — tabel utama

Konfigurasi dasar (kecuali disebut lain): `comm_mode=kv`, `latent_steps=10`,
`step_mode=gumbel`, gate Tahap 2 aktif (B12+B15), prompt Tahap 2 (B4), 6 run
per lengan (2 arah × 3 seed). Sumber: `HASIL_A8.md §3, §3.1, §4b`;
`KESIMPULAN.md §11.3` (klaster, angka final).

| rantai | reliabilitas | lolos gate | hidup | **mutu** (\|IC\| per-ekspresi vs lantai 0,0170) | **cakupan** (klaster vs lantai n-tercocok) | pustaka DSL | detik/faktor | token/faktor |
|---|:-:|---:|---:|---|---|---:|---:|---:|
| `full` (proposal→design→construct) | 6/6 | 83% | 27 | 0,0105 — **di bawah** lantai (p=0,011) | **21** vs 10 [7–15] — **di atas** lantai (**p<0,002**) ★ | 13 | 7,0 | 1095 |
| `nodesign` (proposal→construct) | 6/6 | 89% | 33 | 0,0085 — **di bawah** lantai (p<0,001) | 7 vs 12 [8–16] — **di bawah** lantai (p=0,988) | 12 | **5,8** ★ | **902** ★ |
| `direct` (construct saja) | 3/6 ⚠ | 89% | 16 | 0,0106 — batas lantai (p=0,054) | 7 (belum diuji formal vs lantai) | 7 | 20,6 | 2265 |
| `innovate` (klem kesetiaan OFF, tanpa guided) | 4/6 ⚠ | 87% | 14 | 0,0164–0,0167 — **menyamai** lantai (p=0,32) | 8 (belum diuji formal vs lantai) | **23** ★ | 9,5 | 1482 |
| `innovate_fid` (klem kesetiaan ON) | 6/6 | 70% ⚠ | 24 | 0,0157 — **menyamai** lantai (p=0,131) | 14 vs 10 [6–14] — mendekati lantai (p=0,054) | 16 | 7,4 | 1272 |
| `full_guided` (design + guided decoding) | 5/6 | 62% ⚠ | 17 | 0,0119 — **di bawah** lantai (p=0,064) | 12 vs 8 [4–11] — di atas lantai (p=0,032†) | 15 | **50,1** ⚠ | 2354 |
| **`innovate_guided` = PRODUKSI** (guided hanya di `construct`) | 6/6 | **88%** | 22 | **0,0178–0,0182 — menyamai lantai** (p=0,63–0,67) | 10 vs 9 [6–13] — **menyamai** lantai (p=0,462) | 22 | 15,8 | **997** |

★ = terbaik di kolom itu. ⚠ = terburuk atau tak-reliabel.
† `full_guided` p=0,032 tidak lolos koreksi Bonferroni (ambang 0,05/24≈0,002
untuk 24 lengan yang diuji); hanya `full` yang lolos tanpa syarat.

### Ablasi paling bersih di seluruh proyek: `full` vs `nodesign`

LLM sama, DSL sama, prompt sama, seed & arah sama — **satu-satunya beda
adalah ada/tidaknya agen `design`**. `nodesign` menghasilkan lebih *banyak*
faktor hidup (33 vs 27) tapi menumpuknya di **sepertiga jumlah klaster** (7
vs 21), dan mendarat di posisi yang 98,8% sampel acak berukuran sama
mengalahkannya. Menambahkan satu agen membalik lengan yang sama dari "lebih
buruk daripada melempar dadu" menjadi "tak tersamai oleh 300 lemparan
dadu". Ini satu-satunya klaim kausal bersih (bukan korelasional) di seluruh
dokumen ini.

### Kenapa cakupan bukan artefak derau

Kalau klaster tinggi hanya berarti "sinyal lemah tidak berkorelasi dengan
apa pun", lengan ber-\|IC\| terlemah seharusnya berklaster terbanyak. Datanya
berkata sebaliknya: `nodesign` (\|IC\| 0,0085, terlemah) punya klaster
**paling sedikit** (7); `full` (\|IC\| 0,0105, tengah) punya klaster
**terbanyak** (21); `innovate_guided` (\|IC\| 0,0178, terkuat) punya klaster
menengah (10). Hubungan \|IC\|↔klaster tidak monoton ke arah mana pun —
cakupan adalah sumbu independen, bukan proksi mutu yang tersembunyi.

---

## 4. Kombinasi medium komunikasi (`kv` / `text` / `kv_and_text` / `summary`)

Dua trio, masing-masing **internal-konsisten** (satu ronde, satu
konfigurasi rantai, hanya medium yang berbeda). **Jangan disilangkan** —
trio A memakai prompt/gate pra-Tahap-2, trio B memakai prompt/gate
pasca-Tahap-2 + rantai berbeda.

**Trio A — rantai `design` (konfigurasi lama, pra-Tahap 2), `latent_steps=10`:**

| medium | reliabilitas | lolos gate | hidup | \|IC\|/run | klaster vs lantai n-tercocok | detik/run |
|---|:-:|---:|---:|---:|---|---:|
| `text` | 6/6 | **83%** | 21 | 0,0084 | 13 vs 9 [6–13] — nyaris lantai, **tak signifikan** (p=0,052) | 67 |
| `kv` | 5/6 | 54% | 13 | 0,0152 | 6 vs 7 [4–10] — di bawah lantai (p=0,744) | 29 |
| `kv_and_text` | 4/6 | 50% | 9 | 0,0158 | 6 vs 5 [3–8] (p=0,446) | 99 |

→ Pada konfigurasi ini, `text` reliabilitasnya tertinggi dan satu-satunya
yang mendekati lantai pada cakupan; `kv` kalah di reliabilitas **dan**
cakupan.

**Trio B — rantai `innovate` + guided decoding (konfigurasi SEKARANG), `latent_steps=10`:**

| medium | reliabilitas | lolos gate | hidup | \|IC\|/run | klaster vs lantai n-tercocok | detik/faktor | token/faktor | cacat semantik |
|---|:-:|---:|---:|---:|---|---:|---:|---:|
| **`kv` = PRODUKSI** | 6/6 | **88%** | 22 | **0,0182** | 10 vs 9 [6–13] (p=0,462) | **15,8** | **997** | 0% |
| `summary` | 6/6 | 69% | 24 | 0,0127 | 10 vs 10 [6–14] (p=0,548) | 24,3 | 1899 | 11% |
| `text` | 6/6 | 44% ⚠ | 17 | 0,0180 | 4 vs 8 [5–11] — **di bawah lantai** (p=0,994) ⚠ | 32,8 ⚠ | 2602 ⚠ | 0% |

→ Pada konfigurasi ini, `kv` menang atau menyamai di **setiap** sumbu yang
diukur.

**Yang bertahan dari kedua trio**: peringkat medium **terbalik total**.
`text` nyaris terbaik pada cakupan di trio A (p=0,052) dan terburuk di trio
B (p=0,994); `kv` di bawah lantai pada trio A dan menyamainya di trio B.
Kesimpulan yang bertahan bukan "medium X selalu menang", melainkan
**peringkat medium bergantung pada rantai agen yang menyertainya** —
berlaku hanya untuk kombinasi (rantai, medium) yang benar-benar diuji
bersama, bukan untuk "KV vs teks" sebagai pernyataan umum.

---

## 5. Kombinasi mana yang "terbaik"?

Tidak ada satu lengan yang menang di semua sumbu — itu bukan kontradiksi,
itu struktur datanya. Ringkasan pemenang per sumbu, dari tabel §3:

| sumbu | pemenang | catatan |
|---|---|---|
| Cakupan/keragaman sinyal | `full` (rantai `design`) | satu-satunya yang mengalahkan lantai acak (p<0,002); **dihapus dari produksi** |
| Mutu sinyal per-ekspresi | `innovate_guided` (produksi) | tidak mengalahkan lantai, hanya menyamainya — tapi ini yang TERBAIK dari 7 lengan |
| Biaya (waktu & token) di antara lengan reliabel | `nodesign` | tapi lengan ini terburuk di kedua sumbu lain |
| Reliabilitas (6/6) | `full`, `nodesign`, `innovate_fid`, `innovate_guided` | empat lengan seri |
| Laju lolos gate | `nodesign`/`direct` (89%) | `innovate_guided` dekat di belakang (88%) |
| Pustaka DSL terluas | `innovate` (23 fungsi) | tapi lengan ini sendiri tak reliabel (4/6) |

**Kombinasi yang belum pernah diuji, dan paling langsung diisyaratkan oleh
tabel di atas**: `proposal → design → innovate → construct` — menggabungkan
pemilik keunggulan cakupan dengan pemilik keunggulan mutu/biaya dalam satu
rantai. Hipotesisnya sederhana dan bisa gagal: kalau kontribusi keduanya
aditif, rantai gabungan mengalahkan lantai acak pada KEDUA sumbu sekaligus
— sesuatu yang belum dicapai lengan mana pun sejauh ini. Biayanya satu
ronde GPU (6 run).

---

## 6. Metrik: kenapa paper QuantaAlpha memakai IC, dan apa alternatifnya

### 6.1 Kenapa IC

Paper mendefinisikan tujuan alpha mining sebagai (Eq. 1):
$f^* = \arg\max_f \mathcal{L}(f(X), y) - \lambda\mathcal{R}(f)$, dan
Lampiran A.1 mengoperasionalkan $\mathcal{L}$ sebagai korelasi Pearson
antara nilai faktor dan return periode berikutnya — yaitu IC. Empat alasan
IC dipakai sebagai metrik utama, bukan pilihan sembarang:

1. **IC ada di dalam rumusan formal tujuannya sendiri.** Bukan metrik
   evaluasi yang ditempel belakangan — ia ADALAH $\mathcal{L}$ pada Eq. 1–2,
   sehingga reward trajectory (Eq. 2) yang menggerakkan mutation/crossover
   memakainya langsung.
2. **IC menilai satu faktor SENDIRIAN**, sebelum ada konstruksi portofolio
   (bobot, topk, biaya transaksi). Ini penting karena satu trajectory di
   paper menghasilkan satu faktor ($h\to f_\tau$) — metrik strategi (ARR,
   IR, MDD, CR) baru bisa dihitung SETELAH faktor digabung + aturan trading
   + biaya diterapkan, jadi tak bisa jadi reward per-kandidat saat pencarian
   berjalan.
3. **Landasan teoretis established**: Grinold's Fundamental Law of Active
   Management, $IR \approx IC \times \sqrt{\text{breadth}}$ — IC yang lebih
   tinggi punya arti langsung terhadap potensi nilai strategi pada skala,
   lepas dari pilihan implementasi portofolio.
4. **Sebanding dengan literatur pembanding.** Seluruh baseline di Tabel 1
   paper (Alpha158/360, LSTM/GRU/Transformer/TRA, RD-Agent, AlphaAgent)
   sudah melaporkan IC sebagai angka utama — memakai IC menjaga
   perbandingan tetap apples-to-apples.

### 6.2 Metrik lain yang SUDAH ada di paper sebagai pembanding

Paper sendiri tidak berhenti di IC — Tabel 1 melaporkan delapan kolom
sekaligus:

| metrik | mengukur apa | kenapa perlu selain IC |
|---|---|---|
| Rank IC (Spearman) | korelasi berbasis peringkat, bukan nilai mentah | tahan outlier & hubungan non-linear; return berekor tebal (Fama 1965, dikutip paper §1) |
| ICIR / Rank ICIR | IC dibagi deviasi standarnya sendiri lintas waktu | mengukur **konsistensi**, bukan cuma rata-rata — faktor ber-IC tinggi tapi liar tak sestabil faktor ber-IC sedang yang stabil |
| ARR, IR, MDD, CR | performa portofolio setelah kombinasi faktor + aturan trading + biaya | menjawab "apakah produk akhirnya benar-benar menghasilkan uang", pertanyaan yang berbeda dari "apakah satu faktor ini informatif" |

### 6.3 Metrik TAMBAHAN yang terbukti perlu di proyek ini, dan tak ada di paper

Ini bagian yang paling relevan dari temuan proyek sendiri: IC (dan Rank IC,
karena sama-sama metrik per-faktor) **terbukti buta terhadap satu sumbu
yang justru satu-satunya tempat sistem ini mengalahkan pencarian acak** —
lihat §2 dan §3. Tiga metrik yang mengisi celah itu:

1. **Cakupan/keragaman sinyal (klaster IC, §3).** IC dihitung per-faktor
   tanpa pernah membandingkan satu faktor dengan faktor lain — jadi dua
   lengan bisa ber-IC identik sementara satu menemukan 21 sinyal yang
   saling berbeda dan yang lain menemukan 4 salinan dari sinyal yang sama.
   Keputusan B16 (§1 butir 3) memakai IC sebagai satu-satunya kriteria dan
   karena itu tak bisa melihat perbedaan ini sama sekali. Paper sendiri
   mengisyaratkan kepedulian yang sama secara kualitatif — kontrol
   redundansi Eq. 5–6 dan ablasi "Redundancy Control" di §5.3 — tapi tidak
   mempublikasikan statistik cakupan berdiri sendiri dengan lantai
   pembandingnya sendiri seperti yang dilakukan di sini.
2. **Kestabilan tanda ke holdout sejati (§2, lantai ketiga).** Mirip
   semangat ICIR (stabilitas), tapi diuji lintas batas out-of-sample sejati,
   bukan varians dalam-sampel.
3. **Biaya komputasi LLM (detik & token per faktor diterima).** Paper
   sebagai algoritma pencarian tidak perlu melaporkan biaya inferensi LLM
   sama sekali — tak ada angka semacam itu di paper manapun yang dirujuk.
   Tapi untuk sistem yang benar-benar memilih antar desain rantai agen,
   biaya adalah sumbu nyata: `full_guided` di §3 berbiaya 7× lipat
   `nodesign` untuk laju lolos gate yang lebih RENDAH — sebuah konfigurasi
   yang secara ketat kalah, dan IC saja tidak akan pernah mengungkapnya
   karena IC tidak mengukur berapa mahal faktor itu diperoleh. Ini sumbu
   yang melekat pada alpha mining berbasis agen-LLM secara spesifik, tidak
   ada di sistem pencarian klasik (non-LLM) atau di paper yang dirujuk.

---

## 7. Di luar cakupan dokumen ini

- **Tahap 5** (`mutation`, `crossover`, `feedback`) — belum pernah diuji
  sejak awal proyek. Semua angka di atas adalah batas atas untuk sistem
  evolusioner penuh (karena evolusi memanggil ulang front-end yang sama),
  bukan pengukuran langsung atasnya.
- **Satu backbone (Qwen3-8B), satu pasar (CSI300), satu periode.** Urutan
  di §3–§4 berlaku untuk konfigurasi ini; tak ada klaim di dokumen ini yang
  diperluas ke "KV vs teks secara umum" atau ke backbone lain.
- Kronologi lengkap, derivasi tiap angka, dan cara mereproduksi ada di
  `KESIMPULAN.md` (§9.1–§9.5 untuk pemetaan ke bab skripsi, §11.5 untuk
  perintah reproduksi CPU).
