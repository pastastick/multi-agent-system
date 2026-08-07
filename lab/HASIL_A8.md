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

## 2. Hasil — cakupan struktural

*(diisi setelah skoring; lihat §3 untuk mutu sinyal)*

---

## 3. Hasil — mutu sinyal & keandalan

*(diisi setelah skoring)*

---

## 4. Keputusan

*(diisi setelah aturan §Tahap 3a diterapkan lewat `lab/decide_a8.py`)*
