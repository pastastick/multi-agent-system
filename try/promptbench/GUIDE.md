# Prompt-Bench — Panduan Eksperimen (handoff untuk sesi Claude berikutnya)

> **Baca ini dulu sebelum mengerjakan apa pun di folder `try/promptbench/`.**
> Dokumen ini berisi tujuan, keputusan yang sudah dikunci, peta infrastruktur
> yang HARUS di-reuse, desain fase, dan slot untuk fase-fase baru yang muncul
> setelah kita punya output LLM nyata dari runpod.

Branch: `experiment/prompt-bench` (dicabang dari `newEvol` @ `218c665`).
Repo nyata = submodule `quantalatent/` (parent repo hanya menyimpan pointer).
Gaya komunikasi user: penjelasan Indonesia, depth-first, istilah teknis Inggris.

---

## 1. TUJUAN (kenapa eksperimen ini ada)

Model 4B (Qwen3-4B) **sangat sensitif terhadap prompt**. Perubahan kecil bisa
membuat output hipotesis+ekspresi jadi stabil & siap-backtest, atau malah
collapse. Pernah ADA periode di mana output stabil (lihat artefak lama di
`try/outputs/{multi_agent_kv, pair_construct_coder, pair_propose_construct}`) —
monoton tapi sesuai format, dan KV-cache tidak collapse meski pakai
`latent_steps` + `use_realign`. Prompt yang menghasilkan itu mungkin sudah
terhapus tapi masih bisa diambil dari commit/branch lama.

**Sasaran akhir:** menemukan prompt + setting paling optimal untuk SETIAP agent
sehingga output (terutama **hypothesis** dan **expression**) **stabil,
siap-backtest, dan bervariasi sesuai kebutuhan**.

**Mengapa penting (filosofi "minimalkan ruang error"):** kalau prompt sudah
terbukti stabil menghasilkan faktor bagus secara terisolasi, lalu full-framework
tetap jelek → berarti masalahnya di **workflow**, bukan prompt. Kita mempersempit
ruang pencarian bug.

---

## 2. KEPUTUSAN YANG SUDAH DIKUNCI (jangan diubah tanpa tanya user)

1. **Branch**: `experiment/prompt-bench` dari `newEvol`. Semua artefak & varian
   prompt hidup di sini, tidak mengotori `newEvol`.
2. **Folder baru**: `try/promptbench/` — **reuse** infra `try/` yang sudah ada
   (lihat §4). Jangan reinvent backend wrapper.
3. **Penilaian "terbaik" = AUTO-SCORE + artefak**. Harness memberi skor otomatis
   (parse-rate, regulator-gate pass-rate, variety operator-family, stabilitas
   antar-repetisi) dan me-ranking, TAPI semua output teks tetap disimpan supaya
   user bisa override ranking secara manual.
4. **Mulai dari CORE 4 agent** (proposal → construct → consistency → judger),
   validasi harness+scorer+paralelisme, baru meluas.
5. **Eksperimen ini BOLEH mengubah workflow.** Agent yang terbukti tidak
   berkontribusi signifikan akan **dibuang**; kita boleh **menambah agent/loop
   baru** bila diperlukan (Phase C). Jadi ini ablation arsitektur, bukan sekadar
   tuning teks.
6. **THINKING MODE DIABAIKAN.** Alasan user (berdasar paper LatentMAS): latent
   thoughts via latent-alignment sudah meng-cover & melampaui distribusi token
   embedding TextMAS — "latent thoughts not only capture the valid semantics of
   their corresponding text responses but also encode richer and more expressive
   representations inside." Karena kita sudah pakai latent alignment, thinking
   mode tidak ditambahkan sebagai axis. Semua run: `enable_thinking=False`.
7. **Grid `latent_steps` = {0, 10, 20, 40, 60, 80}.** (Paper LatentMAS optimal
   ≈40–80 untuk Qwen3-14B; kita sweep dari 0 sampai 80 untuk 4B.) Ingat:
   **`latent_steps>0 ⇒ `use_realign=True`** (catatan tegas user).
8. **Temperature**: karena thinking dibuang, decode teks pakai **temp 0.7**
   (non-thinking). Agent `kv_only` punya `temperature: null` (dikendalikan
   sampling default). Judger existing pakai 0.6 / repair 0.7 — saat benchmark
   per-agent, samakan ke 0.7 kecuali sedang menguji pengaruh temp.
9. **Prompt baru**: saat menulis varian baru, **rujuk paper LatentMAS / latent
   collaboration** di `books/2511.20639v2_compressed.pdf` (pola planner/critic/
   refiner, instruksi structured walau kv_only). QuantaAlpha di
   `books/2602.07085v1_compressed.pdf`.
10. **Hardware runpod**: GPU 48 GB, RAM 50 GB, 9 vCPU (Xeon Gold 6342).
    Latent+realign ≈ 12–14 GB/proses → **paralel 3 proses** muat di 48 GB.
    Paralelisme via **multiprocessing (proses terpisah, backend sendiri-sendiri)**,
    BUKAN batching in-process (lebih robust untuk model singleton + KV per agent).

---

## 3. AXES EKSPERIMEN (grid final)

Per agent, per Phase A (independen):

```
grid = varian_prompt  ×  latent_steps{0,10,20,40,60,80}  ×  R repetisi
```

- **Thinking: TIDAK ada axis** (selalu off — lihat keputusan #6).
- **R repetisi** (default 5) untuk mengukur STABILITAS (varians antar-sample).
- latent_steps=0 ⇒ backend text-only (`get_backend`). latent_steps>0 ⇒ backend
  latent (`get_latent_backend`, `use_realign=True`). Kelompokkan job per tipe
  backend supaya model tak reload.
- Untuk agent `kv_only` (proposal/construct/consistency), saat benchmark
  jalankan **kv_and_text** agar latennya bisa di-decode & dinilai (persis pola
  artefak lama yang men-decode output). latent_steps=0 = generate teks biasa;
  latent_steps>0 = latent reasoning lalu decode.

---

## 4. INFRASTRUKTUR YANG WAJIB DI-REUSE (sudah ada, jangan bikin ulang)

| Komponen | Lokasi | Fungsi untuk eksperimen |
|---|---|---|
| `LatentAgent` + `load_agent(name, backend, path=...)` | `backend/latent_mas/agent.py` | **Kunci.** Muat agent dari YAML varian MANA SAJA (`path=`), jalankan standalone dengan `past_kv` (KV-chaining), `mode`, `latent_steps`, `temperature`, `json_mode`. |
| `get_backend` / `get_latent_backend` | `try/common.py` | Singleton backend text-only & latent(use_realign=True). Load model sekali per proses. |
| `run_case`, `_save_log`, `kv_shape_report` | `try/common.py` | Logging artefak + ringkasan KV (n_tokens/layers/MB) di tiap batas agent. |
| `prompt_ab.py` | `try/prompt_ab.py` | A/B/C varian, **`load_prompts_at_sha(sha, path)`** (ambil prompt dari commit lama via `git show`), helper KV-chain (`concat_kv_raw`, `run_corrected_chain`, `run_full_chain`). |
| `probe.py`, `test_kv_probe_v2.py` | `try/` | Probe isi KV-cache (untuk diagnosa akumulasi/collapse di Phase B). |
| `fixtures.py` | `try/` | Input ter-mock per agent + **backtest/feedback rekayasa** (untuk terminal chain). |
| `operator_families.py::families_of(expr)` | `backend/latent_mas/operator_families.py` | **Metrik variety** deterministik (operator → family). Juga `diversity_hint`. |
| `parsers.py::parse_hypothesis_exprs` | `backend/latent_mas/parsers.py` | Parse output judger → hypothesis + list expr. Untuk parse-rate. |
| `FrontEndPipeline._build_regulator_gate` | `backend/latent_mas/pipeline.py` | Bangun `FactorRegulator` gate deterministik → **expr valid/tidak** (pass-rate). |
| `introspect` agent | `prompts.yaml` | Decode isi KV (diagnostik) di Phase B. |

**Catatan import**: `try/` adalah package; jalankan via `python -m try.promptbench.<modul>`.
`runpy -m` bisa meng-import package bernama `try` walau `import try` (keyword)
gagal — pola ini sudah dipakai (`python -m try.run`).

---

## 5. SUMBER VARIAN PROMPT (untuk dikumpulkan di Phase 0c)

Simpan tiap varian sebagai **single-agent YAML** kompatibel `load_agent(path=)`
di `try/promptbench/variants/<agent>/<variant_id>.yaml`, lalu catat di
`variants_manifest.yaml`.

Sumber:
- **Current** (newEvol HEAD `218c665`) — `backend/latent_mas/prompts.yaml`.
- **feat/prompt-redesign-v2** branch (varian redesign Claude).
- **Commit historis** (via `git show <sha>:backend/latent_mas/prompts.yaml`):
  - `82cf925` redesign gaya QuantaAlpha ./factors (10 agent)
  - `5ac82f7` judger-only mutation/crossover
  - `b817b70` guidance re-entry
  - `117540a` fondasi LatentMAS vanilla
  - `e63bd08`, `c5fb7fe`, `f4b53b1`, `66b3768`, `67f4e95` (iterasi prompt lain)
- **Branch lain**: `MulaiDariNol`, `baseline-mutation`, `experiment/try`,
  `feat/latentmas-rework`.
- **Original QuantaAlpha/AlphaAgent**: `original-prompt/quantaalpha/*`
  (`proposal.yaml`, `consistency_prompts.yaml`, `evolution_prompts.yaml`,
  `factors/prompts/prompts.yaml`, `components/proposal/prompts.yaml`).
- **LatentMAS paper** Appendix E (planner/critic/refiner/judger) — transkrip
  manual dari `books/2511.20639v2_compressed.pdf`.
- **Varian baru buatan Claude** — dirujuk dari paper di atas.

---

## 6. SCORING RUBRIC (per role) — semua deterministik, simpan artefak mentah

- **proposal**: ada hipotesis 1 kalimat? menyebut $columns observable + pola
  temporal/cross-sectional? BUKAN klise volume-volatility yang dilarang?
  variety/non-duplikat antar-repetisi?
- **construct / judger**: `parse_hypothesis_exprs` sukses; tiap expr lolos
  `FactorRegulator` gate; arity benar; jumlah expr dalam rentang (≤3); variety
  family via `families_of`; diversitas antar-repetisi (bukan template di-rename).
- **consistency**: efeknya diukur terutama di Phase B (apakah parse/gate/variety
  membaik vs tanpa consistency). Di Phase A: decode & cek tidak meng-corrupt.
- **feedback**: JSON valid + key wajib + mengutip metrik [A]/[B].
- **mutation / crossover**: diukur via downstream (Phase B) — apakah hipotesis/
  expr hasil BERUBAH (bukan mirroring parent) & terarah.

Output skor: `results/phaseA/<agent>/scoreboard.{csv,md}` + artefak teks per-rep.

---

## 7. FASE (peta kerja; lihat juga TodoWrite di sesi)

- **Phase 0** — Setup. (a) branch ✅, (b) scaffold folder ✅, (c) skrip koleksi
  varian + manifest, (d) modul scorer, (e) driver paralel (process-pool 3 worker;
  grid §3; GPU guard).
- **Phase A** — ✅ DIJALANKAN (GPU, 2026-06-16). Benchmark per-agent independen
  CORE 4 (text-emitting mode). Hasil di `results/phaseA/scoreboard.{md,csv}` +
  artefak per (variant×ls×rep). Top per-agent (baris teratas tiap seksi):
    - proposal    : `proposal__working__258abdbbccea`        @ ls=60 (0.84, observable 0.8)
    - construct   : `construct__working__56396e7d44b0`        @ ls=0  (0.80); latent terbaik
                    `construct__git_optimalisasi__e9a6a179b181` @ ls=60 (0.707, variety 6)
    - consistency : flat 0.5 (kv_only → skor isolasi tak informatif; sinyal di Phase B)
    - judger      : `judger__git_gate_deterministik__6ce003675450` @ ls=20 (0.90, gate 0.8)
  Catatan: parse_rate < 1.0 di beberapa construct/judger meski teks tampak benar →
  "bagus-tapi-gagal-parse" → ditangani via `chain/parsing_hook.py` (lihat Phase B).
  A-ext (mutation/crossover/feedback/repair) belum diukur terpisah.
- **Phase B** — ✅ KODE SIAP (`chain/`, dry-run lolos; butuh GPU untuk run).
  Rantai multi-agent via KV-cache, bertahap, disiplin KV identik `pipeline.py`
  (front-end in-place; judger & feedback dari `deepcopy(kv_consist)`). Tahap:
    `s1_pc` proposal→construct[tip] · `s2_pcj` +judger[tip] (consistency dilewati) ·
    `s3_pccj` front-end penuh→judger[tip] · `s4_full_fb` +feedback[tip] (backtest
    rekayasa) · `s5_mut` mutation-guidance→… · `s6_cross` crossover-guidance→….
  Tiap batas agent → `Boundary` (kv_tokens/in/out/ls) dicetak+disimpan; tiap tip →
  scorer Phase A + `collapse.detect` (repetisi / unparseable / lonjakan-token KV).
  Varian default auto dari `scoreboard.csv` (top per-agent), override via
  `--pick agent=substr` / `--ls agent=N` / `--uniform-ls N`.
  **Parser hook**: `chain/parsing_hook.py` = EXTENSION POINT. Default delegate ke
  `parse_hypothesis_exprs` (nol perubahan). Saat prompt final dipilih & ketemu pola
  output-bagus-gagal-parse, daftarkan pre-normalizer di `_PRENORMALIZERS`; parser
  PRODUKSI `backend/latent_mas/parsers.py` TIDAK disentuh sampai pola terbukti aman.
  **Loop diagnosa**: output memburuk → coba kombinasi prompt lain → kalau tetap
  buruk → KV-probe (`introspect`/`kv_probe_v2`); detektor collapse menandai otomatis.
- **Phase C** — Ablation & redesign workflow: chain dengan/tanpa tiap agent
  kv_only (terutama consistency) → buang yang tak signifikan; prototipe agent/
  loop baru bila ada gap; output **rekomendasi workflow** (agent, urutan,
  latent_steps, per-agent setting).
- **Phase D** — Konsolidasi: tulis prompt pemenang ke `prompts.yaml` bersih +
  config workflow + `results/REPORT.md`.

---

## 8. SLOT FASE BARU (diisi SETELAH ada output LLM nyata)

> User mengantisipasi akan muncul fase baru yang hanya bisa dirancang setelah
> kita melihat output LLM dari runpod. Tambahkan di sini saat itu terjadi —
> jangan menebak sekarang. Format tiap entri:
>
> ```
> ### Phase X — <judul>  (ditambahkan <tanggal>, dipicu oleh <observasi output>)
> Hipotesis masalah:
> Rancangan uji:
> Metrik keputusan:
> Hasil & keputusan:
> ```

*(kosong — belum ada output LLM)*

---

## 9. CARA MENJALANKAN

```bash
cd quantalatent
# (0c) kumpulkan/refresh varian + manifest — tanpa GPU
.venv/bin/python -m try.promptbench.runners.collect_variants

# (0d) smoke-test scorer — tanpa GPU
.venv/bin/python try/promptbench/scoring/score.py

# (0e) dry-run driver (render + skor placeholder) — tanpa GPU, validasi pipa
.venv/bin/python -m try.promptbench.runners.bench \
    --agents proposal,construct,consistency,judger --latent-steps 0,10 --reps 2 --dry-run

# Phase A penuh di runpod (GPU):
.venv/bin/python -m try.promptbench.runners.bench \
    --agents proposal,construct,consistency,judger \
    --latent-steps 0,10,20,40,60,80 --reps 5 --workers 3
# → hasil: results/phaseA/scoreboard.{csv,md} + artefak NESTED:
#   results/phaseA/<agent>/<variant_short>/ls<N>/rep<R>.txt   (mudah disortir manual)

# Rapikan artefak Phase A LAMA (datar) → nested (idempoten; --apply utk eksekusi):
.venv/bin/python -m try.promptbench.runners.reorg_phaseA --apply

# ── Phase B: rantai multi-agent via KV-cache ──────────────────────────────
# KONSOLIDASI 2026-06-18: dua desain DIPERTAHANKAN (mengukur hal berbeda — lihat
# §10) di atas INFRASTRUKTUR BERSAMA impl 2: diagnostics/collapse.py, artifacts.py,
# scoring/score_chain.py + scoring/parsing_hook.py (parser hook, satu-satunya
# bagian impl 1 yang dipertahankan). Disiplin KV = LINEAR penuh (clone-on-transfer);
# DEFAULT_MAX_NEW=30000; metode input = seragam **FIXTURES.
#
# (B-STAGES) paket `chain/` — CLI berbasis --stages (prefix bertahap, ls TETAP)
# dry-run (tanpa GPU): cek wiring + picks auto dari scoreboard.csv
.venv/bin/python -m try.promptbench.chain.chain --stages all --reps 1 --dry-run
# runpod (GPU): rantai bertahap + kv_shape + collapse detector (paralel)
.venv/bin/python -m try.promptbench.chain.chain \
    --stages s1_pc,s2_pcj,s3_pccj,s4_full_fb --reps 3 --workers 3
# override varian/ls: --pick judger=gate_deterministik --ls construct=60 --uniform-ls 20
# evolution entry: --stages s5_mut,s6_cross
# → hasil: results/phaseB/scoreboard_stages.{csv,md} + artefak nested
#   results/phaseB/<stage>/<config>/rep<R>/{NN_<agent>.txt, chain.json}
#   (KV boundaries, collapse verdict, parse trace, tip response, score detail).
#
# (B-CHAINS) `chains/` + runners/bench_chain.py — CLI berbasis --chains
# dry-run (tanpa GPU): render tiap langkah + validasi wiring KV (delta = prompt+latent)
.venv/bin/python -m try.promptbench.runners.bench_chain \
    --chains pc_2agent,pcj_judger --latent-steps 0,20 --reps 1 --dry-run
# runpod (GPU): chain penuh + deteksi collapse/penimbunan KV
.venv/bin/python -m try.promptbench.runners.bench_chain \
    --chains front_end_full,front_end_feedback \
    --latent-steps 0,10,20,40 --reps 5 --workers 3
# override varian pemenang per agent (kalau sortir manual beda dari scoreboard):
#   --pick judger=working,construct=git_optimalisasi
# → hasil: results/phaseB/scoreboard.{csv,md} + per run:
#   results/phaseB/<chain>/<config>/rep<R>/{NN_<agent>.txt, chain.json}
# CATATAN scoreboard: STAGES → scoreboard_stages.{csv,md}; CHAINS → scoreboard.{csv,md}
#        (file terpisah, tidak saling menimpa di results/phaseB/).
```

Catatan teknis penting:
- `try` adalah keyword Python → `import try` / `from try.` ILEGAL di source.
  Jalankan via `-m try.promptbench...` (runpy/importlib boleh), dan di dalam paket
  pakai **relative import** (`from ..fixtures_pb import ...`, `from ...common import ...`).
- Bila file .py muncul error "null bytes"/SyntaxError aneh: cek
  `tr -cd '\000' < file | wc -c`. Pernah ada 1 spasi ter-tulis sebagai `\x00`.
- Scorer memakai **FactorRegulator PENUH** (regulator, bukan fallback) di env ini —
  terverifikasi. Regulator mencetak `factor_expression:` ke stdout → di driver
  di-redirect ke devnull.

## 10. STATUS TERKINI (2026-06-18)

- [x] Phase 0a–0e — setup, varian (47 unik; core4: proposal 7, construct 6,
      consistency 4, judger 7 + `_authored/claude_latentpaper.yaml`), scorer
      (regulator gate + variety + stabilitas), driver paralel. (lihat riwayat)
- [x] Phase A — **SUDAH JALAN di GPU 2026-06-16** (commit `06b1ea2`):
      scoreboard.{csv,md} + ~720 artefak core4. Top per-agent (lihat §7):
      proposal `working`@ls60 (0.84), construct `working`@ls0 (0.80),
      judger `git_gate_deterministik`@ls20 (0.90), consistency semua ~0.5
      (harus diukur DOWNSTREAM di Phase B). → User sedang **sortir manual**.
- [x] Artefak Phase A dirapikan ke layout **nested** `<agent>/<variant_short>/ls<N>/rep<R>.txt`
      (reorg_phaseA.py). `bench.py` juga sudah menulis nested untuk run berikutnya.

- [x] **Phase B — DIKONSOLIDASI (2026-06-18).** Dua desain DIPERTAHANKAN karena
      MENGUKUR HAL BERBEDA, di atas infrastruktur impl 2 yang sama. Keputusan user:
      semua fundamental ikut impl 2 (KV LINEAR penuh, DEFAULT_MAX_NEW=30000,
      collapse=diagnostics/, paralelisme ProcessPool, input seragam **FIXTURES),
      KECUALI parser hook (impl 1, dipertahankan). Keduanya dry-run lolos
      (ok=6/6, no MISSING var), **butuh GPU untuk run nyata**:
      - **STAGES `chain/`** — `chain.py` (CLI `--stages`): prefix bertahap pada
        latent_steps TETAP (top per-agent) → isolasi kontribusi MARGINAL tiap agent
        + skip-consistency (s2 vs s3) + evolution-seed (s5/s6). Kini memakai
        diagnostics/collapse.py, artifacts.py (nested), scoring/score_chain.py.
        KV linear clone-on-transfer (feedback meng-chain dari judger). ProcessPool
        text/latent split. Scoreboard → `scoreboard_stages.{csv,md}`.
      - **CHAINS `chains/` + `runners/bench_chain.py`** (CLI `--chains`): skenario
        tematik × GRID latent_steps × rep (termasuk evolution-first). Tak berubah
        selain parser hook kini lewat score_chain. Scoreboard → `scoreboard.{csv,md}`.
      - **Parser hook** dipindah `chain/parsing_hook.py` → `scoring/parsing_hook.py`
        (dependensi searah), di-wire ke `scoring/score_chain.py` → KEDUA desain dapat
        fallback + audit trace. `scoring/score.py` dapat param `parsed=` (Phase A
        tetap byte-identik by-default). `chain/collapse.py` DIHAPUS (redundan dgn
        diagnostics/collapse.py). FIXTURES dilengkapi var alt-variant (hypothesis_text,
        focus_hint, parent_*, diagnosis_*, backtest_summary) → semua pick render bersih.
- [ ] **Phase B run GPU** (stages + chains) → Phase C (ablation) → Phase D (konsolidasi prompt).

User sedang sortir manual varian mana yang layak utk Phase B; default = top
scoreboard (boleh diganti via `--pick`). Parser produksi belum diubah (tunggu pilihan
prompt final → tambal pre-normalizer di `scoring/parsing_hook.py`).

**Catatan KV-transfer (penting, sudah diverifikasi di kode):**
`latent_pass()` memutasi `past_key_values` IN-PLACE (HF DynamicCache). Karena itu
bench_chain SELALU `kv_deepcopy()` sebelum mengoper ke langkah berikutnya — output
tiap langkah membeku, pertumbuhan KV murni = prompt+latent_steps (terlihat di
chain.json: `delta` vs `expected_delta`). KNN auto-OFF saat latent_steps>0
(RoPE desync guard), jadi panjang KV tidak berubah diam-diam.

---

## 11. REDESIGN PROMPT v2 — ABLATION ALUR WORKFLOW (2026-06-19)

> Dibuat SETELAH user menganalisa output Phase A/B nyata + diskusi dengan Claude.
> Belum di-run GPU (prompt + chain + dry-run SAJA). Branch `feat/prompt-redesign-v2`.

**Diagnosis inti (dari artefak `results/phaseA` + `results/phaseB/{pc_2agent,pcj_judger}`):**
KV laten **lossy untuk payload SIMBOLIK, bukan untuk gist**. Hipotesis (gist)
selamat melintasi hop laten; EKSPRESI (string presisi) TIDAK — judger
menghalusinasi operator (`TS_EMA`, `MOA`) → gate 0/2. 4 agen = 3 hop = error
menumpuk (*compound failure*). Bukti pendukung: (a) contoh di system-prompt
proposal DISALIN verbatim → `cliche=true`; (b) `diversity_hint` di user-prompt
judger lama menyuruh "ganti family" → judger MALAH re-derive (smoking gun);
(c) construct meng-emit BOOLEAN MASK, bukan skor kontinu; (d) ls60 = glitch/
repetisi (`smoothedoothed`, `2,2,2,5,5`); (e) 4 varian consistency identik,
SILENT (kv_only), inert. **Prinsip desain**: begitu ekspresi jadi simbol, jangan
biarkan ia melintasi hop laten lagi → kurangi agen pasca-commit ATAU oper sebagai
TEKS.

**Keputusan user**: "build all, benchmark decides" + WAJIB prompt berbeda per
skenario (tugas role berbeda per alur). PROPOSAL sengaja IDENTIK lintas alur
(variabel terkontrol). Perbaikan prompt yang ditanam di semua alur: contoh
non-copyable (slot template `<...>`), anti-cliché ("jangan parafrase avoided
pattern"), WAJIB skor kontinu (bukan boolean mask), BUANG self-counting
(SL/feature → biar regulator), judger di-strip dari `diversity_hint`+`direction`,
read-back hipotesis 1 baris (anti-drift, angkat laten→token).

**Sumber prompt**: `variants/_authored/redesign_c{2_solo,3_finalizer,4_hybrid,4_latent}.yaml`
(multi-agen; `collect_variants` → per-agen `authored_redesign_c*`). **4 chain** di
`chains/chain_manifest.yaml`:

| chain | topologi | medium ekspresi | mengisolasi |
|---|---|---|---|
| `r_c2_solo` | proposal[kv_only] → construct[kv_and_text, TERMINAL] | pure-laten, commit @terminal | builder serba-bisa; 0 hop pasca-commit (paling robust) |
| `r_c3_finalizer` | proposal → construct[kv_only] → consistency[kv_and_text, TERMINAL] | pure-laten, commit @terminal | apakah refinement laten > c2? |
| `r_c4_hybrid` | proposal → construct → consistency → judger (semua kv_and_text) | **TEKS** (`prior_factors`) | full division-of-labor + kanal simbolik anti-korupsi |
| `r_c4_latent` | idem c4_hybrid tapi middle **kv_only** | pure-laten (lewat KV) | CONTROL: efek kanal TEKS (vs c4_hybrid) & efek jumlah agen (vs c2/c3) |

**Mekanisme hand-off TEKS (hanya c4_hybrid)**: `runners/bench_chain.py` —
helper `_factor_block()` + dict `carried`; tiap langkah yang DI-DECODE menyuntik
blok `HYPOTHESIS:`/`EXPRESSION N:` (hasil `parse_hypothesis_exprs`, sama dgn
scorer) ke var `prior_factors` langkah berikut. **Backward-compatible**: chain
lain tak mereferensikan var itu → no-op (Jinja `default('')`). Langkah `kv_only`
tak meng-update `carried` → alur pure-laten tak pernah pakai teks.

**Cara run (GPU)**:
```bash
.venv/bin/python -m try.promptbench.runners.bench_chain \
    --chains r_c2_solo,r_c3_finalizer,r_c4_hybrid,r_c4_latent \
    --latent-steps 10,20,40,60 --reps 5 --workers 3
```
Bandingkan di `results/phaseB/scoreboard.{csv,md}`: `parse_rate`, `gate_pass_rate`,
`variety_families`, `n_distinct_hypotheses`, `score_std`. Knob: ls rendah (10-20)
mungkin jaga fidelity simbolik > ls60 (glitch ls60 sebagian murni degradasi 4B).

**Status**: dry-run 8/8 ok (4 chain × ls{0,60}), chain lama tak regresi, py_compile
ok. **Hasil & keputusan: PENDING GPU.**
