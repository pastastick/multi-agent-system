# Metode QuantaLatent — LatentMAS × QuantaAlpha

> Rangkuman teknis mandiri (bukan skripsi): persamaan matematika, cara kerja,
> workflow, dan pointer kode dari mekanisme inti QuantaLatent. Ditujukan sebagai
> basis pengembangan menuju sistem **optimal, real-time, dan berkelanjutan** —
> bukan hanya berjalan di lingkungan eksperimen.
>
> Semua pointer kode relatif terhadap `quantalatent/backend/`.
> Disusun 2026-07-09.

---

## 0. Peta besar: dua pilar yang disatukan

QuantaLatent menggabungkan **dua algoritma yang ortogonal**:

| Pilar | Asal | Peran di sistem | Satuan kerja |
|---|---|---|---|
| **LatentMAS** | multi-agent yang "berpikir" di ruang laten (KV-cache), bukan teks | *bagaimana* agen berkomunikasi & bernalar | KV-cache antar-agen |
| **QuantaAlpha** | evolusi trajectory untuk alpha-mining kuantitatif | *apa* yang dicari (faktor alfa) & *bagaimana* populasi solusi berevolusi | StrategyTrajectory |

Rumusan intinya: **satu iterasi mining** menghasilkan satu *trajectory* (hipotesis
+ ekspresi faktor + metrik). Front-end LatentMAS menyusun trajectory itu lewat
rantai agen yang saling mengoper KV-cache; QuantaAlpha mengevolusikan populasi
trajectory lintas generasi (original → mutation → crossover → …).

```
                      ┌──────────── QuantaAlpha (evolusi populasi) ─────────────┐
                      │                                                          │
   direction  ──►  ORIGINAL ──►  MUTATION ──►  CROSSOVER ──►  MUTATION ──►  ...  │
                      │             │              │                             │
                      ▼             ▼              ▼                             │
              ┌─────────────── LatentMAS front-end (per-trajectory) ────────────┐
              │  proposal → design → construct → [gate → repair]  →  backtest    │
              │  (handoff KV-cache; comm_mode: text | kv_and_text | kv)          │
              └──────────────────────────────────────────────────────────────────┘
```

Model dasar: **Qwen3-4B** (HF Transformers, DynamicCache). Satu instance model
melayani semua peran agen — yang berbeda hanya *prompt*, *mode KV*, dan *parser*
(`latent_mas/agent.py`).

---

## 1. Mekanisme laten (LatentMAS)

Inti LatentMAS: alih-alih agen A menuliskan jawaban sebagai **teks** lalu agen B
membacanya kembali (lossy: teks → token → embedding), agen B langsung mewarisi
**KV-cache** agen A. KV-cache adalah *working memory* laten — representasi internal
setiap token yang sudah "dicerna" model.

### 1.1 Latent reasoning step (berpikir tanpa menulis)

Sebuah agen `kv_only` melakukan forward pass atas prompt-nya lalu **N langkah
laten**: hidden-state terakhir diumpankan balik sebagai *input embedding* langkah
berikutnya, tanpa pernah men-decode ke token teks.

Rekurensi (`llm/models.py::generate_latent_batch`, baris 281–301):

$$
h^{(0)} = \text{Transformer}(\text{prompt})_{[:,-1,:]}
$$
$$
e^{(t)} = \text{Realign}\!\left(h^{(t-1)}\right), \qquad
h^{(t)},\; \text{KV}^{(t)} = \text{Transformer}\!\left(e^{(t)} \,\middle|\, \text{KV}^{(t-1)}\right)
$$

untuk $t = 1 \dots N$ (`latent_steps`, default 10 di
`core/latent/latent_method.py`; dikonfigurasi per-agen di `latent_mas/prompts.yaml`).
Setiap langkah **memperpanjang KV-cache** tanpa menghasilkan token — inilah
"berpikir dalam diam". KV akhir menjadi warisan untuk agen berikutnya.

### 1.2 Latent realignment — jembatan output→input space (persamaan kunci)

Masalah: $h$ hidden-state hidup di **output space** (setelah semua layer), sedangkan
input embedding $e$ hidup di **input space** (sebelum layer pertama). Mengumpankan
$h$ mentah sebagai input membingungkan model. `LatentRealigner`
(`llm/_shared.py:646`) mencari matriks $M$ yang memetakan keduanya via **ridge
regression** atas matriks embedding masukan $W_{in}$ dan keluaran $W_{out}$
(keduanya $\in \mathbb{R}^{V \times d}$):

$$
\min_{M}\; \lVert W_{out} M - W_{in} \rVert_F^2 + \lambda \lVert M \rVert_F^2
$$

Solusi normal equations (di-*cache* per model, dihitung sekali):

$$
M = \left(W_{out}^\top W_{out} + \lambda I\right)^{-1} W_{out}^\top W_{in},
\qquad \lambda = 10^{-5}
$$

Lalu proyeksi dinormalisasi ke magnitudo rata-rata embedding input agar skala
vektor laten tetap "seperti token biasa":

$$
\tilde{e} = h M, \qquad
e = \tilde{e}\cdot \frac{\bar{n}}{\lVert \tilde{e}\rVert}, \qquad
\bar{n} = \frac{1}{V}\sum_i \lVert W_{in}[i]\rVert
$$

Bila `use_realign=False`: $M = I$ (identitas) — hanya normalisasi magnitudo.
Ini variabel ablasi penting untuk riset.

### 1.3 KV-cache sebagai objek yang dioper — aturan emas

`LocalLLMBackend.run()` **memutasi `past_key_values` in-place**. Maka KV yang dibaca
lebih dari satu konsumen **wajib** di-clone dulu. Seluruh disiplin ini ada di
`latent_mas/kv_ops.py`:

| Operasi | Fungsi | Rumus/aksi | Kegunaan |
|---|---|---|---|
| Isolasi | `kv_deepcopy` | clone semua tensor $(K,V)$ per layer | cegah kontaminasi in-place lintas cabang |
| Konkatenasi | `kv_concat` | $K = \operatorname{cat}(K_1,\dots,K_m,\,\text{dim}=\text{seq})$ | **hierarchical working-memory transfer (LatentMAS Eq. 4)** — gabung KV beberapa parent |
| Truncate | `kv_truncate` | simpan $\max$-tokens terakhir pada dim seq | batasi memori (anti-OOM) |
| Seleksi relevansi | `kv_knn_filter` | lihat §1.4 | pilih token relevan, bukan sekadar N terakhir |
| Persist | `kv_save/kv_load` | pindah ke CPU → `torch.save` | ambil KV agen dari run kemarin |

Bentuk tensor per layer: $K, V \in \mathbb{R}^{B \times H \times S \times d_h}$;
concat/truncate beroperasi di $\text{dim}=-2$ (panjang sekuens $S$).

### 1.4 KNN filtering — seleksi token berdasar relevansi (bukan recency buta)

`kv_knn_filter` (`llm/_shared.py:261`) memilih token KV paling relevan terhadap
query saat ini, alih-alih memotong buta. Skor = cosine similarity antara query
hidden dan **key vector di layer tengah** (dianggap paling seimbang low/high-level):

$$
s_j = \frac{\bar{k}_j \cdot q}{\lVert \bar{k}_j\rVert\,\lVert q\rVert},
\qquad \bar{k}_j = \frac{1}{H}\sum_{h=1}^{H} K^{(\text{mid})}_{h,j,:}
$$

Pertahankan $k = \max(\lfloor S \cdot p\rfloor, \text{min\_keep})$ token: `min_keep`
token terbaru **selalu** disimpan (konteks tersegar) + top-$(k-\text{min\_keep})$
skor tertinggi dari token awal. Urutan temporal dijaga.

**Subtilitas RoPE (penting).** Setelah subset token diseleksi, posisi aslinya
berlubang (mis. `[3,7,12,400,401]`) tapi panjang fisik menyusut ke $k$. Karena RoPE
bersifat aditif $R(a)R(b)=R(a+b)$, key di-rotasi ulang $R(\text{new}-\text{old})$
agar seakan berada di posisi kontigu $[0..k-1]$
(`_rerotate_keys_contiguous`, baris 197). Tanpa ini → desync posisi → output
degenerate. Hanya **key** yang dirotasi (value tak terkena RoPE). Asumsi:
`attention_scaling == 1.0` (RoPE standar Qwen3-4B); untuk YaRN/linear scaling
komposisi ini hanya hampiran — **catatan untuk deployment long-context**.

### 1.5 Injeksi laten ke decoder (jalur vLLM)

Pada jalur vLLM (`latent_method.py::run_batch_vllm`), hidden-state laten dari
agen-agen pemikir **disisipkan sebagai `prompt_embeds`** tepat setelah token
`<|im_start|>user\n` milik judger, lalu digenerate oleh vLLM. Ini alternatif dari
mewariskan KV langsung — laten diperlakukan sebagai "embedding tambahan" pada
prompt decoder.

---

## 2. Abstraksi Agen & tiga medium komunikasi

Setiap agen = **(spec prompt Jinja + mode KV + parser)**, dimuat dari
`latent_mas/prompts.yaml` (`agent.py::_load_specs`). Tidak ada agen yang tahu
tentang pipeline → tiap agen bisa dijalankan sendiri untuk debug.

Mode KV per-agen:
- `kv_only` — bernalar laten, **tidak** decode teks (hanya membentuk KV). Proposal/design/guidance.
- `kv_and_text` — bernalar laten **lalu** generate teks dari KV. Construct/repair/feedback.
- `text_only` — generate teks tanpa menyimpan KV.

**`comm_mode`** (`FrontEndPipeline`, `pipeline.py:125`) adalah **variabel eksperimen
utama** — medium handoff antar-agen:

| `comm_mode` | Handoff | Deskripsi |
|---|---|---|
| `text` | via TEKS | semua agen decode teks; agen berikut membaca teks (baseline terkendali, no KV) |
| `kv_and_text` | via KV-cache | semua agen decode teks **tetapi** handoff tetap lewat KV |
| `kv` | via KV-cache | hanya construct & feedback yang decode teks; sisanya laten murni (paling hemat token) |

Ini memungkinkan **studi banding terkendali**: prompt identik, yang berbeda hanya
medium. `keep_answer_in_kv=True` (default, "NO-CROP") mempertahankan jawaban yang
digenerate di dalam KV agar agen berikut membaca output **asli**, bukan vektor laten
lossy (perbaikan drift; `agent.py:71`).

---

## 3. Workflow front-end (menyusun satu trajectory)

`FrontEndPipeline.run()` (`pipeline.py:260`). Rantai **sequential** dengan disiplin
isolasi KV:

```
seed_kv ─► proposal ─KV─► design ─KV─► construct ─┬─► [gate semua ekspresi]
          (hipotesis)   (pilih       (TERMINAL     │        │ ada ≥1 lolos → pakai
                        palette       EMITTER:      │        │ SEMUA gagal ↓
                        VAR+FUNC)     JSON faktor)  │        ▼
                                                    │   repair ×N (deepcopy(kv_construct)
                                                    │   per attempt; mode minimal→different→bold)
                                                    ▼
                                              FrontEndOutput(hypothesis, expressions, kv_final, ...)
```

Tahapan:
1. **proposal** — merumuskan *hipotesis mekanisme* pasar. Handoff `text` (ronde
   original) atau `kv` (evolution: membaca arah Director dari `seed_kv`).
2. **design** — memilih *palette* VARIABLE + FUNCTION dari operator DSL, membaca
   hipotesis **asli** dari KV (no-crop memperbaiki drift hipotesis).
3. **construct** — *terminal emitter*: menulis JSON
   `{hypothesis, factors:[{name, expression, explanation}]}`. Disuntik
   **diversity_hint** (§5.3) agar tak monokultur operator.
4. **gate & repair** (`_gate_and_repair_factors`, baris 490) — lihat §4.

### 3.1 Disiplin distribusi KV (mengapa pipeline ada)

Aturan emas: *KV yang akan dibaca >1 konsumen HARUS `kv_deepcopy` dulu*. Contoh:
tiap attempt repair berangkat dari `kv_deepcopy(kv_construct)` — baseline identik,
tak terkontaminasi attempt sebelumnya. Tanpa ini, mutasi in-place bocor antar-cabang
(bug senyap).

---

## 4. Kendali mutu faktor — regulator berlapis

Gate deterministik memutuskan ekspresi mana yang boleh ke backtest
(`pipeline.py::_build_regulator_gate` + `factors/regulator/factor_regulator.py`).
Urutan lapisan (fail-closed):

1. **auto-fix arity deterministik** (tanpa LLM): tulis ulang yang tak-ambigu,
   mis. `RANK(A,n) → TS_RANK(A,n)` (cross-sectional→time-series).
2. **parsable** — AST valid (`is_parsable`).
3. **arity** — jumlah argumen fungsi benar (`validate_function_arity`).
4. **variabel dikenal** — tolak halusinasi seperti `$return_1d`.
5. **degenerate-args** — tolak `REGRESI(x,x)` (arity valid tapi faktor mati).
6. **kompleksitas & redundansi** (`is_expression_acceptable`, baris 413):

$$
\text{terima} \iff
\underbrace{D(f) \le \theta_D}_{\text{duplikasi alpha-zoo}} \;\wedge\;
\underbrace{\text{SL}(f) \le \theta_{SL}}_{\text{symbol length}} \;\wedge\;
\underbrace{\text{ER}(f) \le \theta_{ER}}_{\text{\# base features}}
$$

Ambang default: $\theta_D = 8$, $\theta_{SL} = 300$, $\theta_{ER} = 6$
(`FactorRegulator.__init__`, baris 288). $D(f)$ = ukuran subtree yang duplikat
terhadap **alpha-zoo** (bank faktor yang sudah diterima — anti mining ulang).

**Repair** hanya dijalankan bila **SEMUA** ekspresi gagal gate. Loop adaptif ≤N
attempt (mode `minimal → different → bold`), tiap attempt membaca konteks via
`past_kv=deepcopy(kv_construct)` **plus** teks (`former_expression`+`error_log`).
Early-exit bila output tak berubah (echo). Gagal total → **fail-closed** (drop, tak
ke backtest).

---

## 5. Evolusi trajectory (QuantaAlpha)

`EvolutionController` (`pipeline/evolution/controller.py`) mengorkestrasi siklus
`ORIGINAL → MUTATION → CROSSOVER → MUTATION → …` sampai `max_rounds`.

### 5.1 Mutation & Crossover sebagai GUIDANCE (bukan pewarisan KV)

Pemetaan kanonis LatentMAS → QuantaAlpha:
- **Mutation** = eksploitasi. Agen `mutation` (`kv_only`, seed=None) membaca **teks**
  satu parent → membentuk KV *arah refine* (`guidance_kv`) → **menyemai** front-end
  via `run(seed_kv=guidance_kv)`.
- **Crossover** = eksplorasi. Agen `crossover` membaca **teks** $k$ parent → KV
  *arah fusi* → menyemai front-end. (Pemetaan langsung hierarchical `kv_concat`.)

Keputusan desain krusial (`pipeline.py::run_evolution`, baris 381): materi parent
ditransfer sebagai **TEKS**, dan `trajectory.kv_cache` **tidak** diwariskan
antar-generasi. Konsekuensi: KV per ronde terbatas (~guidance + front-end ≈ 3k token)
→ **tak ada akumulasi/over-KV lintas generasi** → mencegah *collapse* (akar regresi
yang pernah terjadi). Ini pelajaran penting untuk berkelanjutan: **bounded memory
per generasi**.

### 5.2 Seleksi induk & "best" — skor dengan penalti diversitas

`get_best_trajectories` (baris 1088). Pertama filter *sukses* lalu skor:

**Gate sukses** (model-free, OOS; `trajectory.py::is_successful`):

$$
\text{sukses}(\tau) \iff \text{FactorIC}(\tau) > \theta_{IC} \;\wedge\; \text{FactorICIR}(\tau) > \theta_{ICIR}
$$

Default $\theta_{IC}=\theta_{ICIR}=0$ (longgar: IC>0 ∧ ICIR>0), diperketat via
`experiment.yaml` setelah ada data nyata.

**Skor efektif** (HYBRID part-2, `operator_families.py::diversity_penalized`):

$$
s(\tau) = m(\tau) - \lambda \cdot \text{pen}_{\text{family}}(\tau)
$$

dengan $m(\tau)=$ FactorIC_mean, dan penalti redundansi family operator:

$$
\text{pen}_{\text{family}}(\tau) = \min_{g \in \mathcal{F}(\tau)} \frac{\text{count}_{\text{pop}}(g)}{|\text{pop}|} \;\in [0,1]
$$

Memakai **min** rarity → memperkenalkan **satu** family langka sudah menyelamatkan
dari penalti (reward novelty, bukan menghukum yang juga memakai family umum).
$\lambda=0$ default (murni $m(\tau)$); $\lambda \approx 0.01\text{–}0.03$ mendemosikan
faktor redundan **tanpa** hard-reject. Urut menurun → ambil top-$n$.

### 5.3 Diversity hint (soft, altitude operator)

`diversity_hint` (`operator_families.py:91`): melawan monokultur operator (4B default
ke `RANK`/`TS_ZSCORE`/`TS_PCTCHANGE`). Family "RICH" yang kronis absen
(smoothing/regression/technical/conditional/math/ts_pair/quantile) ditonjolkan **plus
afordansinya** ("SMA/WMA = ekstraksi tren/momentum"), ditutup klausa kesetiaan
("hanya bila benar melayani mekanisme"). Family diklasifikasi deterministik dari nama
fungsi via regex (`families_of`).

### 5.4 Negative memory & correlation gate (altitude mekanisme)

- **Negative memory** (`negative_memory.py`): mekanisme yang **gagal** (evaluasi
  menyeluruh: hipotesis + ekspresi + metrik) direkam lintas-generasi → di-render jadi
  daftar "AVOID" → disuntik ke prompt proposal ronde berikut. Berbeda dari
  `diversity_lambda` (operator) — ini di **level mekanisme**.
- **Correlation gate** ($|\text{Pearson}| > 0.7$ default): drop faktor baru yang
  terlalu mirip faktor lain se-ronde ATAU di persistent store — melawan duplikasi.

---

## 6. Evaluasi — metrik yang jujur vs tercemar

Peringatan metodologis penting (`trajectory.py::get_primary_metric`, baris 100):

- **Metrik jujur (dipakai seleksi)**: **per-factor RankIC OOS standalone**
  (`FactorIC_mean`), model-free. Tidak tercemar *baseline floor*.
- **Metrik tercemar (JANGAN untuk banding)**: **RankIC combined LightGBM**.
  Terbukti empiris (2026-06-14) 95–103% didominasi 4 fitur baseline
  NestedDataLoader → tak bisa membedakan faktor baik/buruk. Hanya fallback untuk
  trajectory lama, di-log WARNING.

Definisi (per hari $t$, cross-section saham):

$$
\text{IC}_t = \operatorname{corr}\!\left(f_t,\; r_{t+1}\right),\quad
\text{RankIC}_t = \operatorname{corr}\!\left(\operatorname{rank}(f_t),\; \operatorname{rank}(r_{t+1})\right)
$$
$$
\text{ICIR} = \frac{\overline{\text{IC}}}{\operatorname{std}(\text{IC})}
$$

$\text{ICIR}$ menangkap **stabilitas** sinyal (bukan sekadar kekuatan) — kunci untuk
klaim "berkelanjutan".

### Status empiris terkini (n=1 run/mode, JANGAN overclaim)

Batch `2026-07-05` (Qwen3-4B, `prod/backend-v1`), per-factor RankIC OOS:

| Mode | n IC | mean | max | IC>0 |
|---|---:|---:|---:|---:|
| `text` | 12 | −0.0087 | **+0.0449** | 2/12 |
| `kv_and_text` | 16 | **−0.0059** | +0.0215 | 6/16 |
| `kv` | 17 | −0.0174 | −0.0049 | 0/17 |

Bacaan: `kv_and_text ≥ text >> kv` pada n=1. **Caveat riset**: pembacaan awal "KV
laten lossy untuk muatan simbolik" kemungkinan sebagian **artefak bug operasional**
(cap latensi, degenerasi repetisi pada KV panjang), bukan sifat intrinsik laten;
framing yang lebih tepat adalah **diversity collapse**. Butuh replikasi ≥3 seed/mode
untuk signifikansi. Kualitas absolut masih lemah di semua mode (mayoritas IC negatif)
→ masalah *sistem mining*, ortogonal terhadap pertanyaan KV vs TEXT.

---

## 7. Menuju real-time & berkelanjutan (roadmap engineering)

Bagian ini menghubungkan mekanisme di atas dengan tujuan **deployment**. Hambatan
nyata sudah terdokumentasi di `MONITORING_NOTES.md`; berikut pemetaannya ke solusi.

### 7.1 Stabilitas memori (syarat mutlak untuk long-running)
- **Leak GPU ~600MB/loop** (B1) & **OOM di crossover** (deepcopy KV ~40k token):
  akar = KV membengkak + akumulasi antar-loop. Mitigasi terpasang:
  `expandable_segments:True`, proses segar per resume. **Belum ada fix leak** → ini
  blocker #1 untuk *sustained*. Arah: budget KV eksplisit per loop
  (`kv_truncate`/`kv_knn_filter` agresif) + free cache determistik antar-agen.
- Prinsip **bounded memory per generasi** (§5.1) sudah benar — pertahankan dan
  perluas: jangan pernah biarkan KV tumbuh monoton lintas ronde.

### 7.2 Latensi yang dapat diprediksi (syarat real-time)
- **Degenerasi repetisi** (B2): generasi teks berkondisi KV panjang bisa "ngoceh"
  30–46 menit sampai konteks penuh. Mitigasi: `max_time=300s`
  (`LATENT_MAX_GEN_SECONDS`) + cap `max_new_tokens` wajar. Untuk real-time: turunkan
  cap, tambah *repetition penalty* / early-stop deteksi loop, dan pertimbangkan
  jalur **vLLM** (§1.5) untuk throughput decode.
- KNN filtering (§1.4) + truncate menjaga panjang konteks → latensi laten stabil.

### 7.3 Kualitas sinyal (syarat "berguna", bukan sekadar jalan)
- Mayoritas faktor IC negatif → prioritaskan: (a) kualitas hipotesis (negative memory
  + prompt), (b) anti-duplikasi (correlation gate + alpha-zoo), (c) diversitas
  operator (`diversity_lambda` > 0, diversity_hint). Perketat gate sukses
  ($\theta_{IC}, \theta_{ICIR}$) begitu ada data GPU nyata.
- Ganti metrik seleksi ke **FactorIC/ICIR** (sudah), pensiunkan RankIC combined.

### 7.4 Aliran data kontinu (syarat online/streaming)
- Pipeline saat ini batch/offline (Qlib backtest). Untuk real-time: pisahkan
  **discovery** (mining faktor, lambat, offline) dari **serving** (evaluasi faktor
  terpilih atas data live, cepat). Faktor lolos gate → `kv_save` + ekspresi DSL yang
  ringan dieksekusi streaming; re-mining terjadwal (walk-forward) bukan per-tick.
- ICIR & walk-forward OOS adalah kontrak "berkelanjutan": faktor hanya di-*promote*
  bila stabil lintas jendela waktu, dan di-*retire* bila decay (perluas negative
  memory ke *live decay*).

### 7.5 Reprodusibilitas & observabilitas
- `RunLogger` (`latent_mas/runlog.py`) + `gate_log`/`extra_info` per trajectory sudah
  menyediakan audit. Untuk produksi: tambah tracing latensi per-agen (sudah ada
  `latent_s`/`gen_s`), alarm bila cap latensi/OOM sering mengikat.

---

## 8. Indeks file kunci

| Area | File | Isi |
|---|---|---|
| Forward pass laten | `llm/models.py` | `generate_latent_batch`, `_apply_latent_realignment` |
| Realignment & KV math | `llm/_shared.py` | `LatentRealigner` (ridge), `kv_knn_filter`, RoPE re-rotate, `kv_truncate` |
| Operasi KV | `latent_mas/kv_ops.py` | deepcopy, `kv_concat` (Eq.4), save/load, distribute |
| Agen | `latent_mas/agent.py` + `prompts.yaml` | `LatentAgent`, spec, mode KV, parser |
| Front-end | `latent_mas/pipeline.py` | proposal→design→construct→gate/repair, comm_mode |
| Diversitas operator | `latent_mas/operator_families.py` | families_of, diversity_hint, family penalty |
| Evolusi | `pipeline/evolution/controller.py` | siklus ronde, seleksi $s(\tau)$, metrik |
| Trajectory | `pipeline/evolution/trajectory.py` | is_successful, get_primary_metric |
| Regulator | `factors/regulator/factor_regulator.py` | SL/ER/duplikasi, arity, variabel |
| Loop utama | `pipeline/loop.py` | penyambung end-to-end + backtest |
| Status empiris | `MONITORING_NOTES.md` | hasil batch, daftar bug (B1–B14), TODO |
```
