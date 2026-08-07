# MONITORING NOTES — Eksperimen KV vs TEXT (batch 2026-07-05)

> Catatan lintas-sesi (dibuat Claude, 2026-07-06). Sumber kebenaran status batch,
> daftar bug/kelemahan sistem, dan TODO. Update file ini setiap ada perkembangan.
> Konteks skripsi: studi banding terkendali comm_mode `text` / `kv_and_text` / `kv`
> (Qwen3-4B, branch `prod/backend-v1`).

---

## 1. STATUS BATCH (stamp `2026-07-05_09-36-48`)

| Mode | Status | Pool final | Catatan |
|---|---|---|---|
| `text` | ✅ **6/6 LENGKAP** (06/07 04:31) | `prod_text_.../trajectory_pool_MERGED.json` | Run asli 5 + `c30de8e38aaa` (resume). Backtest resume TIDAK hang → kematian run asli ≈ I/O volume (B12), bukan hang inheren. |
| `kv_and_text` | ✅ **6/6 LENGKAP** | `prod_kv_and_text_.../trajectory_pool_MERGED.json` | Run asli 5 + `92db70581b71` (resume 00:10). Tanpa cap gen. |
| `kv` | ✅ **6/6 LENGKAP** (1 kosong) | `prod_kv_..._-resume/trajectory_pool.json` | take-3 pasca-fix B11 (3 traj) + resume pasca-ErrIO (3 traj, 04:09). `5228ed3135cf` (mutation dir1) KOSONG (B13). Dengan cap 300s + fix B11. |

### HASIL AKHIR — per-factor RankIC OOS terekam (metrik jujur, model-free), 2026-07-06

| Mode | n IC terekam | mean | median | max | min | IC>0 |
|---|---:|---:|---:|---:|---:|---:|
| `text` | 12 | −0.0087 | −0.0110 | **+0.0449** | −0.0356 | 2/12 |
| `kv_and_text` | 16 | **−0.0059** | −0.0059 | +0.0215 | −0.0387 | **6/16** |
| `kv` | 17 | −0.0174 | −0.0201 | −0.0049 | −0.0313 | **0/17** |

Bacaan awal (n=1 run/mode — JANGAN overclaim):
1. `kv` murni TERBURUK kategoris: TIDAK ADA satu pun faktor ber-IC positif, +1 trajectory kosong. Bahkan setelah fix B11 membuatnya *jalan*, KUALITAS faktornya tetap paling rendah → konsisten hipotesis skripsi (handoff laten murni lossy utk muatan simbolik), kini dengan implementasi yang fair.
2. `kv_and_text` vs `text`: mean/median per-faktor sedikit lebih baik di kv_and_text (−0.0059 vs −0.0087; IC>0: 6/16 vs 2/12), TAPI faktor tunggal terbaik ada di text (+0.0449). Belum ada pemenang tegas — butuh replikasi (≥3 seed/run per mode) untuk uji signifikansi.
3. Kualitas absolut masih lemah di SEMUA mode (mayoritas IC negatif) — masalah sistem mining (kualitas hipotesis/faktor, duplikasi B9), ortogonal terhadap pertanyaan KV vs TEXT.
4. Duplikasi lintas-ronde parah & terkuantifikasi: banyak corr_dropped |corr|=0.999–1.000 terhadap faktor lama di store (mis. `small_cap_stocks_with_unusually_1` muncul sebagai duplikat di KETIGA mode) — LLM regenerasi faktor yang sama berulang (perkuat B9).
5. Catatan protokol utk Bab 4: run kv memakai fix B11 + cap 300s; text/kv_and_text tidak (cap tak mengikat generasi sehat). Metrik RankIC combined LightGBM JANGAN dipakai lintas-mode (B8).

**Run dir**: `backend/runs/prod_{mode}_2026-07-05_09-36-48[...-resume]`, summary: `backend/runs/prod_2026-07-05_09-36-48_summary.log`.

### Pool metrik saat ini (FactorIC_mean = metrik jujur / model-free; RankIC = combined LightGBM, MASIH tercemar baseline)

`text`: orig0 **+0.0449**, orig1 −0.0029, mut0 −0.0356, mut1 −0.0228, cross1 −0.0075
`kv_and_text`: orig0 **+0.0157**, orig1 −0.0150, mut0 −0.0027, mut1 −0.0079, cross1 −0.0137

⚠️ Mayoritas faktor ber-IC negatif; hanya original dir-0 yang positif di kedua mode.
RankIC combined seragam ~0.033–0.037 lintas SEMUA trajectory & mode → konsisten dengan
temuan lama bahwa RankIC combined didominasi 4 fitur baseline, BUKAN faktor baru.
Perbandingan KV vs TEXT harus pakai FactorIC_mean/per-factor RankIC OOS.

---

## 2. BUG / KELEMAHAN DITEMUKAN (sesi 2026-07-06)

### B1. OOM di crossover#2 kv_and_text — `kv_ops.kv_deepcopy` [MITIGASI DIPASANG]
- Trace: `pipeline.py:358 run() → kv_deepcopy(r_design.kv_cache)` — alokasi gagal 166MB, 44.38/44.42 GiB terpakai, 4.19 GiB reserved-unallocated (fragmentasi).
- Akar gabungan: (a) KV design ~40k token (lihat B2) → deepcopy ≈ +6GB; (b) akumulasi 5 task sebelumnya di proses yang sama (leak GPU ~600MB/loop, isu lama); (c) fragmentasi PyTorch.
- Mitigasi: run resume & run berikutnya pakai `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + proses segar per resume. **Belum ada fix leak per-loop** → TODO T6.

### B2. Degenerasi generasi EPISODIK = jawaban "kenapa design/construct kadang sangat lama"
- Timing text mode: design mean 23s, construct 18s — selalu cepat.
- Timing kv_and_text: design 8/9 panggilan ~29s, **1 panggilan 2794s**; crossover 3/4 normal, **1 panggilan 1768s** — keduanya di task crossover#2, disertai warning transformers "exceeded the model's predefined maximum length (40960)".
- DIAG kv mode: proposal 2.7s + design 3.0s (laten murni, cepat) → **construct 755s** output repetisi ("selection, selectioninvestment ... expression expression expression").
- Pola: generasi teks berkondisi KV panjang / KV laten tanpa jangkar teks → repetition collapse; TIDAK ada cap output (`max_new_tokens: 40960` = seluruh konteks) → model "ngoceh" sampai konteks penuh, 30–46 menit per panggilan, lalu output unparseable → retry → dobel.
- Ini sekaligus **temuan ilmiah** (KV laten lossy/destabilizing untuk emisi simbolik pada 4B) DAN masalah operasional.
- **MITIGASI DITERAPKAN (2026-07-06, perintah user "inferensi agent max 5 menit")**: `max_time=300s` di kedua panggilan `model.generate` (`client.py::generate_text` & `generate_from_kv`), default via `_CoreEngine.max_gen_seconds`, override env `LATENT_MAX_GEN_SECONDS`. Catatan komparabilitas: generasi SEHAT di semua mode selesai <40s (cap tak pernah mengikat); yang terpotong hanya episode degenerate yang outputnya unparseable juga. Run text & kv_and_text yang sudah selesai TIDAK memakai cap ini — nyatakan di Bab 4 bila relevan. Run kv parsial tanpa-cap diarsipkan `DIAG2-kvdegen-nocap_prod_kv_...`; run kv final (mulai 01:31) memakai cap.

### B3. Bug resume: `load_state()` menimpa `crossover_idx` [FIXED — uncommitted]
- `load_state` set `_crossover_idx` dari state LALU memanggil `_prepare_crossover_groups()` yang me-reset `_crossover_idx = 0` → resume di fase crossover selalu mengulang group yang sudah selesai.
- Fix: `backend/pipeline/evolution/controller.py::load_state` — simpan idx sebelum prepare, pulihkan sesudahnya (`min(idx, len(groups))`). **Belum di-commit.**

### B4. Semantik kandidat crossover saat resume mid-phase [WORKAROUND]
- `_get_crossover_candidates()` melihat cross#1 (round 2) sudah ada di pool → beralih ke mode "subsequent crossover" (parent = mutation + crossover), padahal group disiapkan dari "original + mutation" saat fase dimulai. Resume naif = parent berbeda dari protokol run asli.
- Workaround: `backend/runs/resume_src_kv_and_text_5of6/` = pool TANPA cross#1 + state buatan (`round=2, phase=crossover, crossover_idx=1`) → kandidat kembali "first crossover", group deterministik (strategy `best`), group[1] = parent yang sama dengan task yang OOM (`4846e5bb4d94` + `f3c0cd497d01`).
- ⚠️ Konsekuensi: pool hasil resume TIDAK memuat cross#1 (`e233b041bce8`). **Pool final kv_and_text = gabungan** run asli (5 traj) + cross#2 dari run resume. Lihat §4 langkah merge.

### B5. Run `text` mati diam-diam + `evolution_state.json` 0 byte
- `save_state()` hanya dipanggil SEKALI di akhir `run_evolution_loop` (factor_mining.py:907). Proses mati di tengah → state kosong/hilang, resume butuh state buatan tangan.
- Kronologi terverifikasi: factor_logs crossover#2 tersimpan 11:04:29 → proses masuk backtest gabungan LightGBM → mati TANPA output apa pun; mode kv mulai 11:51:03 (= saat python text exit). Durasi backtest ≥47 mnt vs normal 10–20 mnt → indikasi HANG lalu SIGKILL (manual/cgroup). `dmesg`/`journalctl` tak tersedia di kontainer (T4 buntu dari sini); resume text (T3) mengulang backtest ini di bawah watchdog — bila hang lagi, debug live (py-spy).

### B6. Exit code menyesatkan: task gagal (OOM) tapi batch mencatat `rc=0`
- `run_evolution_loop` menangkap exception per-task (`Task failed: ...`) lalu lanjut → proses exit 0 → summary log "END rc=0" walau 1/6 task gagal total. Sulit membedakan run sehat vs cacat dari rc. TODO T7: propagasi status task-failure ke exit code / summary.

### B7. Inkonsistensi config: `latent.steps: 60` vs komentar "disamakan ke 10 mengikuti hasil /try latent_steps_sweep"
- Sweep lama + diagnosis promptbench: ls rendah (10–20) menjaga fidelity simbolik; ls60 pernah memunculkan glitch/repetisi. Nilai aktif sekarang 60 di SEMUA mode (di text tak terpakai). Bukan bug runtime, tapi keputusan eksperimen yang perlu disadari saat menulis Bab 4 — dan kandidat variabel bila kv mode collapse total.

### B9. Kualitas output judger/crossover: duplikat & ternary non-boolean [OBSERVASI]
- Resume crossover#2 kv_and_text (00:02): 6 ekspresi = 3 unik × 2 (duplikat persis, `['A','B','C','A','B','C']`) — parser tidak dedup di titik parse (dedup baru di `_build_experiment`/corr-gate, membuang jatah `factors_per_hypothesis`).
- Pola meragukan berulang: `TS_ZSCORE(x,10) ? y : 0` (z-score kontinu dipakai sebagai kondisi boolean) dan threshold keras (`TS_MEAN($volume,10) < 500000`) yang tidak cross-sectionally comparable. Kandidat penyebab correlation-gate sering membuang mayoritas kandidat (temuan sesi lalu: 4/5 dibuang duplikat).
- Sisi positif: resume crossover#2 SUKSES parse dalam 83s tanpa degenerasi (parent sama dengan attempt yang dulu OOM) → degenerasi B2 stokastik, bukan deterministik per-input.

### B10. Mode `kv`: collapse construct REPRODUSIBEL (2/2 run terpisah) [OBSERVASI KUNCI Bab 4]
- Run kv 2026-07-06 01:05 (fresh process, expandable_segments): construct PERTAMA kembali degenerate ("selection of selection selection sele...") lalu retry — identik dengan DIAG-kvdegen 2026-07-05 12:06.
- Kontras terkontrol: di kv_and_text task original construct hampir selalu sehat (mean 21s), di kv construct rusak sejak task pertama. Delta tunggal = ada/tidaknya TEKS agen hulu di KV (no-crop) sebagai jangkar distribusi. Di kv murni, KV = prompt + 60×N virtual token laten → emisi teks pertama runtuh jadi repetisi.
- Implikasi skripsi: komunikasi KV laten murni pada Qwen3-4B TIDAK stabil untuk emisi simbolik — konsisten dgn temuan promptbench (payload simbolik lossy di hop laten) tapi lebih keras: bahkan gist emission collapse. Run tetap dibiarkan selesai sebagai data.

### B11. Jalur `kv_only`: turn asisten MENGGANTUNG di KV (cacat higienis template) [DITEMUKAN 2026-07-06, BELUM DIPERBAIKI]
- `client.py` kv_only branch (≈:1456) memanggil `latent_pass(messages, past_kv)` TANPA `add_generation_prompt=False` → default `True` → KV berisi `<|im_start|>assistant\n` + 60 vektor laten, dan turn TIDAK PERNAH ditutup (tidak ada `_close_open_turn` di branch kv_only — hanya ada di branch kv_and_text no-crop :1536).
- Akibat di comm_mode="kv": prompt agen berikutnya (system/user block) ter-append DI DALAM turn asisten yang menggantung, berulang tiap hop → struktur chat-template yang tak pernah dilihat model saat training, DI ATAS konten laten yang sudah OOD.
- Ini pembeda mekanis nyata vs kv_and_text (yang menutup turn dengan `<|im_end|>\n` dan punya teks nyata di turn asisten).
- **FIX DITERAPKAN (2026-07-06 ~02:15, perintah user "perbaiki dulu")**: `add_generation_prompt=False` di branch kv_only `client.py::run` — kv_only kini = kv_and_text minus langkah generate, sesuai desain eksperimen ("satu-satunya perbedaan = siapa yang emit teks"). Uncommitted → T10.
- Data pembanding "sebelum fix" terarsip: `DIAG-kvdegen_` (tanpa cap), `DIAG2-kvdegen-nocap_` (tanpa cap, take-1 05/07), `DIAG3-kvdegen-danglingturn_` (cap 300s, 3 loop no-expression berturut).
- **HASIL UJI (take-3, 02:18)**: construct PERTAMA langsung sehat — hypo_len=73, n_expr=6, ekspresi beragam tanpa duplikat, repaired=False, gate_error=none. Pra-fix: collapse 2/2 run + 3 loop berturut. → **Collapse construct mode kv = ARTEFAK dangling assistant turn (B11), BUKAN bukti KV laten inheren merusak.** Narasi Bab 4 harus direvisi dari B10: dengan template well-formed, handoff laten murni BISA menghasilkan output simbolik parseable di 4B (minimal pada task awal; degenerasi pada KV panjang/late-crossover masih mungkin — pantau sisa run).

### B12. Gangguan I/O volume network mematikan task via RUNLOG [FIXED — uncommitted]
- Insiden 2026-07-06 03:04 (run kv take-3): volume MooseFS runpod flaky sesaat → `OSError [Errno 5]` di `runlog.py::log` (`self._run_log.write`) → file handle rusak permanen → 3 task terakhir (mutation dir1 + crossover×2) mati BERUNTUN oleh baris logging, bukan oleh LLM/backtest. rc tetap 0 (B6!).
- Fix: `runlog.py::_safe_write` — tulis best-effort: gagal → reopen handle sekali → masih gagal → buang baris + peringatan stderr. Logging tak pernah lagi melempar ke pipeline.
- Volume yang sama juga penyebab kelambatan fs ~02:00–03:05 (tail/grep timeout). Kandidat penyebab kematian run text kemarin (B5) — pola konsisten: proses mati tanpa jejak di volume network.
- Sisa run kv (3 task) di-resume via `resume_src_kv_3of6/` (pool 3 traj apa adanya; dedup mutation bawaan `_get_mutation_task` melewatkan target yang sudah jadi; dry-run terverifikasi: mutation[a1706537aec6] → crossover×2 → selesai).

### B13. Mode kv pasca-fix B11: degenerasi jadi EPISODIK + mode gagal baru "output super-pendek" [OBSERVASI]
- Pasca-fix B11, 5 dari 6 task kv menghasilkan 5-6 ekspresi valid (termasuk KEDUA crossover full-chain). Yang gagal hanya mutation dir1 (`5228ed3135cf`): construct unparseable 5/5 loop → trajectory KOSONG (n_factors=0, tanpa metrik) tetap masuk pool.
- Mode gagalnya BEDA dari rambling: output nyaris kosong ("**.", "** ") atau repetisi pendek ("the same, the same, ..."), berhenti sendiri <300s. Terjadi di jalur re-entry mutation (guidance_kv → proposal → design → construct = rantai laten terdalam). Mutation dir0 dengan jalur sama SUKSES → stokastik/parent-specific.
- Catatan analisis: trajectory kosong ikut jadi kandidat crossover (parent teks tanpa faktor) — crossover tetap sukses. Pertimbangkan filter trajectory kosong dari pool/parent-selection (TODO T14).

### B14. Mode kv: DAUR-ULANG faktor lintas-trajectory (diversity collapse) [TEMUAN 2026-07-06, sesi analisis]
- Verifikasi pool kv: nilai per-factor IC IDENTIK berulang lintas trajectory — {−0.0201, −0.0220, −0.0049} muncul di original `13cf02744ae3` DAN crossover `1f446c8631d0`; −0.0220 dobel dalam `e94b4773b111`. Crossover kv mendaur-ulang faktor original, bukan menghasilkan kombinasi baru.
- Hipotesis mekanis: `latent_pass` = forward pass deterministik (TANPA sampling); temperature hanya bekerja saat emisi teks. Di mode text, SEMUA agen men-sample teks → keragaman ide tiap task. Di mode kv, proposal/design deterministik → dengan direction sama, "pikiran laten" nyaris identik antar-task → construct konvergen ke faktor yang sama. Evolutionary search butuh VARIANS; jalur laten murni menekan sumber varians utamanya.
- Relevan utk Bab 4 (penjelasan mekanis kenapa kv terburuk utk mining evolusioner) dan utk desain perbaikan (injeksi noise/sampling di latent rollout — lihat keputusan sesi analisis 2026-07-06).

### B8. (Konteks lama, masih relevan) RankIC/IC combined LightGBM tercemar 4 fitur baseline
- `conf_combined_factors.yaml` masih mencampur baseline; metrik keputusan sudah dialihkan ke FactorIC_mean (commit f2d881a + 2158daa warning fallback). Perbandingan antar-mode WAJIB pakai FactorIC_mean / per-factor RankIC OOS.

---

## 3. TODO (prioritas)

- [x] T1. Resume kv_and_text crossover#2 — **SELESAI 2026-07-06 00:10**: trajectory `92db70581b71` (dir=1, 4 faktor unik dari 6 ekspresi, FactorIC_mean=−0.0019), front-end 83s TANPA degenerasi, backtest 7 mnt normal. rc=0.
- [x] T2. Run penuh mode `kv` (6 task) — **SELESAI 2026-07-06 04:09** (take-3 pasca-fix B11 + resume pasca-ErrIO; lihat §1). 1 trajectory kosong (B13).
- [x] T3. Resume `text` crossover#2 — **SELESAI 2026-07-06 04:31** rc=0 (`c30de8e38aaa`); backtest resume TIDAK hang → menguatkan hipotesis B12 (I/O volume) sebagai penyebab kematian run asli.
- [ ] T4. Investigasi kenapa proses text mati (dmesg/OOM killer?).
- [ ] T5. `controller.save_state()` per-task (bukan hanya di akhir) — supaya crash tidak menghilangkan cursor.
- [ ] T6. Fix leak GPU ~600MB/loop (task selesai → KV/tensor task lama belum dibebaskan penuh; kandidat: `empty_cache()` + lepas referensi trajectory KV di loop).
- [ ] T7. Exit code / summary jujur saat ada task gagal (B6).
- [x] T8. Cap inferensi per-agen — DIPUTUSKAN user & DITERAPKAN: `max_time=300s` di client.py (lihat B2). Uncommitted, masuk daftar T10.
- [x] T9. Merge pool kv_and_text → `prod_kv_and_text_.../trajectory_pool_MERGED.json` (6 trajektori: 2 orig + 2 mut + 2 cross). PAKAI FILE INI untuk analisis kv_and_text.
- [ ] T10. Commit: fix `load_state` (B3) + `runner.py` uncommitted ("All" data + factor_values gzip OOS — fix FactorIC_mean None + kuota disk) + configs `experiment_{mode}.yaml` + skrip batch.
- [ ] T11. Harness perbandingan Alpha158 (blocking skripsi, belum ada kode — dari sesi sebelumnya).
- [x] T13. Fix B11 diterapkan LANGSUNG (keputusan user 2026-07-06) — run kv take-3 = uji hidupnya. DIAG1-3 = data "sebelum fix" untuk Bab 4. Sisa opsional: ls 60→10-20 (B7) bila take-3 masih sering degen.
- [ ] T14. Filter trajectory kosong (n_factors=0) dari pool / parent selection (B13).
- [ ] T12. Analisis akhir KV vs TEXT dari FactorIC_mean per mode 6/6 + tulis ke Bab 4.

## 4. MERGE POOL kv_and_text (setelah resume selesai)

```python
# pool final = 5 traj run asli + cross#2 dari run resume
import json
a = json.load(open('backend/runs/prod_kv_and_text_2026-07-05_09-36-48/trajectory_pool.json'))
b = json.load(open('backend/runs/prod_kv_and_text_2026-07-05_09-36-48-resume/trajectory_pool.json'))
new = {k: v for k, v in b['trajectories'].items() if k not in a['trajectories']
       and v.get('phase') == 'crossover'}
a['trajectories'].update(new)  # + update by_direction/by_phase sesuai isi new
json.dump(a, open('backend/runs/prod_kv_and_text_2026-07-05_09-36-48/trajectory_pool_MERGED.json','w'), indent=2)
```

## 5. LOG KEPUTUSAN SESI INI (2026-07-06, Claude)

1. `controller.py::load_state` dipatch (B3) — perubahan kode satu-satunya, uncommitted.
2. `resume_src_kv_and_text_5of6/` dibuat (pool minus cross#1 + state buatan); dry-run tanpa GPU memverifikasi task yang dijadwalkan = crossover round2 dir1 parent `[4846e5bb4d94, f3c0cd497d01]`, lalu evolusi selesai.
3. Resume dijalankan dengan `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; run asli TIDAK dimodifikasi (state/pool asli utuh).
4. Watchdog background: deteksi OOM / "exceeded maximum length" / stall >55 mnt.
5. Urutan berikut: T2 (kv penuh) → T3 (resume text) → T9/T12 (merge + analisis).
