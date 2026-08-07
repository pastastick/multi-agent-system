"""Harness front-end: proposal → design → construct → gate, TANPA backtest Qlib.

Kenapa ada. Satu trajectory produksi ≈ 13 menit, mayoritasnya backtest LightGBM
gabungan — metrik yang oleh AUDIT_KRITIS §S3/B8 justru dinyatakan TIDAK boleh
dipakai membandingkan mode. Metrik yang jujur (per-factor RankIC OOS) dihitung
`lab/core.py` di CPU dan sudah divalidasi identik 7 desimal terhadap produksi.
Jadi: jalankan bagian yang butuh GPU (front-end LLM) saja, lalu skor ekspresinya
di CPU. Ini yang membuat G2/G3/G4/G5 muat dalam anggaran GPU yang wajar.

Satu "run" = satu (arah × seed) → satu FrontEndOutput. Yang direkam:
  - hipotesis, faktor mentah construct, gate_log per-ekspresi, repair
  - waktu per agen + panjang KV (deteksi degenerasi/rambling)
  - untuk tiap ekspresi: cacat semantik (validate_semantics + static_flags),
    lalu IC/ICIR/t/n_unique OOS dari lab.core

Pemakaian:
    python lab/frontend_probe.py --comm-mode kv --latent-steps 60 \
        --seeds 0,1,2 --tag g2_ls60
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

QL = Path(__file__).resolve().parent.parent
BACKEND = QL / "backend"
# Buang direktori skrip (lab/) dari sys.path: `lab/core.py` di sana akan
# MEMBAYANGI paket `backend/core` dan membuat
# `factors.regulator → from core.evaluation import Evaluator` gagal.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
for p in (str(QL), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = QL / "lab" / "out"

# Dua arah eksplorasi yang BENAR-BENAR dipakai batch produksi 2026-07-05
# (backend/runs/prod_text_.../stdout.log baris "Direction 0/1"). Dipakai ulang
# supaya hasil sebanding; planning LLM sengaja tidak dijalankan agar variansnya
# tidak mencemari perbandingan antar-lengan.
DIRECTIONS = {
    "d0": "short-term reversal after abnormally high-volume days in small-cap stocks",
    "d1": "mean-reversion in low-volatility stocks during regime transitions",
}


# ── runlog stub: kumpulkan event tanpa menulis ke volume network ─────────────
class Collector:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def _rec(self, level: str, msg: str, **f):
        self.events.append({"level": level, "msg": msg, **{k: str(v)[:400] for k, v in f.items()}})

    def info(self, msg, **f): self._rec("INFO", msg, **f)
    def warn(self, msg, **f): self._rec("WARNING", msg, **f)
    def error(self, msg, **f): self._rec("ERROR", msg, **f)
    def event(self, kind, **f): self._rec("EVENT", kind, **f)

    def step(self, name, **ctx):
        import contextlib
        return contextlib.nullcontext()


def repetition_ratio(text: str) -> float:
    """Fraksi token yang merupakan pengulangan token sebelumnya (deteksi
    degenerasi 'selection selection selection...' — B2/B10)."""
    w = re.findall(r"\w+", (text or "").lower())
    if len(w) < 20:
        return 0.0
    return 1.0 - len(set(w)) / len(w)


def instrument(pipeline, collector, keep_text: int = 6000):
    """Bungkus tiap LatentAgent.run agar per-agen tercatat (durasi, panjang KV,
    panjang teks, rasio repetisi) tanpa mengubah kode produksi.

    `n_in_tok` dicatat karena sumbu A6 (biaya per faktor diterima) butuh TOKEN
    DIPROSES, bukan hanya token yang di-emit; tanpa ini biaya lengan `text`
    (prompt panjang, KV nol) tak bisa dibandingkan adil dengan lengan `kv`.

    `text` (dipotong `keep_text` char) dicatat karena sumbu A7 butuh membaca
    keluaran ASLI agen hulu: kepatuhan palette hanya bisa dihitung bila palette
    design tersimpan. Nol biaya GPU, ~5 KB per run.
    """
    trace: list[dict] = []
    for name, agent in pipeline.agents.items():
        orig = agent.run

        def wrapped(_orig=orig, _name=name, **kw):
            t0 = time.time()
            res = _orig(**kw)
            trace.append({
                "agent": _name, "mode": res.mode,
                "s": round(time.time() - t0, 2),
                "latent_s": res.latent_s, "gen_s": res.gen_s,
                # B6: anggaran vs langkah yang benar-benar berjalan.
                "n_latent_steps": getattr(res, "n_latent_steps", 0),
                "latent_stop": getattr(res, "latent_stop", "off"),
                "kv_len": res.kv_seq_len, "n_out_tok": res.n_output_tokens,
                "n_in_tok": res.n_input_tokens,
                "text_len": len(res.text or ""),
                "rep_ratio": round(repetition_ratio(res.text or ""), 3),
                "parsed_ok": res.parsed is not None,
                "text": (res.text or "")[:keep_text],
            })
            return res

        agent.run = wrapped
    return trace


def build_backend(args):
    from llm.client import LocalLLMBackend
    return LocalLLMBackend(
        model_name=args.model,
        device="cuda",
        latent_steps=args.latent_steps,
        use_realign=not args.no_realign,
        enable_thinking=False,
        log_tensors=False,
        store_kv=False,
        output_log_dir=str(QL / "lab" / "out" / "llm_outputs" / (args.tag or "probe")),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=0.95,
        knn_enabled=False,          # auto-disabled saat latent_steps>0 (client.py)
        # B6. Default None → engine memakai env/0,999. Lengan yang ingin
        # mereplikasi baseline pra-B6 memberi 1.0 (mematikan early-stop).
        latent_early_stop_cos=getattr(args, "early_stop_cos", None),
    )


def run_once(backend, args, direction: str, seed: int, prompts_path: Path):
    import torch
    from latent_mas.agent import load_all_agents
    from latent_mas.pipeline import FrontEndPipeline

    torch.manual_seed(seed)
    col = Collector()
    agents = load_all_agents(backend, runlog=col, path=prompts_path)
    chain = getattr(args, "chain", None)
    if isinstance(chain, str):
        chain = tuple(c.strip() for c in chain.split(",") if c.strip()) or None
    pipe = FrontEndPipeline(backend, runlog=col, agents=agents,
                            use_regulator=True, comm_mode=args.comm_mode,
                            max_repair_attempts=args.max_repair,
                            chain=chain,
                            free_form=getattr(args, "free_form", None))
    trace = instrument(pipe, col)

    t0 = time.time()
    err = None
    try:
        fe = pipe.run(direction=direction)
    except Exception:                                # noqa: BLE001
        err = traceback.format_exc()[-2000:]
        fe = None
    dur = round(time.time() - t0, 2)

    if fe is None:
        return {"error": err, "duration_s": dur, "agent_trace": trace,
                "chain": ",".join(pipe.chain), "free_form": pipe.free_form,
                "events": col.events}

    return {
        "duration_s": dur,
        "chain": ",".join(pipe.chain),
        "free_form": pipe.free_form,
        "hypothesis": fe.hypothesis,
        "factors": fe.factors,
        "passing": fe.expressions,
        "gate_log": fe.gate_log,
        "repaired": fe.repaired,
        "repair_attempts": fe.repair_attempts,
        "gate_error": fe.gate_error,
        "construct_text_len": len(fe.judger_text or ""),
        "construct_rep_ratio": round(repetition_ratio(fe.judger_text or ""), 3),
        "construct_text_head": (fe.judger_text or "")[:600],
        "agent_trace": trace,
        "events": [e for e in col.events if e["level"] in ("WARNING", "ERROR")],
    }


# ── skoring CPU ─────────────────────────────────────────────────────────────
class _time_budget:
    """Batas waktu per-ekspresi. Tanpa ini satu operator lambat menyandera
    seluruh sweep: `TS_QUANTILE` rolling dan `REGBETA`/`REGRESI` (joblib
    per-instrumen) bisa memakan belasan menit untuk SATU ekspresi, dan sweep
    yang macet 11 menit di satu faktor terlihat seperti GPU yang menggantung.
    Ekspresi yang lewat batas ditandai `eval_error='timeout'` — dilaporkan apa
    adanya, bukan disamarkan jadi faktor tanpa IC."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        import signal

        def _raise(signum, frame):  # noqa: ARG001
            raise TimeoutError(f"melebihi {self.seconds}s")

        try:
            self._old = signal.signal(signal.SIGALRM, _raise)
            signal.alarm(self.seconds)
        except ValueError:
            self._old = None
        return self

    def __exit__(self, *exc):
        import signal

        if self._old is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        return False


def score_expressions(runs: list[dict], window=None, series_path: Path | None = None,
                      budget_s: int = 90) -> None:
    """Isi setiap faktor dengan cacat semantik + IC/ICIR OOS (in-place).

    Deret IC harian juga disimpan (parquet) supaya analisis bisa mengelompokkan
    faktor jadi KLASTER SINYAL — ukuran keragaman pencarian (AUDIT_KRITIS §2.4)
    yang jauh lebih informatif daripada sekadar jumlah ekspresi.
    """
    import pandas as pd

    from lab.core import Lab
    from lab.audit_batch import static_flags
    # Pre-import factor_ast memutus circular import factor_regulator →
    # coder/__init__ → evaluators → factor_regulator (sama seperti yang
    # dilakukan FrontEndPipeline._build_regulator_gate saat import COLD).
    import factors.coder.factor_ast  # noqa: F401
    from factors.regulator.factor_regulator import validate_semantics

    lab = Lab(mode="fast", window=window)
    cache: dict[str, dict] = {}
    series: dict[str, "pd.Series"] = {}

    for r in runs:
        for f in r.get("factors", []) or []:
            e = f.get("expression", "")
            if not e:
                continue
            if e not in cache:
                ok, errs = validate_semantics(e)
                try:
                    with _time_budget(budget_s):
                        res, ser = lab.ic_full(e)
                except TimeoutError:
                    from lab.core import ICResult
                    res, ser = ICResult(None, None, 0, None, 0.0, 0.0,
                                        error=f"timeout>{budget_s}s"), None
                cache[e] = {
                    "sem_ok": bool(ok), "sem_errors": errs,
                    "flags": static_flags(e),
                    "ic": res.ic, "icir": res.icir, "tstat": res.tstat,
                    "n_days": res.n_days, "coverage": res.coverage,
                    "n_unique": res.n_unique, "eval_error": res.error,
                }
                if ser is not None:
                    series[e] = ser
            f.update(cache[e])
            f["passed_gate"] = e in (r.get("passing") or [])

    if series_path is not None and series:
        pd.DataFrame(series).to_parquet(series_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--comm-mode", default="kv", choices=["kv", "kv_and_text", "text"])
    ap.add_argument("--latent-steps", type=int, default=60)
    ap.add_argument("--latent-mode", default="raw",
                    help="raw | gumbel | sample | soft  (G3; via LATENT_STEP_MODE)")
    ap.add_argument("--latent-temp", type=float, default=0.7)
    ap.add_argument("--early-stop-cos", dest="early_stop_cos", type=float,
                    default=None,
                    help="B6: berhenti bila cos(h_k,h_k-1) > nilai ini. "
                         "1.0 = matikan (baseline pra-B6). None = default engine (0,999)")
    ap.add_argument("--no-realign", action="store_true", help="use_realign=False (G6)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--directions", default="d0,d1")
    ap.add_argument("--prompts", default="", help="path prompts.yaml alternatif")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-repair", type=int, default=3)
    ap.add_argument("--holdout", action="store_true",
                    help="skor pada 2022-01-01..2025-12-26 (holdout sejati)")
    ap.add_argument("--chain", default="",
                    help="susunan agen front-end, mis. 'proposal,innovate,construct' "
                         "(kosong = proposal,design,construct)")
    ap.add_argument("--free-form", dest="free_form", default=None,
                    action="store_true",
                    help="lepas klem FIDELITY di construct (default: ikut chain)")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--score-only", action="store_true",
                    help="lewati GPU; skor ulang frontend_<tag>.json yang sudah ada")
    args = ap.parse_args()

    if args.score_only:
        path = OUT / f"frontend_{args.tag}.json"
        doc = json.loads(path.read_text())
        window = ("2022-01-01", "2025-12-26") if args.holdout else None
        score_expressions(doc["runs"], window=window,
                          series_path=OUT / f"icseries_{args.tag}.parquet")
        path.write_text(json.dumps(doc, indent=2, default=str))
        print(f"di-skor ulang → {path}")
        return

    # G3: mode langkah laten diteruskan ke client.py lewat env (patch minimal).
    os.environ["LATENT_STEP_MODE"] = args.latent_mode
    os.environ["LATENT_STEP_TEMP"] = str(args.latent_temp)

    OUT.mkdir(parents=True, exist_ok=True)
    prompts_path = (Path(args.prompts) if args.prompts
                    else BACKEND / "latent_mas" / "prompts.yaml")

    backend = build_backend(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    dirs = [d.strip() for d in args.directions.split(",") if d.strip()]

    runs = []
    for d in dirs:
        for s in seeds:
            print(f"\n=== {args.tag} | dir={d} seed={s} | comm={args.comm_mode} "
                  f"ls={args.latent_steps} latent_mode={args.latent_mode} ===", flush=True)
            r = run_once(backend, args, DIRECTIONS[d], s, prompts_path)
            r.update({"direction": d, "seed": s, "comm_mode": args.comm_mode,
                      "latent_steps": args.latent_steps, "latent_mode": args.latent_mode,
                      "model": args.model, "use_realign": not args.no_realign,
                      "prompts": str(prompts_path)})
            n_fac = len(r.get("factors") or [])
            n_pass = len(r.get("passing") or [])
            print(f"    -> {r['duration_s']}s  n_factors={n_fac} n_pass={n_pass} "
                  f"repaired={r.get('repaired')} err={bool(r.get('error'))}", flush=True)
            runs.append(r)
            # buang KV/tensor sisa antar-run
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()

    # Tulis SEBELUM skoring: kerja GPU tak boleh hilang gara-gara error di
    # tahap CPU (skoring bisa diulang dengan --score-only).
    path = OUT / f"frontend_{args.tag}.json"
    path.write_text(json.dumps({"args": vars(args), "runs": runs}, indent=2, default=str))

    window = ("2022-01-01", "2025-12-26") if args.holdout else None
    print("\n[probe] skoring ekspresi di CPU ...", flush=True)
    score_expressions(runs, window=window,
                      series_path=OUT / f"icseries_{args.tag}.parquet")
    path.write_text(json.dumps({"args": vars(args), "runs": runs}, indent=2, default=str))
    print(f"tersimpan → {path}")

    # ringkasan cepat
    allf = [f for r in runs for f in (r.get("factors") or [])]
    ok = [f for f in allf if f.get("ic") is not None]
    alive = [f for f in ok if (f.get("n_unique") or 0) > 2]
    sem_bad = [f for f in allf if f.get("sem_ok") is False]
    print(f"\n[{args.tag}] runs={len(runs)} ekspresi={len(allf)} "
          f"lolos-gate={sum(1 for f in allf if f.get('passed_gate'))} "
          f"ber-IC={len(ok)} hidup={len(alive)} cacat-semantik={len(sem_bad)}")
    if ok:
        import statistics as st
        print(f"  mean IC={st.mean(f['ic'] for f in ok):+.5f} "
              f"mean |IC|={st.mean(abs(f['ic']) for f in ok):.5f} "
              f"max |IC|={max(abs(f['ic']) for f in ok):.5f} "
              f"IC>0={sum(1 for f in ok if f['ic'] > 0)}/{len(ok)}")


if __name__ == "__main__":
    main()
