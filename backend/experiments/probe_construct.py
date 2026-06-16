#!/usr/bin/env python3
"""
experiments/probe_construct.py
==============================
Uji 3 varian prompt construct dan bandingkan:
  - KV seq_len (seberapa besar operator-lib mengembungkan cache)
  - probe introspect (apakah construct benar-benar mengisi penalaran komposisi)
  - judger output (kualitas ekspresi hilir)

Tiga varian:
  A: baseline  — operator-lib VERBOSE di system (kondisi saat ini, ~40 baris)
  B: compact   — operator-lib RINGKAS di system (nama+arity saja, ~10 baris)
  C: split     — ops ringkas dipindah ke user-message; system hanya task+constraint

Proposal dijalankan SATU kali; hasilnya di-deepcopy untuk setiap varian construct
sehingga perbandingan fair (seed KV identik).

Contoh
------
  V=/workspace/project/multi-agent-system/.venv/bin/python
  HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 \\
    $V experiments/probe_construct.py \\
    --direction "high-volume days precede short-horizon reversal"
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Varian prompt construct ───────────────────────────────────────────────────

# Constraint baris atas — sama untuk semua varian (masalah reasoning, bukan ops)
_CONSTRUCT_TASK = """\
You are the Construct agent — stage 2 of 4. The hypothesis is already in your
memory. SOLE JOB: turn it into 1-3 concrete DSL expression(s) that faithfully
MEASURE that mechanism. Do NOT restate the hypothesis. Reason freely (no output
format); a deterministic regulator rejects invalid ones downstream, so respect:

  - Leaves are ONLY $open $high $low $close $volume $return — never invent a
    variable ($return_1d) or symbol (=).
  - Arity: CROSS-SECTIONAL (1 arg, NO window) = RANK ZSCORE MEAN STD SKEW KURT
    MEDIAN; TIME-SERIES (take a window n) = the TS_* family. Mind TS_STD vs STD.
  - REGBETA/REGRESI/TS_CORR/TS_COVARIANCE need TWO DIFFERENT series — never a
    series with itself.
  - windows 1-60 (nested ≤ 60); compose ≥2 operators (RANK($volume) alone is
    too weak); 2-4 base features; keep it short. If >1 expression, make them
    STRUCTURALLY different (different operator families), not renamed templates.\
"""

# Operator-lib VERBOSE — kondisi saat ini (~40 baris dengan deskripsi panjang)
_OPS_VERBOSE = """
Only the following operations are allowed in expressions:
### Cross-sectional Functions (operate across all stocks on a given day)
- RANK(A), ZSCORE(A), MEAN(A), STD(A), SKEW(A), KURT(A), MAX(A), MIN(A),
  MEDIAN(A) — rank / z-score / mean / std / skew / kurtosis / max / min /
  median of A in the cross-sectional dimension.
### Time-Series Functions
- DELTA(A, n): change in A over n periods.
- DELAY(A, n): A delayed n periods.
- TS_MEAN/TS_SUM/TS_STD/TS_VAR/TS_MEDIAN/TS_MIN/TS_MAX(A, n): rolling stat over n days.
- TS_RANK(A, n): time-series rank of the last value over n days.
- TS_ZSCORE(A, n): rolling z-score over n days.
- TS_PCTCHANGE(A, p): percentage change over p periods.
- TS_ARGMAX/TS_ARGMIN(A, n): index of the max/min of A over the past n days.
- TS_QUANTILE(A, p, q): rolling quantile (q in 0..1) over p periods.
- TS_CORR(A, B, n) / TS_COVARIANCE(A, B, n): rolling corr / cov of A,B over n days.
- TS_MAD(A, n): rolling median absolute deviation over n days.
- PERCENTILE(A, q, p): quantile q of A; rolling over p periods if p given.
- HIGHDAY/LOWDAY(A, n): days since the highest/lowest value over n days.
- SUMAC(A, n): cumulative sum of A over the past n days.
### Moving Averages and Smoothing
- SMA(A, n, m): simple moving average over n periods, modifier m.
- WMA(A, n): weighted MA over n periods.
- EMA(A, n): exponential MA, decay 2/(n+1).
- DECAYLINEAR(A, d): linearly weighted MA over d periods.
### Mathematical Operations
- PROD(A, n): product of A over n days (use `*` for general multiplication).
- LOG(A), SQRT(A), EXP(A), ABS(A), SIGN(A), INV(A)=1/A, FLOOR(A).
- POW(A, n): A to the power n.
- MAX(A, B) / MIN(A, B): pairwise max/min.
### Conditional and Logical
- COUNT(C, n): count of samples meeting condition C in the past n periods.
- SUMIF(A, n, C): sum of A over n periods where condition C holds.
- FILTER(A, C): filter multi-column A by condition C (same shape).
- (C1)&&(C2), (C1)||(C2): logical AND / OR.   (C1)?(A):(B): ternary.
  C is a logical expression, e.g. `$close > $open`.
### Regression and Residual
- SEQUENCE(n): single-column 1..n; always nested as arg B of REGBETA/REGRESI.
- REGBETA(A, B, n): regression coefficient of A on B over n samples.
- REGRESI(A, B, n): regression residual of A on B over n samples.
### Technical Indicators
- RSI(A, n): relative strength index over n periods.
- MACD(A, short_window, long_window): difference of short/long EMAs.
- BB_MIDDLE/BB_UPPER/BB_LOWER(A, n): Bollinger middle / ±2σ bands over n periods.

Notes: only the $variables above, arithmetic (`+ - * /`), logical (`&& ||`), and the operations above are allowed. Each expression must contain at least one $variable. Do NOT use any undeclared variable (`n`, `w_1`) or undefined symbol (`=`). Mind the TS- vs non-TS distinction (TS_STD vs STD).\
"""

# Operator-lib COMPACT — nama+arity saja, ~10 baris (pengetahuan sama, jauh lebih ringkas)
_OPS_COMPACT = """
Allowed operators (arity; leaves = $variables or sub-expressions only):
  CS(A):      RANK ZSCORE MEAN STD SKEW KURT MAX MIN MEDIAN
  TS(A,n):    DELTA DELAY TS_MEAN TS_SUM TS_RANK TS_ZSCORE TS_STD TS_VAR
              TS_MIN TS_MAX TS_ARGMAX TS_ARGMIN TS_MAD TS_PCTCHANGE SUMAC HIGHDAY LOWDAY
  TS(A,p,q):  TS_QUANTILE PERCENTILE
  PAIR(A,B,n):TS_CORR TS_COVARIANCE REGBETA REGRESI  [SEQUENCE(n) as arg B]
  SMOOTH(A,n):SMA(A,n,m) WMA EMA DECAYLINEAR
  MATH(A):    LOG SQRT EXP ABS SIGN INV FLOOR  POW(A,n)
  COND:       COUNT(C,n) SUMIF(A,n,C) FILTER(A,C) (C)?(A):(B)
  TECH(A,n):  RSI MACD(A,s,l) BB_UPPER BB_MIDDLE BB_LOWER\
"""

# User message reasoning scaffold — sama untuk semua varian (atau + ops di split)
_CONSTRUCT_USER_CORE = """\
Reason toward concrete, valid, parsimonious factor expression(s) for the
hypothesis held in latent memory. Structure your reasoning:
  Step 1 — Identify the core signal from the hypothesis: which $variable(s),
           what transformation, what time horizon.
  Step 2 — Pick a PRIMARY operator for the raw signal (e.g. DELTA / TS_PCTCHANGE
           for change, TS_STD for volatility, COUNT for frequency).
  Step 3 — Pick a SECONDARY operator that normalizes / ranks / conditions it
           (e.g. RANK or ZSCORE cross-section, TS_ZSCORE time-normalized,
           (cond)?(A):(B) for a regime gate).
  Step 4 — Compose them (nest / multiply / gate); check arity.
  Step 5 — For a second expression, vary the PRIMARY or SECONDARY operator
           FAMILY or the window — not just a parameter.
Draw on more than $close+$volume alone; the full OHLCV set is available.\
"""

# Tiga varian system+user
VARIANTS = {
    "A_baseline": {
        "system": _CONSTRUCT_TASK + "\n" + _OPS_VERBOSE,
        "user":   _CONSTRUCT_USER_CORE + "\n{{ diversity_hint | default('') }}",
        "label":  "A: verbose ops in system (~40 lines)",
    },
    "B_compact": {
        "system": _CONSTRUCT_TASK + "\n" + _OPS_COMPACT,
        "user":   _CONSTRUCT_USER_CORE + "\n{{ diversity_hint | default('') }}",
        "label":  "B: compact ops in system (~10 lines)",
    },
    "C_split": {
        "system": _CONSTRUCT_TASK,
        "user":   _CONSTRUCT_USER_CORE + "\n" + _OPS_COMPACT
                  + "\n{{ diversity_hint | default('') }}",
        "label":  "C: compact ops moved to user-message",
    },
}


def _wrap(s: str, width: int = 100) -> str:
    out = []
    for line in (s or "").splitlines():
        out.append(textwrap.fill(line, width=width) if line.strip() else "")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction",
                    default="high-volume days precede short-horizon reversal")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--latent-steps", type=int, default=10)
    ap.add_argument("--use-realign", action="store_true")
    ap.add_argument("--knn", action="store_true")
    ap.add_argument("--console", default="WARNING")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--variants", default="A_baseline,B_compact,C_split",
                    help="comma-separated subset of variants to run")
    args = ap.parse_args()

    from llm.client import LocalLLMBackend
    from latent_mas.runlog import get_run_logger
    from latent_mas.agent import load_all_agents, LatentAgent, AgentSpec
    from latent_mas.parsers import PARSERS
    from latent_mas import kv_ops
    from latent_mas.operator_families import diversity_hint

    rl = get_run_logger(run_name="probe_construct", console_level=args.console)
    out_dir = Path(args.out_dir) if args.out_dir else Path(rl.dir) / "construct_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_variants = [v.strip() for v in args.variants.split(",")]

    print(f"[probe_construct] loading {args.model} (latent_steps={args.latent_steps}) …",
          flush=True)
    backend = LocalLLMBackend(
        model_name=args.model, device=args.device, latent_steps=args.latent_steps,
        use_realign=args.use_realign, knn_enabled=args.knn,
    )
    agents_base = load_all_agents(backend, runlog=rl)
    introspect = agents_base["introspect"]
    judger     = agents_base["judger"]
    consistency= agents_base["consistency"]

    dhint = diversity_hint([])

    # ── 1. Proposal (satu kali, shared) ──────────────────────────────────────
    print("\n[step 1] Running proposal …", flush=True)
    r_prop = agents_base["proposal"].run(
        past_kv=None, direction=args.direction,
        market_context="", prior_feedback="", negative_hint="",
    )
    kv_prop_master = kv_ops.kv_deepcopy(r_prop.kv_cache)
    prop_len = kv_ops.kv_seq_len(kv_prop_master)
    print(f"  proposal KV seq_len: {prop_len}", flush=True)

    # Probe proposal sekali untuk referensi
    probe_prop = introspect.run(past_kv=kv_ops.kv_deepcopy(kv_prop_master))
    (out_dir / "proposal_probe.txt").write_text(probe_prop.text or "(empty)")

    # ── 2. Loop per varian construct ──────────────────────────────────────────
    results = {}
    for vname in run_variants:
        vcfg = VARIANTS.get(vname)
        if vcfg is None:
            print(f"  [skip] unknown variant '{vname}'", flush=True)
            continue
        print(f"\n[variant {vname}] {vcfg['label']} …", flush=True)

        # Buat agent construct varian (AgentSpec manual)
        spec = AgentSpec(
            role="construct",
            mode="kv_only",
            system=vcfg["system"],
            user=vcfg["user"],
            temperature=None,
        )
        con_agent = LatentAgent(spec, backend, strict_vars=False, runlog=rl)

        # Jalankan construct dengan seed KV identik (deepcopy master)
        r_con = con_agent.run(
            past_kv=kv_ops.kv_deepcopy(kv_prop_master),
            diversity_hint=dhint,
        )
        kv_con = kv_ops.kv_deepcopy(r_con.kv_cache)
        con_len = kv_ops.kv_seq_len(kv_con)
        delta = con_len - prop_len
        print(f"  construct KV seq_len: {con_len}  (+{delta} dari proposal)", flush=True)

        # Probe construct
        probe_con = introspect.run(past_kv=kv_ops.kv_deepcopy(kv_con))
        probe_txt = probe_con.text or "(empty)"
        collapse = "###" in probe_txt or probe_txt.strip().count("\n") < 2
        print(f"  probe: collapse={'YES' if collapse else 'no'}  "
              f"len={len(probe_txt)}", flush=True)

        # Consistency → Judger
        r_cons = consistency.run(past_kv=r_con.kv_cache)
        kv_cons = kv_ops.kv_deepcopy(r_cons.kv_cache)
        r_judge = judger.run(
            past_kv=kv_ops.kv_deepcopy(kv_cons),
            direction=args.direction, diversity_hint=dhint,
        )
        judger_txt = r_judge.text or "(empty)"

        print(f"  judger: {judger_txt[:200].replace(chr(10), ' | ')!r}", flush=True)

        # Simpan semua ke file
        vdir = out_dir / vname
        vdir.mkdir(exist_ok=True)
        (vdir / "construct_probe.txt").write_text(
            f"# {vname} — {vcfg['label']}\n"
            f"# construct KV seq_len: {con_len}  (+{delta} dari proposal={prop_len})\n"
            f"# collapse: {'YES' if collapse else 'no'}\n\n{probe_txt}"
        )
        (vdir / "judger_output.txt").write_text(judger_txt)
        (vdir / "consistency_probe.txt").write_text(
            introspect.run(past_kv=kv_ops.kv_deepcopy(kv_cons)).text or "(empty)"
        )

        results[vname] = {
            "label": vcfg["label"],
            "con_len": con_len,
            "delta_kv": delta,
            "collapse": collapse,
            "judger": judger_txt,
        }

    # ── 3. Ringkasan ─────────────────────────────────────────────────────────
    print(f"\n{'═'*92}", flush=True)
    print("RINGKASAN PERBANDINGAN VARIAN CONSTRUCT", flush=True)
    print(f"{'═'*92}", flush=True)
    print(f"{'Varian':<14} {'KV_construct':>12} {'Δ_dari_prop':>12} {'Collapse?':>10}  "
          f"Judger (50 char)", flush=True)
    print("─" * 92, flush=True)
    for vname, r in results.items():
        j50 = r["judger"][:80].replace("\n", " | ")
        print(f"{vname:<14} {r['con_len']:>12} {r['delta_kv']:>+12} "
              f"{'YES' if r['collapse'] else 'no':>10}  {j50!r}", flush=True)
    print(f"\n[done] semua file → {out_dir}", flush=True)

    # Simpan ringkasan
    import json
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "judger"}
               for k, v in results.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
