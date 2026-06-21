"""
promptbench/scoring/score.py
============================
Phase 0d — scorer DETERMINISTIK untuk output agent. Tidak butuh GPU.

Per-output (satu sample):
  - proposal           : heuristik hipotesis (1 kalimat, observable, bukan klise)
  - construct / judger : parse_hypothesis_exprs → gate FactorRegulator + variety family + arity
  - feedback           : JSON valid + key wajib
  - mutation/crossover : passthrough (panjang, ada DIRECTION/diagnosis)
  - consistency/introspect/repair: passthrough ringan (efek sebenarnya diukur di chain)

Per-konfigurasi (R repetisi → agregat):
  - parse_rate, gate_pass_rate (mean), variety (jumlah family unik lintas rep),
    n_distinct_hypotheses (diversitas), score_mean & score_std (stabilitas).

Gate = `FrontEndPipeline._build_regulator_gate()` PENUH (fallback sintaks bila
modul tak tersedia). families_of dari operator_families.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_BACKEND = Path(__file__).resolve().parent.parent.parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from latent_mas.parsers import parse_hypothesis_exprs           # noqa: E402
from latent_mas.operator_families import families_of            # noqa: E402

# ── gate regulator (build sekali, cache) ────────────────────────────────────
_GATE = None
_GATE_KIND = None


def _gate():
    global _GATE, _GATE_KIND
    if _GATE is None:
        try:
            from latent_mas.pipeline import FrontEndPipeline
            _GATE, reg = FrontEndPipeline._build_regulator_gate()
            _GATE_KIND = "regulator" if reg is not None else "syntax_fallback"
        except Exception as e:  # noqa: BLE001
            _GATE_KIND = f"unavailable:{type(e).__name__}"
            _GATE = lambda expr: (bool(expr and "$" in expr), "syntax-only")
    return _GATE


# ── DSL banned cliché (proposal) ────────────────────────────────────────────
_CLICHE = re.compile(r"low[- ]?volume.*volatilit|volatilit.*spike.*mean[- ]?revers", re.I)
_DOLLAR_COL = re.compile(r"\$(open|high|low|close|volume|return)\b", re.I)

# ── faithfulness (construct): apakah ekspresi MELAYANI mekanisme hipotesis ────
# Tujuan: menghukum "library dump" — construct yang meng-enumerasi seluruh
# palette tanpa mengikat ke hipotesis. DETERMINISTIK, tanpa GPU.
#   - keyword hipotesis -> mekanisme yang dimaksud (_HYP_MECH)
#   - mekanisme -> tanda-tangan operator/variabel yang benar-benar mengukurnya
#     (_MECH_SIG); family struktural (time_series dll.) tak membedakan
#     momentum vs reversal, jadi dicocokkan pada nama fungsi, bukan family.
_HYP_MECH = {
    "momentum":    r"\bmomentum|\btrend|persist|continu\w*|\bdrift|follow[- ]?through",
    "reversal":    r"revers|mean[- ]?revert|overbought|oversold|rebound|over[- ]?react|snap[- ]?back|correct\w*",
    "volatility":  r"volatil|turbulen|\brisk\b|dispersion",
    "volume":      r"volume|liquid|turnover|trading activity|participation|\bflow\b",
    "correlation": r"correlat|covar|co[- ]?move|comove|relationship|lead[- ]?lag|\bbeta\b",
    "seasonal":    r"season|calendar|day[- ]?of|recurr",
}
_MECH_SIG = {
    "momentum":    r"\b(DELTA|TS_MEAN|TS_SUM|SUMAC|EMA|WMA|SMA|DECAYLINEAR|MACD|RSI|TS_PCTCHANGE|TS_RANK|REGBETA|DELAY)\b",
    "reversal":    r"\b(TS_ZSCORE|ZSCORE|RANK|TS_RANK|BB_UPPER|BB_MIDDLE|BB_LOWER|RSI|TS_MIN|TS_MAX|HIGHDAY|LOWDAY|PERCENTILE|TS_QUANTILE|TS_MEDIAN)\b",
    "volatility":  r"\b(TS_STD|TS_VAR|TS_MAD|STD|BB_UPPER|BB_LOWER)\b|\$high|\$low",
    "volume":      r"\$volume|\b(TS_CORR|TS_COVARIANCE|REGBETA|REGRESI|SUMIF|SUMAC)\b",
    "correlation": r"\b(TS_CORR|TS_COVARIANCE|REGBETA|REGRESI)\b",
    "seasonal":    r"\b(TS_ARGMAX|TS_ARGMIN|HIGHDAY|LOWDAY|DELAY|COUNT)\b",
}
# hipotesis menyebut keadaan/regime -> butuh gate kondisional di minimal 1 ekspresi
_COND_HYP = re.compile(
    r"\bwhen\b|\bregime|\bonly\b|condition|gate|during|\bwhile\b|provided|\bif\b|\bstate\b|trending",
    re.I)
_COND_EXPR = re.compile(r"\?|&&|\|\||\bCOUNT\(|\bSUMIF\(|\bFILTER\(", re.I)
_WS = re.compile(r"\s+")


def score_faithfulness(hypothesis: Optional[str], exprs: List[str]) -> Dict[str, Any]:
    """Faithfulness ∈ [0,1] = seberapa setia kumpulan ekspresi pada mekanisme
    hipotesis. Tiga sub-cek (alignment dominan):
      alignment   : fraksi ekspresi yang memuat operator/variabel penanda
                    mekanisme yang disebut hipotesis (anti library-dump).
      conditional : bila hipotesis menyebut keadaan/regime, minimal 1 ekspresi
                    harus punya gate kondisional; jika tak disebut, netral (1.0).
      distinct    : fraksi ekspresi unik (anti duplikat).
    faithfulness = 0.6*alignment + 0.2*conditional + 0.2*distinct.
    """
    n = len(exprs)
    if n == 0:
        return {"faithfulness": 0.0, "mech_detected": [], "alignment_frac": 0.0,
                "conditional_required": False, "conditional_present": False,
                "distinct_frac": 0.0}
    h = hypothesis or ""
    mechs = [m for m, pat in _HYP_MECH.items() if re.search(pat, h, re.I)]
    if mechs:
        sig = re.compile("|".join(_MECH_SIG[m] for m in mechs), re.I)
        on = sum(1 for e in exprs if e and sig.search(e))
        alignment = on / n
    else:
        alignment = 0.5  # mekanisme tak terdeteksi → tak bisa dinilai, netral
    cond_req = bool(_COND_HYP.search(h))
    cond_present = any(e and _COND_EXPR.search(e) for e in exprs)
    conditional = 1.0 if (not cond_req or cond_present) else 0.0
    distinct = len({_WS.sub("", (e or "")).upper() for e in exprs}) / n
    faith = 0.6 * alignment + 0.2 * conditional + 0.2 * distinct
    return {
        "faithfulness": round(faith, 3),
        "mech_detected": mechs,
        "alignment_frac": round(alignment, 3),
        "conditional_required": cond_req,
        "conditional_present": cond_present,
        "distinct_frac": round(distinct, 3),
    }


# ════════════════════════════════════════════════════════════════════════════
# per-output scorers
# ════════════════════════════════════════════════════════════════════════════

def score_expressions(exprs: List[str]) -> Dict[str, Any]:
    gate = _gate()
    fams: set = set()
    per = []
    n_pass = 0
    for e in exprs:
        ok, reason = gate(e)
        f = families_of(e)
        fams |= f
        n_pass += int(ok)
        per.append({"expr": e, "gate_ok": bool(ok), "reason": reason,
                    "families": sorted(f)})
    n = len(exprs) or 1
    return {
        "n_expr": len(exprs),
        "gate_pass": n_pass,
        "gate_pass_frac": round(n_pass / n, 3),
        "families": sorted(fams),
        "n_families": len(fams),
        "per_expr": per,
    }


def score_construct_judger(text: str, parsed: Any = None) -> Dict[str, Any]:
    # `parsed` boleh diinjeksi (mis. hasil parsing_hook dgn fallback) supaya
    # skor Phase B memakai parser yang sama dgn detektor collapse. Default None
    # → parser produksi (perilaku Phase A tak berubah).
    if parsed is None:
        parsed = parse_hypothesis_exprs(text or "")
    if parsed is None:
        return {"parse_ok": False, "hypothesis": None, "n_expr": 0,
                "gate_pass": 0, "gate_pass_frac": 0.0, "n_families": 0,
                "families": [], "score": 0.0}
    ex = score_expressions(parsed.expressions)
    fa = score_faithfulness(parsed.hypothesis, parsed.expressions)
    # skor 0..1: parse(0.2) + gate(0.4*frac) + variety(0.1 bila >=2 family)
    #            + faithfulness(0.3). Faithfulness menghukum library-dump yang
    #            lolos gate tapi tak melayani hipotesis (lihat score_faithfulness).
    score = (0.2
             + 0.4 * ex["gate_pass_frac"]
             + (0.1 if ex["n_families"] >= 2 else 0.0)
             + 0.3 * fa["faithfulness"])
    return {
        "parse_ok": True,
        "hypothesis": parsed.hypothesis,
        **ex,
        **fa,
        "score": round(score, 3),
    }


def score_proposal(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    has_text = bool(t)
    cols = sorted({m.group(0).lower() for m in _DOLLAR_COL.finditer(t)})
    observable = len(cols) >= 1
    cliche = bool(_CLICHE.search(t))
    # heuristik kalimat hipotesis: cari pola when/then atau panah
    conditional = bool(re.search(r"\bwhen\b.*\b(then|→|->|reverse|predict|follow)", t, re.I))
    score = (0.3 * has_text + 0.3 * observable + 0.3 * conditional - 0.3 * cliche)
    return {
        "has_text": has_text, "observable_cols": cols, "observable": observable,
        "conditional": conditional, "cliche": cliche,
        "score": round(max(0.0, score), 3),
    }


def score_feedback(text: str, required=("Observations", "Feedback for Hypothesis",
                                        "Replace Best Result")) -> Dict[str, Any]:
    obj = _extract_json(text)
    if obj is None:
        return {"json_ok": False, "missing": list(required), "score": 0.0}
    missing = [k for k in required if k not in obj]
    score = 0.5 + 0.5 * (1 - len(missing) / max(1, len(required)))
    return {"json_ok": True, "keys": list(obj.keys()), "missing": missing,
            "score": round(score, 3)}


def score_passthrough(text: str, *, want_markers=()) -> Dict[str, Any]:
    t = (text or "").strip()
    hit = [m for m in want_markers if re.search(re.escape(m), t, re.I)]
    score = (0.5 if t else 0.0) + (0.5 * len(hit) / len(want_markers) if want_markers else 0.0)
    return {"len": len(t), "markers_hit": hit, "score": round(min(1.0, score), 3)}


def score_output(role: str, text: str) -> Dict[str, Any]:
    if role in ("construct", "judger"):
        return score_construct_judger(text)
    if role == "proposal":
        return score_proposal(text)
    if role == "feedback":
        return score_feedback(text)
    if role == "mutation":
        return score_passthrough(text, want_markers=("DIAGNOSIS", "DIRECTION"))
    if role == "crossover":
        return score_passthrough(text, want_markers=("OBSERVATION", "DIRECTION"))
    return score_passthrough(text)


# ════════════════════════════════════════════════════════════════════════════
# agregat lintas repetisi (stabilitas + variety)
# ════════════════════════════════════════════════════════════════════════════

def aggregate(role: str, per_rep: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_rep:
        return {"role": role, "n_rep": 0}
    scores = [r.get("score", 0.0) for r in per_rep]
    out = {
        "role": role,
        "n_rep": len(per_rep),
        "score_mean": round(statistics.fmean(scores), 3),
        "score_std": round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
    }
    if role in ("construct", "judger"):
        out["parse_rate"] = round(sum(int(r.get("parse_ok", False)) for r in per_rep) / len(per_rep), 3)
        gp = [r.get("gate_pass_frac", 0.0) for r in per_rep]
        out["gate_pass_rate"] = round(statistics.fmean(gp), 3)
        fams: set = set()
        hyps = set()
        for r in per_rep:
            fams |= set(r.get("families", []))
            if r.get("hypothesis"):
                hyps.add(r["hypothesis"].strip().lower())
        out["variety_families"] = len(fams)          # breadth lintas rep
        out["families_union"] = sorted(fams)
        out["n_distinct_hypotheses"] = len(hyps)      # diversitas hipotesis
        fa = [r.get("faithfulness", 0.0) for r in per_rep]
        out["faithfulness_mean"] = round(statistics.fmean(fa), 3)
    elif role == "proposal":
        out["observable_rate"] = round(sum(int(r.get("observable", False)) for r in per_rep) / len(per_rep), 3)
        out["cliche_rate"] = round(sum(int(r.get("cliche", False)) for r in per_rep) / len(per_rep), 3)
    elif role == "feedback":
        out["json_rate"] = round(sum(int(r.get("json_ok", False)) for r in per_rep) / len(per_rep), 3)
    return out


# ── util JSON ───────────────────────────────────────────────────────────────
def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL | re.I)
    if fence:
        t = fence.group(1).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1 or e < s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except json.JSONDecodeError:
        return None


def gate_kind() -> str:
    _gate()
    return _GATE_KIND or "unknown"


if __name__ == "__main__":
    # smoke test tanpa GPU
    print("gate:", gate_kind())
    demo = ("HYPOTHESIS: When 5-day volume z-score is high and range narrow, returns reverse.\n"
            "EXPRESSION 1: TS_ZSCORE($volume, 5) - RANK(($high-$low)/$close)\n"
            "EXPRESSION 2: REGBETA($return, $volume, 20)")
    import pprint
    pprint.pprint(score_output("judger", demo))
    pprint.pprint(score_output("proposal",
        "When 5-day $volume rises while ($high-$low) range stays narrow, next-week returns reverse."))
    print(aggregate("judger", [score_output("judger", demo), score_output("judger", demo)]))
