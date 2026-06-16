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


def score_construct_judger(text: str) -> Dict[str, Any]:
    parsed = parse_hypothesis_exprs(text or "")
    if parsed is None:
        return {"parse_ok": False, "hypothesis": None, "n_expr": 0,
                "gate_pass": 0, "gate_pass_frac": 0.0, "n_families": 0,
                "families": [], "score": 0.0}
    ex = score_expressions(parsed.expressions)
    # skor 0..1: parse(0.3) + gate(0.5*frac) + variety(0.2 bila >=2 family)
    score = 0.3 + 0.5 * ex["gate_pass_frac"] + (0.2 if ex["n_families"] >= 2 else 0.0)
    return {
        "parse_ok": True,
        "hypothesis": parsed.hypothesis,
        **ex,
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
