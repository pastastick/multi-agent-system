"""prod/runner.py — gate + repair + backtest (deterministik, non-LLM).

Mengubah output construct (JSON faktor) → metrik backtest yang dibaca feedback.
Reuse gate regulator & parser ASLI dari latent_mas (backend), BUKAN dari `try/`
(nama paket `try` adalah keyword Python). Lihat DESIGN.md §3-4.

backtest punya dua mode:
  - "mock" : metrik deterministik (hash ekspresi) → menjalankan loop evolusi penuh
             + semua logging TANPA subsystem Qlib. Untuk menguji perilaku transfer
             laten end-to-end di GPU.
  - "real" : adapter ke Qlib (factors.QlibFactorRunner._compute_factor_ic). Butuh
             data qlib + environment; di-wire tapi belum diuji di sini (F3 lanjut).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_BACKEND = Path(__file__).resolve().parent.parent / "backend"


def _ensure_backend_path() -> None:
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))


@contextmanager
def _silenced():
    """Bungkam stdout+stderr (regulator memakai loguru→stderr) selama panggilan
    gate, agar log terminal pipeline hanya berisi baris [prod ...]."""
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        yield


# ── gate (regulator asli, di-cache) ──────────────────────────────────────────
_GATE: Optional[Callable[[str], Tuple[bool, str]]] = None
_GATE_KIND: str = "uninit"


def build_gate() -> Callable[[str], Tuple[bool, str]]:
    global _GATE, _GATE_KIND
    if _GATE is None:
        _ensure_backend_path()
        try:
            from latent_mas.pipeline import FrontEndPipeline
            with _silenced():
                gate, reg = FrontEndPipeline._build_regulator_gate()
            try:  # senyapkan chatter loguru regulator (INFO/WARNING/ERROR) agar
                  # terminal pipeline hanya berisi baris [prod ...]. disable() tak
                  # cukup (regulator menambah sink sendiri) → reset ke CRITICAL.
                from loguru import logger as _lg
                _lg.remove()
                _lg.add(sys.stderr, level="CRITICAL")
            except Exception:
                pass
            _GATE, _GATE_KIND = gate, ("regulator" if reg is not None else "syntax_fallback")
        except Exception as e:  # noqa: BLE001
            _GATE_KIND = f"unavailable:{type(e).__name__}"
            _GATE = lambda expr: (bool(expr and "$" in expr), "syntax-only")
    return _GATE


def gate_kind() -> str:
    build_gate()
    return _GATE_KIND


def _families(expr: str) -> List[str]:
    _ensure_backend_path()
    try:
        from latent_mas.operator_families import families_of
        return sorted(families_of(expr))
    except Exception:
        return []


def _gate_one(expr: str) -> Tuple[bool, str]:
    gate = build_gate()
    with _silenced():
        ok, reason = gate(expr)
    return bool(ok), (reason or "")


# ── parse construct output → faktor ──────────────────────────────────────────

def parse_construct(text: str) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """(hypothesis, [{name, expression, explanation}]) dari output construct.

    Utama: JSON construct (name/expression/explanation). Fallback: parser DSL
    latent_mas (parse_hypothesis_exprs) bila JSON rusak → nama f1..fn.
    """
    obj = _extract_json(text)
    if obj and isinstance(obj.get("factors"), list):
        facs = []
        for i, f in enumerate(obj["factors"]):
            if not isinstance(f, dict):
                continue
            facs.append({
                "name": str(f.get("name") or f"f{i+1}").strip(),
                "expression": str(f.get("expression") or "").strip(),
                "explanation": str(f.get("explanation") or "").strip(),
            })
        if facs:
            return (str(obj.get("hypothesis") or "").strip() or None, facs)

    _ensure_backend_path()
    try:
        from latent_mas.parsers import parse_hypothesis_exprs
        parsed = parse_hypothesis_exprs(text or "")
    except Exception:
        parsed = None
    if parsed is None:
        return None, []
    facs = [{"name": f"f{i+1}", "expression": e, "explanation": ""}
            for i, e in enumerate(parsed.expressions)]
    return parsed.hypothesis, facs


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


# ── gate + repair ────────────────────────────────────────────────────────────

def _repair_brackets(expr: str) -> str:
    """Repair ringan kurung: (1) potong junk/kelebihan ')' via parser latent_mas,
    lalu (2) tutup kurung '(' yang masih menggantung. Model 4B sering salah salah
    satu dari keduanya."""
    _ensure_backend_path()
    try:
        from latent_mas.parsers import _balance_parens as bp
        expr = bp(expr)  # buang kelebihan ')' / trailing junk
    except Exception:
        pass
    opens = expr.count("(") - expr.count(")")
    if opens > 0:
        expr = expr + (")" * opens)
    return expr


def gate_and_repair(factors: List[Dict[str, str]], log=None) -> List[Dict[str, Any]]:
    """Gate tiap faktor; coba repair ringan (balance kurung) bila gagal.

    Mengembalikan list diperkaya: {name, expression, ok, reason, families,
    repaired(bool), original}. ALASAN penolakan gate di-LOG (permintaan F3).
    """
    out: List[Dict[str, Any]] = []
    for f in factors:
        expr = f["expression"]
        ok, reason = _gate_one(expr)
        repaired = False
        original = expr
        if not ok and expr:
            fixed = _repair_brackets(expr)
            if fixed != expr:
                ok2, reason2 = _gate_one(fixed)
                if ok2:
                    expr, ok, reason, repaired = fixed, True, reason2, True
        rec = {"name": f["name"], "expression": expr, "ok": ok, "reason": reason,
               "families": _families(expr), "repaired": repaired, "original": original}
        out.append(rec)
        if log is not None:
            if not ok:
                log.line(f"  GATE REJECT  {f['name']}: {original}  -> {reason}")
            elif repaired:
                log.line(f"  GATE REPAIR  {f['name']}: {original}  ->  {expr}")
    return out


# ── backtest ─────────────────────────────────────────────────────────────────

def _mock_metric(expr: str, salt: str, lo: float, hi: float) -> float:
    h = int(hashlib.md5((salt + "|" + expr).encode()).hexdigest(), 16)
    return round(lo + (h % 10_000) / 10_000 * (hi - lo), 4)


def _backtest_mock(legal: List[Dict[str, Any]]) -> Dict[str, Any]:
    per = {}
    for r in legal:
        per[r["name"]] = {
            "rankic": _mock_metric(r["expression"], "ic", -0.02, 0.06),
            "icir": _mock_metric(r["expression"], "icir", 0.0, 0.5),
        }
    combined = round(0.05 + _mock_metric("|".join(r["expression"] for r in legal) or "x",
                                         "comb", 0.0, 0.015), 4)
    maxdd = round(0.25 + _mock_metric("x", "dd", 0.0, 0.12), 4)
    return {"per_factor": per, "combined_rankic": combined, "max_drawdown": maxdd,
            "engine": "mock"}


def _backtest_real(legal: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Adapter Qlib (BELUM diuji di sini). Integrasi: bangun DataFrame nilai faktor
    dari ekspresi lalu panggil factors.QlibFactorRunner._compute_factor_ic. Butuh
    data qlib + environment. Lihat factors/runner.py:_compute_factor_ic."""
    raise NotImplementedError(
        "backtest 'real' butuh environment Qlib + data. Wire ke "
        "factors.QlibFactorRunner._compute_factor_ic (factors/runner.py). "
        "Sementara pakai mode='mock' untuk menguji loop + transfer laten.")


def backtest(legal: List[Dict[str, Any]], mode: str = "mock") -> Dict[str, Any]:
    t0 = time.time()
    res = _backtest_real(legal) if mode == "real" else _backtest_mock(legal)
    res["duration_s"] = round(time.time() - t0, 3)
    return res


# ── orkestrasi: construct text → var feedback ────────────────────────────────

def run_construct(construct_text: str, *, sota_rankic: Optional[float] = None,
                  mode: str = "mock", log=None) -> Dict[str, Any]:
    """Gate+repair+backtest sebuah output construct → var untuk agent feedback.

    Return: {hypothesis, factor_block, backtest_results, sota_block, _score, _best_rankic}
    """
    hypothesis, factors = parse_construct(construct_text)
    if log is not None:
        log.line(f"construct parse: hypothesis={'yes' if hypothesis else 'NO'} "
                 f"factors={len(factors)} gate={gate_kind()}")
    graded = gate_and_repair(factors, log=log)
    legal = [r for r in graded if r["ok"]]
    n_total, n_legal = len(graded), len(legal)

    bt = backtest(legal, mode=mode) if legal else {
        "per_factor": {}, "combined_rankic": 0.0, "max_drawdown": 0.0,
        "engine": mode, "duration_s": 0.0}
    if log is not None:
        log.line(f"backtest[{bt['engine']}] {n_legal}/{n_total} legal  "
                 f"dur={bt['duration_s']:.2f}s  combinedRankIC={bt['combined_rankic']} "
                 f"maxDD={bt['max_drawdown']}")

    # best standalone RankIC
    best_name, best_ic, best_icir = None, None, None
    for r in legal:
        m = bt["per_factor"].get(r["name"], {})
        if m.get("rankic") is not None and (best_ic is None or m["rankic"] > best_ic):
            best_name, best_ic, best_icir = r["name"], m["rankic"], m.get("icir")

    replace = bool(best_ic is not None and (sota_rankic is None or best_ic > sota_rankic))
    if log is not None:
        log.line(f"best={best_name} RankIC={best_ic} ICIR={best_icir}  "
                 f"SOTA={sota_rankic}  REPLACE_BEST={'yes' if replace else 'no'}")

    return {
        "hypothesis": hypothesis or "(unparsed)",
        "factor_block": _fmt_factor_block(graded, bt),
        "backtest_results": _fmt_backtest(graded, bt),
        "sota_block": (f"REPLACE BEST = {'yes' if replace else 'no'} "
                       f"(best RankIC={best_ic} vs SOTA={sota_rankic})"),
        "_score": {"score": best_ic, "best": best_name, "replace": replace,
                   "n_legal": n_legal, "n_total": n_total,
                   "backtest_s": bt["duration_s"]},
        "_best_rankic": best_ic,
    }


def _fmt_factor_block(graded: List[Dict[str, Any]], bt: Dict[str, Any]) -> str:
    n_legal = sum(int(r["ok"]) for r in graded)
    lines = [f"FACTORS (total={len(graded)}, legal={n_legal} after gate/repair):"]
    for r in graded:
        if not r["ok"]:
            tag = f"REJECTED: {r['reason']}"
        elif r["repaired"]:
            tag = f"legal (repaired from: {r['original']})"
        else:
            tag = "legal"
        lines.append(f"- {r['name']}: {r['expression']}  [{tag}]")
    return "\n".join(lines)


def _fmt_backtest(graded: List[Dict[str, Any]], bt: Dict[str, Any]) -> str:
    lines = [f"Block A (standalone per-factor RankIC/ICIR, OOS) [engine={bt['engine']}]:"]
    for r in graded:
        if not r["ok"]:
            continue
        m = bt["per_factor"].get(r["name"], {})
        lines.append(f"- {r['name']}: RankIC={m.get('rankic')} ICIR={m.get('icir')}")
    lines.append(f"Block B (combined LightGBM): RankIC={bt['combined_rankic']} "
                 f"MaxDrawdown={bt['max_drawdown']}")
    lines.append(f"(backtest duration: {bt['duration_s']:.2f}s, engine={bt['engine']})")
    return "\n".join(lines)
