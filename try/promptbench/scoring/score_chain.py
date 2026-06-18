"""
promptbench/scoring/score_chain.py
==================================
Skor DOWNSTREAM untuk Phase B: nilai output agent TERMINAL sebuah rantai,
memakai kembali rubrik deterministik Phase A (gate regulator + variety + parse).

Beda dengan Phase A: input bukan agent terisolasi, melainkan hasil akhir rantai
KV (proposal→…→terminal). Yang kita ukur sama — parse-rate, gate pass-rate,
variety family — plus sinyal `parser_ok` untuk detektor collapse.

PARSER HOOK (konsolidasi 2026-06-18)
------------------------------------
Untuk construct/judger, parsing di sini lewat `parsing_hook.parse_with_trace`
(bukan `parse_hypothesis_exprs` langsung). Jadi SATU titik parser dipakai oleh
KEDUA desain Phase B (stages `chain/chain.py` & chains `runners/bench_chain.py`),
keduanya memanggil `score_terminal`. Default hook = delegate apa adanya ke parser
produksi (zero perubahan), tapi pre-normalizer yang didaftarkan otomatis ikut
serta + jejak audit (`parse_trace`) ikut disertakan untuk artefak.

Tanpa GPU. Murni teks → skor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import score as _A   # reuse Phase A scorers (score_output, dll.)
from . import parsing_hook


def parser_ok_for(role: str, text: str) -> Optional[bool]:
    """Apakah output role ini bisa di-parse jadi struktur yang diharapkan?

    Return True/False untuk role berstruktur (construct/judger/feedback),
    None untuk role yang tak punya parser ketat (proposal/mutation/crossover).
    Dipakai detektor collapse (`unparseable`).
    """
    if role in ("construct", "judger"):
        parsed, _ = parsing_hook.parse_with_trace(text or "", role=role)
        return not parsing_hook._is_empty(parsed)
    if role == "feedback":
        d = _A.score_feedback(text or "")
        return bool(d.get("json_ok"))
    return None


def score_terminal(role: str, text: str) -> Dict[str, Any]:
    """Skor output terminal rantai + sertakan parser_ok untuk collapse-check.

    Untuk construct/judger: parsing lewat hook (fallback pre-normalizer + trace),
    lalu skor memakai hasil parse yang sama → konsisten dengan parser_ok.
    """
    if role in ("construct", "judger"):
        parsed, trace = parsing_hook.parse_with_trace(text or "", role=role)
        detail = _A.score_construct_judger(text or "", parsed=parsed)
        detail["parser_ok"] = not parsing_hook._is_empty(parsed)
        detail["parse_trace"] = trace
        return detail
    detail = _A.score_output(role, text or "")
    detail["parser_ok"] = parser_ok_for(role, text or "")
    return detail


def aggregate_chain(role: str, per_rep: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Agregat lintas repetisi untuk satu (chain, config) — reuse Phase A."""
    return _A.aggregate(role, per_rep)


if __name__ == "__main__":
    demo = ("HYPOTHESIS: When 5-day volume z-score is high and range narrow, returns reverse.\n"
            "EXPRESSION 1: TS_ZSCORE($volume, 5) - RANK(($high-$low)/$close)\n"
            "EXPRESSION 2: REGBETA($return, $volume, 20)")
    import pprint
    pprint.pprint(score_terminal("judger", demo))
    print("parser_ok(feedback, ''):", parser_ok_for("feedback", ""))
