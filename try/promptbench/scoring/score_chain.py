"""
promptbench/scoring/score_chain.py
==================================
Skor DOWNSTREAM untuk Phase B: nilai output agent TERMINAL sebuah rantai,
memakai kembali rubrik deterministik Phase A (gate regulator + variety + parse).

Beda dengan Phase A: input bukan agent terisolasi, melainkan hasil akhir rantai
KV (proposal→…→terminal). Yang kita ukur sama — parse-rate, gate pass-rate,
variety family — plus sinyal `parser_ok` untuk detektor collapse.

Tanpa GPU. Murni teks → skor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import score as _A   # reuse Phase A scorers (score_output, dll.)


def parser_ok_for(role: str, text: str) -> Optional[bool]:
    """Apakah output role ini bisa di-parse jadi struktur yang diharapkan?

    Return True/False untuk role berstruktur (construct/judger/feedback),
    None untuk role yang tak punya parser ketat (proposal/mutation/crossover).
    Dipakai detektor collapse (`unparseable`).
    """
    if role in ("construct", "judger"):
        d = _A.score_construct_judger(text or "")
        return bool(d.get("parse_ok"))
    if role == "feedback":
        d = _A.score_feedback(text or "")
        return bool(d.get("json_ok"))
    return None


def score_terminal(role: str, text: str) -> Dict[str, Any]:
    """Skor output terminal rantai + sertakan parser_ok untuk collapse-check."""
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
