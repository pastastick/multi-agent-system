"""
promptbench/diagnostics/collapse.py
===================================
Detektor COLLAPSE deterministik untuk rantai latent (Phase B). Tanpa GPU.

Dua lapis deteksi:

1. KV-growth (per batas agent) — `detect_kv_growth`
   Tiap agent menambah ke KV: token prompt-nya + `latent_steps` virtual token.
   Jadi pertumbuhan WAJAR per langkah ≈ n_prompt_tokens + latent_steps.
   Yang dicurigai (ditandai):
     - delta >> ekspektasi → KEbocoran/penimbunan tak sengaja (mis. answer token
       ikut ter-chain, atau KV dipakai ulang tanpa clone).
     - delta < 0 atau ~0 padahal bukan langkah-0 → transfer KV gagal.
     - total seq_len meledak melewati cap → akumulasi tak terkendali.
   Catatan: pertumbuhan MONOTON yang wajar BUKAN collapse — itu memang
   "latent working memory transfer" ala LatentMAS. Yang kita kejar adalah
   LONJAKAN anomali.

2. Teks terminal — `detect_text_collapse`
   Output agent terakhir (di-decode). Gejala collapse khas model 4B:
     - runaway: output_tokens ≈ max_new_tokens (model tak berhenti).
     - repetition: rasio n-gram berulang tinggi (ngelantur / loop).
     - unparseable: parser hypothesis/expr gagal (untuk agent yang harusnya
       menghasilkan struktur).
     - empty: teks kosong.

`summarize_chain_health` menggabungkan keduanya jadi satu verdict per run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ════════════════════════════════════════════════════════════════════════════
# rekaman batas agent
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundaryRecord:
    """Snapshot KV pada batas keluaran satu agent."""
    step_idx: int
    agent: str
    mode: str                       # kv_only | kv_and_text
    latent_steps: int
    n_prompt_tokens: int            # token prompt agent ini (sebelum latent)
    kv_seq_len: int                 # panjang KV total SETELAH langkah ini
    kv_size_mb: float
    transfer: str                   # none | chain | concat
    prev_seq_len: int = 0           # panjang KV input (sebelum langkah ini)

    @property
    def delta(self) -> int:
        return self.kv_seq_len - self.prev_seq_len

    @property
    def expected_delta(self) -> int:
        # langkah-0 (transfer none): prompt + latent. langkah chain: idem
        # (prompt agent ini di-encode di atas KV sebelumnya + latent steps).
        return self.n_prompt_tokens + self.latent_steps

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["delta"] = self.delta
        d["expected_delta"] = self.expected_delta
        return d


# ════════════════════════════════════════════════════════════════════════════
# 1. KV-growth
# ════════════════════════════════════════════════════════════════════════════

def detect_kv_growth(
    boundaries: List[BoundaryRecord],
    *,
    tol: float = 1.5,
    abs_slack: int = 8,
    seq_cap: int = 32768,
) -> Dict[str, Any]:
    """Periksa pertumbuhan KV lintas batas agent.

    Args:
        tol       : delta dianggap anomali bila > expected*tol + abs_slack.
        abs_slack : toleransi absolut (token chat-template, BOS, dst.).
        seq_cap   : panjang KV maksimum wajar; di atasnya = akumulasi liar.

    Return dict {ok, flags[], per_step[]}.
    """
    flags: List[str] = []
    per_step: List[Dict[str, Any]] = []

    for b in boundaries:
        exp = b.expected_delta
        upper = exp * tol + abs_slack
        status = "ok"

        # langkah dengan transfer 'none' harus delta>0; 'chain' harus delta>0
        # dan prev_seq_len>0 (KV benar-benar terbawa).
        if b.transfer == "chain" and b.prev_seq_len <= 0:
            status = "transfer_missing"
            flags.append(f"step{b.step_idx}({b.agent}): transfer=chain tapi prev_seq_len=0 "
                         f"→ KV upstream TIDAK terbawa")
        elif b.delta <= 0 and b.step_idx >= 0:
            status = "no_growth"
            flags.append(f"step{b.step_idx}({b.agent}): delta={b.delta} ≤ 0 "
                         f"→ KV tidak bertambah (transfer/append gagal?)")
        elif b.delta > upper:
            status = "spike"
            flags.append(f"step{b.step_idx}({b.agent}): delta={b.delta} >> "
                         f"expected≈{exp} (cap {upper:.0f}) → kemungkinan answer-token "
                         f"bocor / KV menimbun")

        if b.kv_seq_len > seq_cap:
            status = "overflow"
            flags.append(f"step{b.step_idx}({b.agent}): kv_seq_len={b.kv_seq_len} "
                         f"> cap {seq_cap} → akumulasi tak terkendali")

        per_step.append({**b.to_dict(), "status": status})

    return {"ok": not flags, "n_flags": len(flags), "flags": flags, "per_step": per_step}


# ════════════════════════════════════════════════════════════════════════════
# 2. teks terminal
# ════════════════════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"\S+")


def repetition_ratio(text: str, n: int = 3) -> float:
    """Rasio n-gram (default trigram) yang BUKAN unik: 0=variatif, →1=loop.

    Deterministik. Dipakai sebagai proxy "model ngelantur/loop".
    """
    words = _WORD_RE.findall(text or "")
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    uniq = len(set(grams))
    return round(1.0 - uniq / len(grams), 3)


def longest_run(text: str) -> int:
    """Panjang pengulangan kata identik berturut-turut (mis. 'the the the')."""
    words = _WORD_RE.findall(text or "")
    best = run = 0
    prev = None
    for w in words:
        run = run + 1 if w == prev else 1
        best = max(best, run)
        prev = w
    return best


def detect_text_collapse(
    text: Optional[str],
    *,
    output_tokens: int = 0,
    max_new_tokens: int = 512,
    parser_ok: Optional[bool] = None,
    rep_threshold: float = 0.5,
    run_threshold: int = 12,
    runaway_frac: float = 0.97,
) -> Dict[str, Any]:
    """Deteksi degenerasi pada teks terminal.

    parser_ok: hasil parse struktur (hypothesis/expr/JSON). None = tidak relevan.
    """
    t = (text or "").strip()
    reasons: List[str] = []

    if not t:
        return {"collapsed": True, "reasons": ["empty"], "metrics": {
            "len": 0, "repetition": 0.0, "longest_run": 0, "runaway": False}}

    rep = repetition_ratio(t)
    run = longest_run(t)
    runaway = max_new_tokens > 0 and output_tokens >= int(max_new_tokens * runaway_frac)

    if rep >= rep_threshold:
        reasons.append(f"repetition={rep} ≥ {rep_threshold} (loop/ngelantur)")
    if run >= run_threshold:
        reasons.append(f"longest_run={run} ≥ {run_threshold} (kata berulang beruntun)")
    if runaway:
        reasons.append(f"runaway: output_tokens={output_tokens} ≈ max_new_tokens "
                       f"({max_new_tokens}) → tak berhenti")
    if parser_ok is False:
        reasons.append("unparseable: parser struktur gagal")

    return {
        "collapsed": bool(reasons),
        "reasons": reasons,
        "metrics": {
            "len": len(t), "repetition": rep, "longest_run": run,
            "output_tokens": output_tokens, "runaway": runaway,
            "parser_ok": parser_ok,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# verdict gabungan
# ════════════════════════════════════════════════════════════════════════════

def summarize_chain_health(
    kv_result: Dict[str, Any],
    text_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Gabung verdict KV-growth + teks terminal jadi satu status rantai."""
    healthy = bool(kv_result.get("ok")) and not text_result.get("collapsed")
    return {
        "healthy": healthy,
        "kv_ok": bool(kv_result.get("ok")),
        "text_collapsed": bool(text_result.get("collapsed")),
        "kv_flags": kv_result.get("flags", []),
        "text_reasons": text_result.get("reasons", []),
    }


if __name__ == "__main__":
    # smoke test (tanpa GPU)
    b0 = BoundaryRecord(0, "proposal", "kv_only", 20, 180, 200, 30.0, "none", 0)
    b1 = BoundaryRecord(1, "construct", "kv_only", 20, 60, 280, 42.0, "chain", 200)
    b2 = BoundaryRecord(2, "judger", "kv_and_text", 20, 50, 999, 150.0, "chain", 280)  # spike
    kv = detect_kv_growth([b0, b1, b2])
    print("KV ok:", kv["ok"], "flags:", kv["flags"])
    txt = detect_text_collapse("the the the the the the the the the the the the the",
                               output_tokens=512, max_new_tokens=512, parser_ok=False)
    print("text collapsed:", txt["collapsed"], "reasons:", txt["reasons"])
    print("health:", summarize_chain_health(kv, txt))
