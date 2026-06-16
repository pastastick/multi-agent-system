"""
promptbench/chain/collapse.py
=============================
Detektor degradasi/collapse KV-cache untuk Phase B. Deterministik, tanpa GPU.

Dipakai pada SETIAP tip yang men-decode teks dalam rantai. Tiga sinyal
(sesuai GUIDE §7 Phase B "detektor collapse: lonjakan token / repetisi /
unparseable"):

  1. repetition       : teks decode berputar (n-gram / baris berulang) — gejala
                        klasik KV collapse pada model kecil saat KV terlalu panjang.
  2. unparseable_tip  : tip yang seharusnya menghasilkan ekspresi (construct/judger)
                        gagal di-parse → reasoning di KV tidak ter-decode jadi DSL.
  3. kv_token_spike   : pertumbuhan KV di satu batas agent jauh melebihi yang bisa
                        dijelaskan (input + output + latent_steps) → ada akumulasi
                        tak terduga (root-cause over-KV / collapse lintas agent).

`detect()` menggabung jadi satu verdict + alasan, untuk loop diagnosa Phase B.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── 1. repetisi ──────────────────────────────────────────────────────────────

def repetition_score(text: str) -> Dict[str, float]:
    """Ukur derajat pengulangan. Mengembalikan beberapa metrik 0..1:
      - line_repeat_frac : fraksi baris non-kosong yang merupakan duplikat.
      - ngram4_repeat_frac: 1 - (4-gram unik / total 4-gram) — makin tinggi makin berulang.
      - max_token_run     : run token identik terpanjang / total token (mis. "a a a a").
    """
    t = (text or "").strip()
    if not t:
        return {"line_repeat_frac": 0.0, "ngram4_repeat_frac": 0.0, "max_token_run": 0.0}

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    line_repeat_frac = 0.0
    if lines:
        seen: set = set()
        dup = 0
        for ln in lines:
            if ln in seen:
                dup += 1
            seen.add(ln)
        line_repeat_frac = round(dup / len(lines), 3)

    toks = re.findall(r"\S+", t)
    # Guard: metrik n-gram/run hanya bermakna pada teks cukup panjang. Tip pendek
    # (mis. satu ekspresi DSL valid) jangan dianggap "berulang".
    MIN_TOKS = 8
    ngram4_repeat_frac = 0.0
    max_token_run = 0.0
    if len(toks) >= MIN_TOKS:
        grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        if grams:
            ngram4_repeat_frac = round(1 - len(set(grams)) / len(grams), 3)
        max_run = 1
        cur = 1
        for i in range(1, len(toks)):
            if toks[i] == toks[i - 1]:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 1
        max_token_run = round(max_run / len(toks), 3)

    return {
        "line_repeat_frac": line_repeat_frac,
        "ngram4_repeat_frac": ngram4_repeat_frac,
        "max_token_run": max_token_run,
    }


# ── 2 & 3. boundary KV ───────────────────────────────────────────────────────

@dataclass
class Boundary:
    """Snapshot KV pada satu batas agent."""
    label: str
    kv_tokens: int
    n_input: int = 0
    n_output: int = 0
    latent_steps: int = 0
    size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "kv_tokens": self.kv_tokens,
            "n_input": self.n_input, "n_output": self.n_output,
            "latent_steps": self.latent_steps, "size_mb": self.size_mb,
        }


def kv_spikes(boundaries: List[Boundary], *, slack: int = 64) -> List[Dict[str, Any]]:
    """Hitung pertumbuhan KV tak-terjelaskan per batas.

    explained ≈ n_input + n_output + latent_steps (token yang memang ditambahkan
    agent ini). delta = kv_tokens - kv_tokens_sebelumnya. unexplained = delta -
    explained. Bila unexplained > slack → tandai spike (akumulasi tak terduga).
    `slack` menyerap selisih kecil (BOS/template/role tokens).
    """
    out: List[Dict[str, Any]] = []
    prev = 0
    for b in boundaries:
        delta = b.kv_tokens - prev
        explained = b.n_input + b.n_output + b.latent_steps
        unexplained = delta - explained
        out.append({
            "label": b.label, "kv_tokens": b.kv_tokens, "delta": delta,
            "explained": explained, "unexplained": unexplained,
            "spike": unexplained > slack,
        })
        prev = b.kv_tokens
    return out


# ── verdict gabungan ──────────────────────────────────────────────────────────

@dataclass
class CollapseVerdict:
    collapsed: bool
    reasons: List[str] = field(default_factory=list)
    repetition: Dict[str, float] = field(default_factory=dict)
    spikes: List[Dict[str, Any]] = field(default_factory=list)
    unparseable_tip: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collapsed": self.collapsed, "reasons": self.reasons,
            "repetition": self.repetition, "spikes": self.spikes,
            "unparseable_tip": self.unparseable_tip,
        }


# ambang default (boleh dikalibrasi setelah lihat distribusi nyata)
TH_LINE_REPEAT = 0.4
TH_NGRAM4 = 0.5
TH_TOKEN_RUN = 0.15


def detect(
    tip_text: str,
    boundaries: List[Boundary],
    *,
    tip_should_parse: bool,
    tip_parsed_ok: bool,
    spike_slack: int = 64,
) -> CollapseVerdict:
    """Gabung tiga sinyal jadi satu verdict.

    Args:
      tip_text         : teks decode pada tip rantai.
      boundaries       : daftar Boundary terurut dari seed → tip.
      tip_should_parse : True bila tip ini semestinya menghasilkan ekspresi
                         (construct/judger), False untuk feedback (JSON) dll.
      tip_parsed_ok    : hasil parse tip (lewat parsing_hook) sukses atau tidak.
      spike_slack      : toleransi token tak-terjelaskan per batas.
    """
    rep = repetition_score(tip_text)
    spikes = kv_spikes(boundaries, slack=spike_slack)
    reasons: List[str] = []

    if rep["line_repeat_frac"] >= TH_LINE_REPEAT:
        reasons.append(f"line_repeat={rep['line_repeat_frac']}")
    if rep["ngram4_repeat_frac"] >= TH_NGRAM4:
        reasons.append(f"ngram4_repeat={rep['ngram4_repeat_frac']}")
    if rep["max_token_run"] >= TH_TOKEN_RUN:
        reasons.append(f"token_run={rep['max_token_run']}")

    unparseable = bool(tip_should_parse and not tip_parsed_ok)
    if unparseable:
        reasons.append("unparseable_tip")

    spiking = [s for s in spikes if s["spike"]]
    for s in spiking:
        reasons.append(f"kv_spike@{s['label']}(+{s['unexplained']})")

    return CollapseVerdict(
        collapsed=bool(reasons),
        reasons=reasons,
        repetition=rep,
        spikes=spikes,
        unparseable_tip=unparseable,
    )
