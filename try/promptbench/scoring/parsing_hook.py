"""
promptbench/scoring/parsing_hook.py
===================================
EXTENSION POINT untuk parsing output judger/construct di Phase B.

(Dahulu `chain/parsing_hook.py`. Dipindah ke `scoring/` saat konsolidasi Phase B
2026-06-18: kini SATU-SATUNYA hook parser, dipakai oleh KEDUA desain Phase B —
stages `chain/chain.py` dan chains `runners/bench_chain.py` — lewat
`scoring/score_chain.py`. Dependensi searah: chain & chains → scoring.)

KENAPA FILE INI ADA
-------------------
User mengamati: ADA output LLM yang BAGUS (hipotesis + ekspresi valid) tapi
GAGAL terdeteksi oleh `parsers.parse_hypothesis_exprs`. Contoh kelas kegagalan
yang sudah terlihat di scoreboard Phase A: construct/judger dengan parse_rate < 1.0
padahal teks tampak benar saat dibaca manual.

Daripada langsung mengubah parser PRODUKSI (`backend/latent_mas/parsers.py`) —
yang dipakai pipeline nyata & berisiko regresi — kita pasang INDIRECTION di sini:

  - default: delegate apa adanya ke parse_hypothesis_exprs (zero perubahan perilaku).
  - bila user sudah memilih prompt final & menemukan pola "bagus-tapi-gagal-parse"
    yang konkret, tambahkan PRE-NORMALIZER / FALLBACK di bawah (lihat REGISTER
    POINT). Phase B otomatis memakainya; parser produksi tetap utuh sampai pola
    terbukti aman → baru di-port ke parsers.py.

KONTRAK
-------
parse_judger(text) -> Optional[HypothesisExprs]   (sama dgn parse_hypothesis_exprs)
parse_construct(text) -> Optional[HypothesisExprs] (construct sering tanpa hipotesis)

Keduanya juga melaporkan APAKAH fallback dipakai (untuk audit di artefak), via
parse_with_trace(text, role) -> (parsed, trace_dict).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_BACKEND = Path(__file__).resolve().parent.parent.parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from latent_mas.parsers import HypothesisExprs, parse_hypothesis_exprs  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# REGISTER POINT — tambal di sini SETELAH user memilih prompt & menemukan pola
# ════════════════════════════════════════════════════════════════════════════
#
# Tiap entri = (nama, fungsi) di mana fungsi(text:str) -> Optional[str].
# Fungsi mengembalikan teks yang SUDAH dinormalisasi (siap di-feed ulang ke
# parse_hypothesis_exprs), atau None bila tidak relevan untuk teks itu.
#
# Dipanggil HANYA bila parse_hypothesis_exprs(text) gagal (return None / 0 expr).
# Urutan dicoba sesuai daftar; yang pertama menghasilkan parse sukses dipakai.
#
# Contoh kerangka (DINONAKTIFKAN — tunggu user pilih prompt):
#
#   def _strip_numbered_prose(text: str) -> Optional[str]:
#       """Model kadang menulis '1. EXPRESSION: ...' — angka+titik mematahkan
#       regex 'expr\\w*\\s*\\d*\\s*:'. Buang penomoran prosa di awal baris."""
#       import re
#       fixed = re.sub(r"(?m)^\\s*\\d+\\.\\s*(?=expr|hypo)", "", text, flags=re.I)
#       return fixed if fixed != text else None
#
#   _PRENORMALIZERS.append(("strip_numbered_prose", _strip_numbered_prose))
#
_PRENORMALIZERS: List[Tuple[str, Callable[[str], Optional[str]]]] = []


def register_prenormalizer(name: str, fn: Callable[[str], Optional[str]]) -> None:
    """Daftarkan pre-normalizer fallback. Idempoten by name."""
    global _PRENORMALIZERS
    _PRENORMALIZERS = [(n, f) for (n, f) in _PRENORMALIZERS if n != name]
    _PRENORMALIZERS.append((name, fn))


# ════════════════════════════════════════════════════════════════════════════
# API publik
# ════════════════════════════════════════════════════════════════════════════

def _is_empty(parsed: Optional[HypothesisExprs]) -> bool:
    return parsed is None or not getattr(parsed, "expressions", None)


def parse_with_trace(text: str, role: str = "judger"
                     ) -> Tuple[Optional[HypothesisExprs], Dict[str, Any]]:
    """Parse + jejak audit. Coba parser produksi dulu; bila gagal, jalankan
    pre-normalizer terdaftar satu per satu sampai ada yang sukses.

    trace = {
      "primary_ok": bool,                 # parser produksi langsung sukses?
      "fallback_used": Optional[str],     # nama pre-normalizer yang menyelamatkan
      "n_expr": int,
    }
    """
    primary = parse_hypothesis_exprs(text or "")
    if not _is_empty(primary):
        return primary, {"primary_ok": True, "fallback_used": None,
                         "n_expr": len(primary.expressions)}

    # parser produksi gagal → coba fallback yang terdaftar (default: kosong)
    for name, fn in _PRENORMALIZERS:
        try:
            normalized = fn(text or "")
        except Exception:
            normalized = None
        if not normalized:
            continue
        retry = parse_hypothesis_exprs(normalized)
        if not _is_empty(retry):
            return retry, {"primary_ok": False, "fallback_used": name,
                           "n_expr": len(retry.expressions)}

    return primary, {"primary_ok": False, "fallback_used": None,
                     "n_expr": 0}


def parse_judger(text: str) -> Optional[HypothesisExprs]:
    parsed, _ = parse_with_trace(text, role="judger")
    return parsed


def parse_construct(text: str) -> Optional[HypothesisExprs]:
    # construct sering hanya menulis EXPRESSION tanpa HYPOTHESIS — parser sama
    # toleran (hypothesis="" diperbolehkan). Pisahkan role demi audit/override.
    parsed, _ = parse_with_trace(text, role="construct")
    return parsed
