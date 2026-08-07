"""Uji unit `summarize_for_handoff` (B14) — ekstraksi handoff terstruktur.

Kenapa perlu unit test dan bukan sekadar dipercaya: kalau ekstraktornya diam-diam
mengembalikan string kosong, lengan `summary` akan kalah telak di GPU dan kita
akan menyimpulkan "ringkasan terstruktur merusak mutu" — padahal yang rusak
parsernya. Semua kasus di bawah deterministik dan berjalan di CPU.

    .venv/bin/python lab/test_b14_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

QL = Path(__file__).resolve().parent.parent
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
for p in (str(QL), str(QL / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from latent_mas.pipeline import summarize_for_handoff  # noqa: E402

PROPOSAL_OK = """observation: stocks with abnormally wide intraday range tend to
revert over the following two sessions, especially when volume is elevated.

driver: liquidity providers widen spreads during range expansion and are repaid
as the imbalance clears; carried by $high, $low and $volume over a 1-5 day band,
compared cross-sectionally, predicting negative next-period returns.

HYPOTHESIS: When a stock's intraday range expands far beyond its recent norm on
elevated volume, its next-period cross-sectional return is below average.
"""

INNOVATE_OK = """Step 1 - HYPOTHESIS VARIANTS
  1. Range expansion predicts reversal only in the top volume tercile.
  2. The effect flips sign for stocks already trending.

Step 2 - RECIPES
  ... a lot of prose that we do NOT want to carry into the next hop ...

Step 3 - self check: 6 axes, 2 neglected operators, ok.

{
  "hypothesis": "range expansion predicts reversal",
  "hypothesis_variants": ["only in top volume tercile", "flips for trending"],
  "recipes": [
    {"axis": "odd normalisation", "sketch": "range over TS_MAD", "uses": "$high,$low,20d"}
  ]
}
"""

# Kontrak dilanggar: tak ada baris HYPOTHESIS dan tak ada JSON.
BROKEN = "I understand the instruction and will comply with the format."

# JSON rusak diikuti JSON benar — harus memilih yang bisa di-parse.
INNOVATE_TWO_BLOCKS = """noise {"recipes": [oops,]} more text
{"hypothesis": "h", "recipes": [{"axis": "a", "sketch": "s", "uses": "u"}]}
"""


def check(name: str, got, cond: bool, detail: str = "") -> bool:
    print(f"  [{'OK ' if cond else 'GAGAL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        print(f"        dapat: {got!r}")
    return cond


def main() -> None:
    ok = True

    s = summarize_for_handoff("proposal", PROPOSAL_OK)
    ok &= check("proposal → hanya baris HYPOTHESIS", s,
                s.startswith("HYPOTHESIS:") and "observation:" not in s
                and "driver:" not in s,
                f"{len(PROPOSAL_OK)} → {len(s)} char")

    s = summarize_for_handoff("innovate", INNOVATE_OK)
    ok &= check("innovate → hanya blok JSON", s,
                s.startswith("{") and "Step 2" not in s,
                f"{len(INNOVATE_OK)} → {len(s)} char")
    try:
        parsed = json.loads(s)
        ok &= check("innovate → JSON valid & lengkap", s,
                    "recipes" in parsed and "hypothesis" in parsed)
    except ValueError:
        ok &= check("innovate → JSON valid & lengkap", s, False)

    s = summarize_for_handoff("innovate", INNOVATE_TWO_BLOCKS)
    ok &= check("innovate → pilih blok JSON yang bisa di-parse", s,
                s.startswith("{") and json.loads(s).get("hypothesis") == "h")

    # FAIL-OPEN: kontrak dilanggar → kembalikan teks asli, JANGAN string kosong.
    s = summarize_for_handoff("proposal", BROKEN)
    ok &= check("proposal kontrak rusak → fail-open (bukan kosong)", s,
                s.strip() == BROKEN.strip())
    s = summarize_for_handoff("innovate", BROKEN)
    ok &= check("innovate kontrak rusak → fail-open (bukan kosong)", s,
                s.strip() == BROKEN.strip())

    s = summarize_for_handoff("proposal", "")
    ok &= check("teks kosong → kosong (tanpa exception)", s, s == "")

    s = summarize_for_handoff("construct", PROPOSAL_OK)
    ok &= check("agen tanpa kontrak → teks apa adanya", s,
                s.strip() == PROPOSAL_OK.strip()[:1800].strip())

    print("\nsemua kasus lolos." if ok else "\nADA YANG GAGAL.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
