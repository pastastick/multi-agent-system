"""Uji tiga lubang gate B15 dengan ekspresi NYATA dari korpus G5.

Semua kasus di bawah pernah LOLOS gate lalu mati/crash saat dieksekusi. Uji ini
deterministik dan tak butuh GPU:

    python lab/test_gate_b15.py
"""
from __future__ import annotations

import sys
from pathlib import Path

QL = Path(__file__).resolve().parent.parent
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
for p in (str(QL), str(QL / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

import factors.coder.factor_ast  # noqa: E402,F401  — memutus circular import
from factors.regulator.factor_regulator import (  # noqa: E402
    validate_function_arity, validate_semantics,
)

# (ekspresi, harus_ditolak, label)
CASES = [
    # ── B15a: keluaran boolean 2-nilai (7 dari 198 di G5) ───────────────────
    ("($volume > TS_ZSCORE($volume, 5)) ? (-1) : (1)", True, "B15a ternary konstan"),
    ("(TS_RANK($volume, 20) > 0.8) ? (1) : (0)", True, "B15a gate 0/1"),
    ("TS_ZSCORE($volume, 5) > 2", True, "B15a boolean telanjang"),
    ("RANK((TS_RANK($volume, 20) > 0.8) ? (1) : (-1))", True, "B15a 2-nilai dibungkus RANK"),
    # yang HARUS tetap lolos — data mencapai NILAI, bukan cuma kondisi
    ("(TS_ZSCORE($volume, 20) > 2) ? (TS_PCTCHANGE($close, 5)) : (0)", False,
     "gate kondisional sehat"),
    ("ZSCORE(TS_PCTCHANGE($close, 5))", False, "ekspresi kontinu biasa"),
    ("RANK($volume) * TS_ZSCORE($return, 10)", False, "perkalian dua sinyal"),

    # ── B15b: argumen kuantil di luar [0,1] ─────────────────────────────────
    ("TS_QUANTILE($volume, 20, 5)", True, "B15b q=5 di TS_QUANTILE"),
    ("PERCENTILE($close, 20, 5)", True, "B15b q=20 di PERCENTILE"),
    ("TS_QUANTILE($volume, 20, 0.9)", False, "kuantil sah TS_QUANTILE"),
    ("PERCENTILE($close, 0.9, 20)", False, "kuantil sah PERCENTILE"),
]

# (ekspresi, harus_ditolak_arity, label) — B15c
ARITY_CASES = [
    ("TS_MEAN($close)", True, "B15c window hilang (Python diam-diam isi 5)"),
    ("TS_STD($return)", True, "B15c window hilang"),
    ("DELTA($close)", True, "B15c window hilang"),
    ("TS_MEAN($close, 5)", False, "window ditulis eksplisit"),
    ("TS_CORR($close, $volume, 10)", False, "tiga argumen sesuai kontrak"),
    ("RANK($close)", False, "cross-sectional 1 argumen"),
]


def main() -> int:
    fails = []
    print("── validate_semantics (B15a, B15b) ──")
    for expr, want_reject, label in CASES:
        ok, errs = validate_semantics(expr)
        rejected = not ok
        mark = "OK " if rejected == want_reject else "GAGAL"
        if rejected != want_reject:
            fails.append((label, expr, want_reject, rejected, errs))
        print(f"  [{mark}] {label:36s} tolak={rejected!s:5s} {expr[:52]}")
        if rejected and errs:
            print(f"          → {errs[0][:96]}")

    print("\n── validate_function_arity (B15c) ──")
    for expr, want_reject, label in ARITY_CASES:
        ok, errs = validate_function_arity(expr)
        rejected = not ok
        mark = "OK " if rejected == want_reject else "GAGAL"
        if rejected != want_reject:
            fails.append((label, expr, want_reject, rejected, errs))
        print(f"  [{mark}] {label:36s} tolak={rejected!s:5s} {expr[:52]}")
        if rejected and errs:
            print(f"          → {errs[0][:96]}")

    if fails:
        print(f"\n{len(fails)} kasus GAGAL:")
        for label, expr, want, got, errs in fails:
            print(f"  {label}: {expr}  (harus tolak={want}, dapat={got}) {errs[:1]}")
        return 1
    print(f"\nsemua {len(CASES) + len(ARITY_CASES)} kasus lolos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
