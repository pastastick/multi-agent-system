"""Audit ekspresi batch 2026-07-05 di CPU.

Menjawab: apakah IC jelek karena (a) ide faktor lemah, (b) ekspresi cacat/degenerate,
(c) metrik/evaluasi yang salah, atau (d) semua faktor sebenarnya sinyal yang sama.

Output: lab/out/audit_batch.json + ringkasan ke stdout.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.core import Lab  # noqa: E402

QL = Path(__file__).resolve().parent.parent
POOLS = {
    "text": QL / "backend/runs/prod_text_2026-07-05_09-36-48/trajectory_pool_MERGED.json",
    "kv_and_text": QL / "backend/runs/prod_kv_and_text_2026-07-05_09-36-48/trajectory_pool_MERGED.json",
    "kv": QL / "backend/runs/prod_kv_2026-07-05_09-36-48-resume/trajectory_pool.json",
}
OUT = QL / "lab" / "out"

# operator time-series yang butuh window > 1 agar tidak degenerate
TS_OPS = {
    "TS_ZSCORE": 1, "TS_RANK": 1, "TS_MEAN": 1, "TS_MEDIAN": 1, "TS_STD": 1,
    "TS_VAR": 1, "TS_MAX": 1, "TS_MIN": 1, "TS_SUM": 1, "TS_ARGMAX": 1,
    "TS_ARGMIN": 1, "TS_SKEW": 1, "TS_KURT": 1, "TS_MAD": 1, "TS_CORR": 2,
    "TS_COVARIANCE": 2, "TS_QUANTILE": 1,
}
# window minimal agar statistik terdefinisi (std butuh >= 2, skew >= 3, kurt >= 4)
MIN_WIN = {"TS_ZSCORE": 2, "TS_STD": 2, "TS_VAR": 2, "TS_CORR": 2,
           "TS_COVARIANCE": 2, "TS_SKEW": 3, "TS_KURT": 4, "TS_MAD": 2}


def collect() -> list[dict]:
    rows = []
    for mode, path in POOLS.items():
        pool = json.loads(Path(path).read_text())
        for tid, t in pool["trajectories"].items():
            ei = t.get("extra_info") or {}
            fic = ei.get("factor_ic") or {}
            ficir = ei.get("factor_icir") or {}
            for f in (t.get("factors") or []):
                if not isinstance(f, dict):
                    continue
                rows.append({
                    "mode": mode, "traj": tid, "phase": t.get("phase"),
                    "dir": t.get("direction_id"), "round": t.get("round_idx"),
                    "hypothesis": t.get("hypothesis") or "",
                    "name": f.get("name"), "expr": f.get("expression", ""),
                    "ic_rec": fic.get(f.get("name")),
                    "icir_rec": ficir.get(f.get("name")),
                })
    return rows


def static_flags(expr: str) -> list[str]:
    """Cacat yang bisa dideteksi TANPA menjalankan ekspresi."""
    flags = []
    # 1. window degenerate: TS_OP(..., 1) atau window < minimum statistik
    for op, wpos in TS_OPS.items():
        for m in re.finditer(rf"\b{op}\s*\(", expr):
            args, depth, i = [], 0, m.end()
            cur = ""
            while i < len(expr):
                c = expr[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    if depth == 0:
                        args.append(cur)
                        break
                    depth -= 1
                elif c == "," and depth == 0:
                    args.append(cur)
                    cur = ""
                    i += 1
                    continue
                cur += c
                i += 1
            if len(args) > wpos:
                w = args[wpos].strip()
                if re.fullmatch(r"\d+", w):
                    wi = int(w)
                    need = MIN_WIN.get(op, 2)
                    if wi < need:
                        flags.append(f"degenerate-window:{op}(w={wi}<{need})")
    # 2. ternary/kondisi dengan ekspresi kontinu (bukan perbandingan) sebagai syarat
    for m in re.finditer(r"([^?]*)\?", expr):
        cond = m.group(1)
        cond = cond[max(cond.rfind("("), cond.rfind(":")) + 1:].strip()
        if cond and not re.search(r"[<>=!]", cond):
            flags.append("nonboolean-condition")
            break
    # 3. threshold absolut pada besaran yang tidak sebanding lintas saham
    if re.search(r"\$volume[^)]*\)?\s*[<>]\s*\d{4,}", expr) or re.search(
            r"TS_(MEAN|MIN|MAX|SUM)\(\s*\$volume[^)]*\)\s*[<>]\s*\d{4,}", expr):
        flags.append("absolute-volume-threshold")
    # 4. perbandingan pada rank persentil dengan ambang > 1 (TS_RANK pct=True ∈ [0,1])
    for m in re.finditer(r"TS_RANK\([^)]*\)\s*[<>]=?\s*(\d+(?:\.\d+)?)", expr):
        if float(m.group(1)) > 1.0:
            flags.append("pct-rank-vs-threshold>1")
    return sorted(set(flags))


def main(limit: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lab = Lab(mode="fast")
    rows = collect()
    if limit:
        rows = rows[:limit]
    print(f"[audit] {len(rows)} ekspresi, universe/hari ≈ ?, OOS {lab.oos_start.date()}..{lab.oos_end.date()}")

    ic_series = {}
    for i, r in enumerate(rows, 1):
        r["flags"] = static_flags(r["expr"])
        res, series = lab.ic_full(r["expr"])
        r.update({
            "ic": res.ic, "icir": res.icir, "tstat": res.tstat,
            "n_days": res.n_days, "coverage": res.coverage,
            "n_unique": res.n_unique, "error": res.error,
        })
        if series is not None:
            ic_series[f"{r['mode']}|{r['name']}"] = series
        d = "" if r["ic_rec"] is None or r["ic"] is None else f" Δrec={abs(r['ic']-r['ic_rec']):.1e}"
        print(f"  [{i:2d}/{len(rows)}] {r['mode']:11s} {str(r['ic'])[:9]:>9s}{d:>14s} "
              f"{','.join(r['flags']) or '-'} | {r['expr'][:64]}", flush=True)

    (OUT / "audit_batch.json").write_text(json.dumps(rows, indent=2, default=str))
    if ic_series:
        pd.DataFrame(ic_series).to_parquet(OUT / "ic_series_batch.parquet")
    print(f"\n[audit] tersimpan → {OUT/'audit_batch.json'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
