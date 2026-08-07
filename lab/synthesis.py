"""Sintesis: sistem multi-agent LLM vs sampling acak dari DSL yang sama,
plus uji ketahanan pada holdout SEJATI (2022-2025).

Menjawab tiga hal yang belum pernah diuji di proyek ini:
  1. Apakah pipeline LLM mengalahkan lantai acak? (kontrol yang selama ini hilang)
  2. Apakah perbedaan antar comm_mode (kv / kv_and_text / text) signifikan?
  3. Apakah faktor bertahan di luar jendela yang dipakai untuk SELEKSI?
     `_oos_window()` membaca segmen test = 2021 — jendela yang SAMA dipakai
     evolution untuk memilih parent. Jadi angka Bab 4 adalah metrik seleksi,
     bukan holdout. Holdout sejati = 2022-01-01..2025-12-26 (split test QuantaAlpha).

    .venv/bin/python lab/synthesis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.core import Lab  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
HOLDOUT = ("2022-01-01", "2025-12-26")


def _stats(vals: list[float], label: str) -> dict:
    a = np.array([v for v in vals if v is not None and np.isfinite(v)])
    if a.size == 0:
        return {"label": label, "n": 0}
    return {
        "label": label, "n": int(a.size),
        "mean_IC": float(a.mean()), "mean_absIC": float(np.abs(a).mean()),
        "median_absIC": float(np.median(np.abs(a))),
        "max_absIC": float(np.abs(a).max()),
        "frac_IC_pos": float((a > 0).mean()),
    }


def mann_whitney(x: list[float], y: list[float]) -> tuple[float, float]:
    """U-test dua sisi via aproksimasi normal (tanpa scipy)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([x, y])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # koreksi ties
    df = pd.Series(allv)
    ranks = df.rank().to_numpy()
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu) / sigma if sigma > 0 else np.nan
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2)) if np.isfinite(z) else np.nan
    return float(z), float(p)


def main() -> None:
    audit = json.loads((OUT / "audit_batch.json").read_text())
    rnd_files = sorted(OUT.glob("random_baseline_s*.json"))
    rnd = [r for f in rnd_files for r in json.loads(f.read_text())]
    print(f"[sintesis] {len(audit)} faktor LLM, {len(rnd)} ekspresi acak "
          f"({len(rnd_files)} berkas)\n")

    # ── 1. cacat semantik ────────────────────────────────────────────────
    n_flag = sum(1 for r in audit if r["flags"])
    n_dead = sum(1 for r in audit if r["ic"] is None or r.get("n_unique", 9) <= 2)
    print("1) MUTU EKSPRESI (semua LOLOS 9 gate produksi)")
    print(f"   ber-cacat semantik terdeteksi : {n_flag}/{len(audit)} "
          f"({n_flag/len(audit)*100:.0f}%)")
    print(f"   mati numerik (NaN/konstan)    : {n_dead}/{len(audit)} "
          f"({n_dead/len(audit)*100:.0f}%)")
    from collections import Counter
    c = Counter(f for r in audit for f in r["flags"])
    for k, v in c.most_common():
        print(f"     - {k}: {v}")

    # ── 2. LLM vs acak ───────────────────────────────────────────────────
    llm_ic = [r["ic"] for r in audit if r["ic"] is not None]
    rnd_ic = [r["ic"] for r in rnd if r["ic"] is not None]
    rows = [_stats(llm_ic, "LLM (semua mode)"), _stats(rnd_ic, "acak (null model)")]
    for mode in ("text", "kv_and_text", "kv"):
        rows.append(_stats([r["ic"] for r in audit
                            if r["mode"] == mode and r["ic"] is not None], f"LLM {mode}"))
    print("\n2) LLM vs LANTAI ACAK — per-factor RankIC, jendela seleksi 2021")
    print(f"   {'sumber':22s} {'n':>3s} {'mean IC':>9s} {'mean|IC|':>9s} "
          f"{'max|IC|':>8s} {'IC>0':>6s}")
    for r in rows:
        if r["n"] == 0:
            continue
        print(f"   {r['label']:22s} {r['n']:3d} {r['mean_IC']:+9.4f} "
              f"{r['mean_absIC']:9.4f} {r['max_absIC']:8.4f} {r['frac_IC_pos']*100:5.0f}%")
    z, p = mann_whitney([abs(v) for v in llm_ic], [abs(v) for v in rnd_ic])
    print(f"   Mann-Whitney |IC| LLM vs acak: z={z:+.2f}, p={p:.3f} "
          f"→ {'BEDA signifikan' if p < 0.05 else 'TIDAK beda signifikan'}")

    # ── 3. antar comm_mode ───────────────────────────────────────────────
    print("\n3) ANTAR comm_mode (pertanyaan inti skripsi)")
    for a, b in (("kv", "text"), ("kv", "kv_and_text"), ("text", "kv_and_text")):
        xa = [abs(r["ic"]) for r in audit if r["mode"] == a and r["ic"] is not None]
        xb = [abs(r["ic"]) for r in audit if r["mode"] == b and r["ic"] is not None]
        z, p = mann_whitney(xa, xb)
        print(f"   |IC| {a:11s} vs {b:11s}: n={len(xa):2d}/{len(xb):2d} "
              f"z={z:+.2f} p={p:.3f} → {'beda' if p < 0.05 else 'TIDAK beda'}")

    # ── 4. holdout sejati ────────────────────────────────────────────────
    print(f"\n4) HOLDOUT SEJATI {HOLDOUT[0]}..{HOLDOUT[1]} "
          f"(jendela yang TIDAK dipakai seleksi)")
    lab_h = Lab(mode="fast", window=HOLDOUT)
    pairs = []
    for r in sorted((x for x in audit if x["ic"] is not None),
                    key=lambda x: -abs(x["ic"]))[:12]:
        res = lab_h.ic(r["expr"])
        pairs.append((r["expr"], r["ic"], res.ic, res.tstat))
        print(f"   sel2021={r['ic']:+.4f}  holdout={str(res.ic)[:8]:>8s} "
              f"t={('%.2f' % res.tstat) if res.tstat else 'NA':>6s} | {r['expr'][:52]}")
    ok = [(a, b) for _, a, b, _ in pairs if b is not None]
    if len(ok) >= 3:
        sel = np.array([a for a, _ in ok]); hol = np.array([b for _, b in ok])
        keep = float(np.mean(np.sign(sel) == np.sign(hol)))
        corr = float(pd.Series(sel).corr(pd.Series(hol), method="spearman"))
        print(f"   tanda IC bertahan di holdout : {keep*100:.0f}% dari {len(ok)} faktor")
        print(f"   korelasi Spearman seleksi↔holdout: {corr:+.3f}")

    json.dump({"rows": rows, "holdout": [list(p) for p in pairs]},
              open(OUT / "synthesis.json", "w"), indent=2, default=str)
    print(f"\ntersimpan → {OUT/'synthesis.json'}")


if __name__ == "__main__":
    main()
