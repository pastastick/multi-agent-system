"""Apakah TANDA IC bertahan dari jendela seleksi (2021) ke holdout (2022–2025)?

Kenapa pertanyaan ini menentukan. AUDIT_KRITIS §S3 mengusulkan mengganti fungsi
fitness dari IC **bertanda** menjadi **|IC| dengan tanda ditetapkan di jendela
latih**. Usul itu berdiri di atas satu premis empiris: bahwa faktor ber-IC
negatif kuat adalah alfa yang bisa diperdagangkan setelah tandanya dibalik —
yang hanya benar bila tandanya STABIL di luar jendela seleksi. Kalau tandanya
berbalik, membalik tanda di jendela latih adalah overfitting, dan seluruh usul
S2/S3 gugur.

AUDIT §2.6 sudah menguji ini dan menemukan tanda bertahan 100% — tetapi pada
**12 faktor** dari batch lama, dipilih sebagai 12 TERKUAT. Dua kelemahan
sekaligus: n kecil, dan seleksi pada |IC| tertinggi persis kelompok yang paling
mungkin stabil karena alasan sepele (faktor kuat = faktor bervolume/likuiditas
mentah). Alat ini menutup keduanya: korpus front-end pasca-perbaikan, dan
kelompok pembanding berlapis kekuatan — bukan hanya yang terkuat.

Yang dilaporkan:
  · kesesuaian tanda seleksi↔holdout, dipecah per lapis |IC| seleksi;
  · Spearman IC-seleksi vs IC-holdout (apakah PERINGKAT bertahan, bukan cuma tanda);
  · lantai acak untuk sumbu yang sama — karena kalau ekspresi ACAK juga
    mempertahankan tandanya, stabilitas itu sifat data, bukan prestasi sistem.

    PYTHONPATH=backend .venv/bin/python lab/sign_persistence.py --top 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
QL_ROOT = LAB.parent
for p in (str(QL_ROOT), str(QL_ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = LAB / "out"
HOLDOUT = ("2022-01-01", "2025-12-26")   # = split test QuantaAlpha


def collect_llm() -> dict[str, float]:
    """expr → IC jendela seleksi, untuk faktor HIDUP di seluruh korpus."""
    best: dict[str, float] = {}
    for path in sorted(OUT.glob("frontend_*.json")):
        doc = json.loads(path.read_text())
        for r in doc["runs"]:
            for f in r.get("factors", []) or []:
                e, ic = f.get("expression", ""), f.get("ic")
                if e and ic is not None and (f.get("n_unique") or 0) > 2:
                    best[e] = float(ic)
    return best


def collect_random() -> dict[str, float]:
    rows = json.loads((OUT / "random_baseline_s0.json").read_text())
    return {r["expr"]: float(r["ic"]) for r in rows
            if r.get("ic") is not None and (r.get("n_unique") or 0) > 2}


def strata(items: list[tuple[str, float]], top: int) -> list[tuple[str, float]]:
    """Ambil `top` ekspresi BERLAPIS kekuatan |IC|, bukan `top` terkuat saja.

    Memilih hanya yang terkuat akan menguji premis S3 pada kelompok yang paling
    menguntungkannya. Sepertiga kuat / sepertiga menengah / sepertiga lemah
    memberi uji yang bisa gagal.
    """
    ordered = sorted(items, key=lambda kv: -abs(kv[1]))
    n = len(ordered)
    per = max(top // 3, 1)
    kuat = ordered[:per]
    tengah = ordered[n // 2 - per // 2: n // 2 - per // 2 + per]
    lemah = ordered[-per:]
    out, seen = [], set()
    for group in (kuat, tengah, lemah):
        for e, ic in group:
            if e not in seen:
                seen.add(e)
                out.append((e, ic))
    return out


def score_holdout(pairs: list[tuple[str, float]], budget: int, label: str):
    from lab.core import Lab
    from lab.frontend_probe import _time_budget

    lab = Lab(mode="fast", window=HOLDOUT)
    rows = []
    t0 = time.time()
    for i, (e, ic_sel) in enumerate(pairs, 1):
        try:
            with _time_budget(budget):
                res = lab.ic(e)
        except TimeoutError:
            res = None
        rows.append({
            "expr": e, "ic_seleksi": ic_sel,
            "ic_holdout": None if res is None else res.ic,
            "t_holdout": None if res is None else res.tstat,
            "n_unique_holdout": None if res is None else res.n_unique,
        })
        if i % 10 == 0 or i == len(pairs):
            print(f"  [{label} {i}/{len(pairs)}] ({time.time() - t0:.0f}s)", flush=True)
    return rows


def report(rows: list[dict], label: str) -> dict:
    import statistics as st

    ok = [r for r in rows
          if r["ic_holdout"] is not None and (r["n_unique_holdout"] or 0) > 2]
    if len(ok) < 3:
        print(f"\n[{label}] terlalu sedikit faktor hidup di holdout ({len(ok)})")
        return {"label": label, "n": len(ok)}

    same = [r for r in ok if r["ic_seleksi"] * r["ic_holdout"] > 0]
    # Spearman peringkat IC bertanda antar-jendela
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr([r["ic_seleksi"] for r in ok],
                              [r["ic_holdout"] for r in ok]).statistic)
    except Exception:  # noqa: BLE001
        rho = float("nan")

    # per lapis kekuatan |IC| seleksi
    ordered = sorted(ok, key=lambda r: -abs(r["ic_seleksi"]))
    third = max(len(ordered) // 3, 1)
    lapis = {"kuat": ordered[:third],
             "tengah": ordered[third:2 * third],
             "lemah": ordered[2 * third:]}

    print(f"\n=== {label} — n={len(ok)} faktor hidup di kedua jendela ===")
    print(f"  tanda bertahan   : {len(same)}/{len(ok)} = {len(same)/len(ok):.1%}")
    print(f"  Spearman IC seleksi ↔ holdout : {rho:+.3f}")
    print(f"  mean |IC| seleksi {st.mean(abs(r['ic_seleksi']) for r in ok):.5f} → "
          f"holdout {st.mean(abs(r['ic_holdout']) for r in ok):.5f}")
    print(f"  {'lapis':8s} {'n':>4s} {'tanda bertahan':>16s} {'mean|IC| sel':>13s} "
          f"{'mean|IC| hold':>14s}")
    per_lapis = {}
    for name, grp in lapis.items():
        if not grp:
            continue
        s = sum(1 for r in grp if r["ic_seleksi"] * r["ic_holdout"] > 0)
        per_lapis[name] = {"n": len(grp), "tanda_bertahan": s,
                           "frac": s / len(grp)}
        print(f"  {name:8s} {len(grp):4d} {s:>10d}/{len(grp):<4d} "
              f"{st.mean(abs(r['ic_seleksi']) for r in grp):13.5f} "
              f"{st.mean(abs(r['ic_holdout']) for r in grp):14.5f}")
    return {"label": label, "n": len(ok), "tanda_bertahan": len(same),
            "frac_tanda": len(same) / len(ok), "spearman": rho,
            "per_lapis": per_lapis, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=60,
                    help="jumlah ekspresi per sumber (berlapis kuat/tengah/lemah)")
    ap.add_argument("--budget", type=int, default=180)
    ap.add_argument("--skip-random", action="store_true")
    args = ap.parse_args()

    llm = collect_llm()
    print(f"[korpus] LLM: {len(llm)} ekspresi hidup unik di jendela seleksi")
    pairs_llm = strata(list(llm.items()), args.top)
    print(f"[korpus] diuji di holdout {HOLDOUT[0]}..{HOLDOUT[1]}: "
          f"{len(pairs_llm)} (berlapis)", flush=True)

    res = {}
    rows_llm = score_holdout(pairs_llm, args.budget, "LLM")
    res["llm"] = report(rows_llm, "LLM (korpus front-end)")

    if not args.skip_random:
        rnd = collect_random()
        pairs_rnd = strata(list(rnd.items()), args.top)
        print(f"\n[korpus] acak: {len(rnd)} hidup → {len(pairs_rnd)} diuji", flush=True)
        rows_rnd = score_holdout(pairs_rnd, args.budget, "acak")
        res["acak"] = report(rows_rnd, "ACAK (lantai)")

    (OUT / "sign_persistence.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nlaporan → {OUT / 'sign_persistence.json'}")
    print("\nBacaan: tanda bertahan tinggi DAN Spearman positif → premis S3 "
          "(fitness |IC| dgn tanda dari jendela latih) didukung.\n"
          "Kalau lantai ACAK sama stabilnya, stabilitas itu sifat data — "
          "bukan bukti sistemnya menemukan sesuatu.")


if __name__ == "__main__":
    main()
