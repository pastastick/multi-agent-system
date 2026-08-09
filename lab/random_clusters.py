"""Lantai acak untuk sumbu CAKUPAN (A3), bukan cuma untuk sumbu mutu (A1).

Lubang yang ditutup alat ini. AUDIT_KRITIS §2.5 memberi lantai acak untuk
**kekuatan sinyal** (mean |IC| = 0,0170) dan sejak itu setiap lengan diadu
dengan angka tersebut. Tetapi seluruh rantai keputusan arsitektur di
RENCANA_PERBAIKAN — A8/B13/B16, dan pembelaan terhadap agen `design` — bertumpu
pada sumbu yang BERBEDA: **klaster sinyal** (A3), karena varians-nya rendah pada
n=6 sementara |IC| tidak. Sumbu itu tidak pernah punya lantai acak. Jadi
kalimat "`full` menyebar ke 20 klaster, `nodesign` cuma 7" selama ini dibaca
sebagai keunggulan tanpa ada pembanding yang mengatakan berapa klaster yang
didapat **tanpa** agen sama sekali.

Kenapa perbandingan mentah tidak cukup: jumlah klaster tumbuh dengan jumlah
ekspresi. Lengan dengan 33 faktor hidup hampir pasti mengalahkan lengan dengan
14 faktor hidup pada hitungan klaster mentah, tanpa satu pun kaitannya dengan
mutu pencarian. Karena itu pembandingnya adalah **bootstrap n-tercocok**: untuk
lengan ber-k faktor hidup, ambil k ekspresi acak dari kolam acak, hitung
klasternya, ulang B kali → sebaran nol. Yang dilaporkan adalah posisi lengan itu
di dalam sebaran tersebut.

    PYTHONPATH=backend .venv/bin/python lab/random_clusters.py
    PYTHONPATH=backend .venv/bin/python lab/random_clusters.py --boot 500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
QL_ROOT = LAB.parent
for p in (str(QL_ROOT), str(QL_ROOT / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = LAB / "out"


def _clusters(cols: list[str], df, thr: float = 0.7) -> int:
    """Union-find atas |Spearman| deret IC > thr — definisi identik
    `analyze_gpu.signal_clusters`, tetapi bekerja dari DataFrame yang sudah
    di memori (bootstrap memanggilnya ratusan kali)."""
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    if not cols:
        return 0
    corr = df[cols].corr(method="spearman").abs()
    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if v == v and v > thr:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return len({find(c) for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-file", default="random_baseline_s0.json")
    ap.add_argument("--boot", type=int, default=300, help="ulangan bootstrap")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--budget", type=int, default=90)
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--score-only", action="store_true",
                    help="bangun deret IC kolam acak lalu berhenti (bagian mahal); "
                         "perbandingan lengan menunggu icseries_* lengan LLM siap")
    args = ap.parse_args()

    import pandas as pd
    from lab.core import Lab

    rows = json.loads((OUT / args.seed_file).read_text())
    alive = [r for r in rows
             if r.get("ic") is not None and (r.get("n_unique") or 0) > 2]
    print(f"[acak] {len(rows)} ekspresi → {len(alive)} hidup", flush=True)

    # ── deret IC untuk kolam acak (di-cache; ini bagian yang mahal) ────────
    series_path = OUT / f"icseries_random_{Path(args.seed_file).stem}.parquet"
    # Simpan berkala + lanjutkan dari yang sudah ada: skoring 271 ekspresi
    # memakan ±1 jam, dan ekspresi ber-BB_*/REGRESI bisa memicu OOM. Menyimpan
    # hanya di akhir berarti satu kematian menghapus seluruh jam itu.
    series: dict[str, pd.Series] = {}
    if series_path.exists():
        prev = pd.read_parquet(series_path)
        series = {c: prev[c].dropna() for c in prev.columns}
        print(f"[acak] melanjutkan: {len(series)} deret sudah ada", flush=True)

    todo = [r for r in alive if r["expr"] not in series]
    if todo:
        from lab.frontend_probe import _time_budget
        lab = Lab(mode="fast")
        t0 = time.time()
        for i, r in enumerate(todo, 1):
            e = r["expr"]
            try:
                with _time_budget(args.budget):
                    _res, ser = lab.ic_full(e)
            except TimeoutError:
                ser = None
            if ser is not None:
                series[e] = ser
            if i % 20 == 0 or i == len(todo):
                pd.DataFrame(series).to_parquet(series_path)
                print(f"  [{i}/{len(todo)}] deret={len(series)} "
                      f"({time.time() - t0:.0f}s, disimpan)", flush=True)
        pd.DataFrame(series).to_parquet(series_path)
    rnd_df = pd.DataFrame(series)
    print(f"[acak] deret IC → {series_path} ({rnd_df.shape[1]} kolom)", flush=True)

    if args.score_only:
        print(f"[acak] --score-only: {rnd_df.shape[1]} deret siap, berhenti di sini.")
        return

    pool = list(rnd_df.columns)
    k_full = _clusters(pool, rnd_df, args.thr)
    print(f"\n[acak] kolam {len(pool)} ekspresi → {k_full} klaster "
          f"({k_full / len(pool):.3f} klaster/ekspresi)\n", flush=True)

    # ── lengan LLM ────────────────────────────────────────────────────────
    llm_frames = [pd.read_parquet(f) for f in sorted(OUT.glob("icseries_*.parquet"))
                  if "random" not in f.name]
    if not llm_frames:
        print("Tidak ada icseries_*.parquet lengan LLM — jalankan lab/rescore_all.py dulu.")
        return
    llm_df = pd.concat(llm_frames, axis=1)
    llm_df = llm_df.loc[:, ~llm_df.columns.duplicated()]

    rng = random.Random(args.rng)
    results = []
    print(f"{'lengan':30s} {'hidup':>6s} {'klaster':>8s} {'acak n-cocok (p5–p95)':>24s} "
          f"{'posisi':>8s}")
    print("-" * 82)
    for path in sorted(OUT.glob("frontend_*.json")):
        tag = path.stem[len("frontend_"):]
        doc = json.loads(path.read_text())
        exprs = []
        for r in doc["runs"]:
            for f in r.get("factors", []) or []:
                e = f.get("expression", "")
                if (e and f.get("ic") is not None and (f.get("n_unique") or 0) > 2
                        and e in llm_df.columns):
                    exprs.append(e)
        exprs = list(dict.fromkeys(exprs))
        n = len(exprs)
        if n < 3 or n > len(pool):
            continue
        k_obs = _clusters(exprs, llm_df, args.thr)
        boots = [_clusters(rng.sample(pool, n), rnd_df, args.thr)
                 for _ in range(args.boot)]
        boots.sort()
        lo, hi = boots[int(0.05 * len(boots))], boots[int(0.95 * len(boots)) - 1]
        med = boots[len(boots) // 2]
        # fraksi ulangan acak yang MENYAMAI ATAU MELEBIHI lengan LLM
        p_ge = sum(1 for b in boots if b >= k_obs) / len(boots)
        results.append({"tag": tag, "n_hidup": n, "klaster": k_obs,
                        "acak_median": med, "acak_p5": lo, "acak_p95": hi,
                        "p_acak_ge_llm": p_ge})
        print(f"{tag:30s} {n:6d} {k_obs:8d} "
              f"{med:>10d} ({lo}–{hi})   p={p_ge:.3f}")

    rep = OUT / "random_clusters_report.json"
    rep.write_text(json.dumps({
        "thr": args.thr, "boot": args.boot,
        "kolam_acak": {"n": len(pool), "klaster": k_full,
                       "per_ekspresi": k_full / len(pool)},
        "lengan": results,
    }, indent=2))
    print(f"\nlaporan → {rep}")
    print("\np = fraksi sampel acak n-tercocok yang klasternya ≥ lengan LLM.")
    print("p kecil  → lengan LLM menyebar LEBIH luas dari acak pada n yang sama.")
    print("p besar  → acak menyamai/melampaui; keunggulan cakupan tidak terbukti.")


if __name__ == "__main__":
    main()
