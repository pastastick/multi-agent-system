"""A10 — apakah sistem benar-benar MEMBACA arah riset yang diberikan?

G1 sudah menunjukkan gejalanya di level vektor laten (`identical_across_prompt`
3/3 pada Qwen3-8B mode `raw`: tiga arah berbeda menghasilkan jalur laten yang
identik). Sumbu ini mengukurnya di level yang benar-benar penting — **keluaran**:
himpunan ekspresi yang akhirnya dihasilkan.

── KENAPA BUTUH KONTROL DALAM-ARAH ─────────────────────────────────────────
Mengukur "dua arah berlawanan menghasilkan keluaran yang mirip" saja TIDAK
membuktikan apa-apa. Sistem ini stokastik: dua run dengan arah yang SAMA pun
tidak menghasilkan ekspresi yang sama. Yang harus dibandingkan adalah:

    jarak ANTAR-arah   vs   jarak DALAM-arah (seed berbeda, arah sama)

  - antar ≈ dalam  → arah tidak berpengaruh; yang terukur cuma derau sampling.
                     Sistem MENGABAIKAN masukannya.
  - antar >  dalam  → arah menggeser keluaran melebihi derau. Sistem MEMBACA
                     masukannya, dan selisihnya mengukur seberapa kuat.

Ini logika yang sama dengan lantai acak di `lab/random_baseline.py`: sebuah
angka kemiripan tanpa pembanding derau tidak bisa ditafsirkan.

── DUA JARAK YANG DIUKUR ───────────────────────────────────────────────────
  Jaccard fungsi   1 − |A∩B|/|A∪B| atas himpunan fungsi DSL yang dipakai satu
                   run. Menangkap "apakah bentuk konstruksinya bergeser".
  Korelasi deret   |Spearman| rata-rata antar deret IC harian faktor lintas
                   run. Menangkap "apakah SINYALnya bergeser" — dua run bisa
                   memakai fungsi berbeda tetapi menghasilkan sinyal kembar.

Keduanya perlu: yang pertama bisa bergerak karena kosmetik, yang kedua tidak.

    # 1) jalankan dua arah berlawanan (lihat DIRECTIONS di frontend_probe.py)
    python lab/frontend_probe.py --directions opp_mom,opp_rev --seeds 0,1,2 \
        --comm-mode kv --latent-steps 10 --latent-mode gumbel --tag a10
    # 2) analisis
    python lab/direction_sensitivity.py --tag a10
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics as st
import sys
from pathlib import Path

QL = Path(__file__).resolve().parent.parent
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
for p in (str(QL), str(QL / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = QL / "lab" / "out"

_FUNC = re.compile(r"\b([A-Z][A-Z0-9_]{1,})\s*\(")


def funcs_of(expr: str) -> set[str]:
    return set(_FUNC.findall(expr or ""))


def run_funcs(run: dict) -> set[str]:
    """Himpunan fungsi DSL yang dipakai SELURUH ekspresi satu run."""
    out: set[str] = set()
    for f in run.get("factors") or []:
        out |= funcs_of(f.get("expression", ""))
    return out


def jaccard_dist(a: set, b: set) -> float | None:
    if not a and not b:
        return None
    return 1.0 - len(a & b) / len(a | b)


def series_corr(runs_a: dict, runs_b: dict, series) -> float | None:
    """|Spearman| RATA-RATA antar deret IC faktor run A vs run B.

    Dipakai apa adanya sebagai ukuran kemiripan SINYAL: 1 = dua run
    menghasilkan sinyal yang secara statistik sama meski rumusnya beda.
    """
    import pandas as pd

    ea = [f["expression"] for f in (runs_a.get("factors") or [])
          if f.get("expression") in series.columns]
    eb = [f["expression"] for f in (runs_b.get("factors") or [])
          if f.get("expression") in series.columns]
    vals = []
    for x in ea:
        for y in eb:
            if x == y:
                continue
            c = series[x].corr(series[y], method="spearman")
            if c == c:                       # saring NaN
                vals.append(abs(c))
    return st.mean(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a10")
    ap.add_argument("--dirs", default="",
                    help="batasi ke arah tertentu, mis. 'opp_mom,opp_rev'")
    a = ap.parse_args()

    import pandas as pd

    doc = json.loads((OUT / f"frontend_{a.tag}.json").read_text())
    runs = doc["runs"]
    if a.dirs:
        want = {d.strip() for d in a.dirs.split(",")}
        runs = [r for r in runs if r.get("direction") in want]

    spath = OUT / f"icseries_{a.tag}.parquet"
    series = pd.read_parquet(spath) if spath.exists() else pd.DataFrame()

    dirs = sorted({r["direction"] for r in runs})
    by_dir = {d: [r for r in runs if r["direction"] == d] for d in dirs}
    print(f"[a10] tag={a.tag} arah={dirs} "
          f"run/arah={[len(by_dir[d]) for d in dirs]} "
          f"deret_IC={series.shape[1] if len(series) else 0} faktor")

    # ── pasangan DALAM-arah (kontrol derau) & ANTAR-arah ────────────────────
    within_j, within_c, across_j, across_c = [], [], [], []
    for d in dirs:
        for x, y in itertools.combinations(by_dir[d], 2):
            j = jaccard_dist(run_funcs(x), run_funcs(y))
            if j is not None:
                within_j.append(j)
            c = series_corr(x, y, series) if len(series) else None
            if c is not None:
                within_c.append(c)
    for d1, d2 in itertools.combinations(dirs, 2):
        for x in by_dir[d1]:
            for y in by_dir[d2]:
                j = jaccard_dist(run_funcs(x), run_funcs(y))
                if j is not None:
                    across_j.append(j)
                c = series_corr(x, y, series) if len(series) else None
                if c is not None:
                    across_c.append(c)

    def _s(v):
        return f"{st.mean(v):.3f} (n={len(v)})" if v else "—"

    print("\n  ukuran                          DALAM-arah        ANTAR-arah")
    print(f"  jarak Jaccard fungsi (↑ beda)   {_s(within_j):<18s}{_s(across_j)}")
    print(f"  |Spearman| deret IC  (↓ beda)   {_s(within_c):<18s}{_s(across_c)}")

    verdict, dj = "tak dapat disimpulkan", None
    if within_j and across_j:
        dj = st.mean(across_j) - st.mean(within_j)
        # Ambang: selisih harus melebihi setengah simpangan baku pasangan
        # dalam-arah. Dengan n sekecil ini uji-t formal tidak jujur; yang bisa
        # dipertanggungjawabkan hanyalah "apakah efeknya sebanding dengan derau".
        noise = st.pstdev(within_j) if len(within_j) > 1 else 0.0
        verdict = ("MEMBACA arah (antar > dalam, melebihi derau)"
                   if dj > max(0.05, 0.5 * noise)
                   else "TIDAK terbukti membaca arah (antar ≈ dalam)")
        print(f"\n  selisih Jaccard (antar − dalam) = {dj:+.3f} "
              f"| sebaran dalam-arah = {noise:.3f}")
    print(f"  VONIS: {verdict}")

    # fungsi yang KHAS per arah — diagnosis kualitatif kalau vonisnya negatif
    print("\n  fungsi yang hanya muncul di satu arah:")
    sets = {d: set().union(*[run_funcs(r) for r in by_dir[d]]) if by_dir[d] else set()
            for d in dirs}
    for d in dirs:
        others = set().union(*[sets[o] for o in dirs if o != d]) if len(dirs) > 1 else set()
        uniq = sorted(sets[d] - others)
        print(f"    {d:10s} n_fungsi={len(sets[d]):2d} khas={uniq if uniq else '(tidak ada)'}")

    res = {"tag": a.tag, "dirs": dirs,
           "within_jaccard": within_j, "across_jaccard": across_j,
           "within_corr": within_c, "across_corr": across_c,
           "delta_jaccard": dj, "verdict": verdict,
           "funcs_per_dir": {d: sorted(sets[d]) for d in dirs}}
    path = OUT / f"direction_sensitivity_{a.tag}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\ntersimpan → {path}")


if __name__ == "__main__":
    main()
