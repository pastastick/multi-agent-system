"""Null model: sampling ekspresi ACAK dari DSL yang sama.

Pertanyaan yang dijawab: apakah sistem multi-agent LLM menghasilkan faktor yang
lebih baik daripada mengambil ekspresi acak dari ruang DSL yang sama?
Tanpa kontrol ini, klaim apa pun tentang "kualitas faktor" (dan tentang KV vs TEXT)
tidak punya lantai pembanding.

Generator sengaja hanya menghasilkan ekspresi yang SEHAT secara numerik
(window >= 2, kondisi selalu perbandingan, tanpa ambang absolut) — jadi
perbandingannya konservatif: LLM diadu dengan versi acak yang sudah "dibersihkan".

    .venv/bin/python lab/random_baseline.py 300 [seed]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.core import Lab  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

VARS = ["$open", "$high", "$low", "$close", "$volume", "$return"]
WINDOWS = [3, 5, 10, 20, 60]

TS1 = ["TS_MEAN", "TS_STD", "TS_ZSCORE", "TS_RANK", "TS_MEDIAN", "TS_MAX",
       "TS_MIN", "TS_SUM", "TS_ARGMAX", "TS_ARGMIN", "TS_MAD", "TS_VAR",
       "DELTA", "DELAY", "TS_PCTCHANGE", "EMA", "WMA", "DECAYLINEAR",
       "HIGHDAY", "LOWDAY", "SUMAC"]
# REGBETA/REGRESI sengaja DIKELUARKAN: joblib per-instrument, >10 menit/ekspresi
# di CPU. Dicatat sebagai keterbatasan null model (ruang sampel sedikit lebih kecil
# dari ruang yang tersedia bagi LLM).
TS2 = ["TS_CORR", "TS_COVARIANCE"]
CS1 = ["RANK", "ZSCORE"]
MATH1 = ["LOG", "SQRT", "SIGN", "ABS"]


def leaf(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.12:
        a, b = rng.sample(["$high", "$low", "$close", "$open"], 2)
        return f"({a} - {b})"
    return rng.choice(VARS)


def build(rng: random.Random, depth: int) -> str:
    if depth <= 0:
        return leaf(rng)
    r = rng.random()
    if r < 0.42:
        return f"{rng.choice(TS1)}({build(rng, depth-1)}, {rng.choice(WINDOWS)})"
    if r < 0.55:
        return f"{rng.choice(CS1)}({build(rng, depth-1)})"
    if r < 0.65:
        return f"{rng.choice(TS2)}({build(rng, depth-1)}, {build(rng, depth-1)}, {rng.choice(WINDOWS)})"
    if r < 0.72:
        return f"{rng.choice(MATH1)}({build(rng, depth-1)})"
    if r < 0.88:
        op = rng.choice(["*", "-", "+", "/"])
        return f"({build(rng, depth-1)} {op} {build(rng, depth-1)})"
    # gate kondisional — kondisi SELALU perbandingan antar besaran sebanding
    a = build(rng, depth - 1)
    thr = rng.choice(["0", "0.5", "1"]) if rng.random() < 0.5 else build(rng, 0)
    return f"(({a}) > ({thr})) ? ({build(rng, depth-1)}) : (0)"


def sample(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        e = build(rng, rng.choice([1, 1, 2, 2, 2, 3]))
        if e in seen or "$" not in e:
            continue
        seen.add(e)
        out.append(e)
    return out


class time_budget:
    """Batas waktu per-ekspresi supaya satu operator lambat tak menyandera sweep."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        import signal

        def _raise(signum, frame):  # noqa: ARG001
            raise TimeoutError(f"melebihi {self.seconds}s")

        signal.signal(signal.SIGALRM, _raise)
        signal.alarm(self.seconds)

    def __exit__(self, *exc):
        import signal

        signal.alarm(0)
        return False


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    OUT.mkdir(parents=True, exist_ok=True)
    lab = Lab(mode="fast")
    rows = []
    for i, e in enumerate(sample(n, seed), 1):
        with time_budget(90):
            r = lab.ic(e)
        rows.append({"expr": e, "ic": r.ic, "icir": r.icir, "tstat": r.tstat,
                     "n_days": r.n_days, "coverage": r.coverage,
                     "n_unique": r.n_unique, "error": r.error})
        if i % 10 == 0 or i == n:
            ok = [x["ic"] for x in rows if x["ic"] is not None]
            print(f"[{i}/{n}] valid={len(ok)} "
                  f"mean|IC|={sum(abs(v) for v in ok)/max(len(ok),1):.4f} "
                  f"max|IC|={max((abs(v) for v in ok), default=0):.4f}", flush=True)
    (OUT / f"random_baseline_s{seed}.json").write_text(json.dumps(rows, indent=2))
    print(f"tersimpan → {OUT/f'random_baseline_s{seed}.json'}")


if __name__ == "__main__":
    main()
