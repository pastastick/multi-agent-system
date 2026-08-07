"""Agregasi hasil `lab/frontend_probe.py` menjadi tabel siap-Bab-4.

Unit analisis DIBEDAKAN dengan sengaja (AUDIT_KRITIS §3.3):
  - per-EKSPRESI  : untuk laju cacat / gate / faktor mati (pengamatan memang
                    per-ekspresi di sana).
  - per-RUN       : untuk mutu sinyal. 70% ekspresi berada di satu klaster
                    sinyal, jadi ekspresi BUKAN pengamatan independen; run
                    (= trajectory) adalah unit yang benar untuk uji beda.

    python lab/analyze_gpu.py --glob 'frontend_g4_*.json' --by comm_mode
    python lab/analyze_gpu.py --glob 'frontend_g2_*.json' --by latent_steps
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def load(patterns: list[str]) -> list[dict]:
    runs = []
    for pat in patterns:
        for p in sorted(OUT.glob(pat)):
            doc = json.loads(p.read_text())
            for r in doc["runs"]:
                r["_file"] = p.name
                r.setdefault("tag", doc["args"].get("tag"))
                r["_prompts"] = Path(r.get("prompts", "")).name
                runs.append(r)
    if not runs:
        sys.exit(f"tak ada file cocok: {patterns} di {OUT}")
    return runs


def arm_of(r: dict, keys: list[str]) -> str:
    return " | ".join(f"{k}={r.get(k)}" for k in keys)


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    """t Welch + derajat bebas Welch–Satterthwaite."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    if va + vb == 0:
        return float("nan"), float("nan")
    t = (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df


def mannwhitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """z Mann-Whitney (koreksi kontinuitas) + p dua sisi via aproksimasi normal."""
    if not a or not b:
        return float("nan"), float("nan")
    both = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    vals = [v for v, _ in both]
    rank_of = [0.0] * len(both)
    while i < len(both):
        j = i
        while j + 1 < len(both) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank_of[k] = avg
        i = j + 1
    r1 = sum(rank_of[k] for k in range(len(both)) if both[k][1] == 0)
    n1, n2 = len(a), len(b)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sd == 0:
        return float("nan"), float("nan")
    z = (u1 - mu - math.copysign(0.5, u1 - mu)) / sd
    p = math.erfc(abs(z) / math.sqrt(2))
    ranks.clear()
    return z, p


def signal_clusters(exprs: list[str], series_files: list[Path], thr: float = 0.7) -> int | None:
    """Jumlah klaster sinyal: union-find atas |Spearman| deret IC harian > thr.
    Ini ukuran CAKUPAN PENCARIAN (AUDIT_KRITIS §2.4), bukan kekuatan sinyal."""
    try:
        import pandas as pd
    except ImportError:
        return None
    frames = [pd.read_parquet(f) for f in series_files if f.exists()]
    if not frames:
        return None
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    cols = [e for e in dict.fromkeys(exprs) if e in df.columns]
    if not cols:
        return None
    sub = df[cols].dropna(how="all")
    if sub.shape[1] == 0:
        return None
    corr = sub.corr(method="spearman").abs()
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


def cost_of_run(r: dict) -> tuple[float, bool]:
    """A6 — TOKEN DIPROSES satu run (prompt + output, seluruh agen).

    Kembalikan (token, exact). `exact=False` untuk artefak lama yang belum
    merekam `n_in_tok`: di sana panjang prompt dihampiri dari `kv_len` (panjang
    KV setelah agen selesai) dikurangi output — hampiran ini menghitung terlalu
    rendah untuk mode `text` (tanpa KV), jadi kolomnya ditandai '~'.
    """
    tr = r.get("agent_trace") or []
    if not tr:
        return float("nan"), True
    if all("n_in_tok" in t for t in tr):
        return float(sum(t.get("n_in_tok", 0) + t.get("n_out_tok", 0) for t in tr)), True
    tot = 0.0
    for t in tr:
        tot += max(t.get("kv_len", 0), t.get("n_out_tok", 0)) or t.get("n_out_tok", 0)
    return tot, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=["frontend_*.json"])
    ap.add_argument("--by", nargs="+", default=["comm_mode"],
                    help="kunci pembentuk lengan, mis. comm_mode / latent_steps / model _prompts")
    ap.add_argument("--baseline", default="random_baseline_s0.json")
    ap.add_argument("--clusters", action="store_true", help="hitung klaster sinyal (lambat)")
    ap.add_argument("--a7", action="store_true",
                    help="tambahkan sumbu A7 (kesetiaan rantai); --a7-full ikut rank-equivalence")
    ap.add_argument("--a7-full", action="store_true",
                    help="A7 termasuk rank-equivalence ke kolom mentah (evaluasi CPU, lambat)")
    args = ap.parse_args()

    runs = load(args.glob)
    series_files = sorted(OUT.glob("icseries_*.parquet"))
    arms: dict[str, list[dict]] = {}
    for r in runs:
        arms.setdefault(arm_of(r, args.by), []).append(r)

    print(f"\n{len(runs)} run dalam {len(arms)} lengan "
          f"(kunci: {', '.join(args.by)})\n")
    hdr = (f"{'lengan':<46s} {'run':>4s} {'ada':>4s} {'expr':>5s} {'gate':>6s} "
           f"{'cacat':>6s} {'hidup':>6s} {'meanIC':>8s} {'mn|IC|':>7s} {'mx|IC|':>7s} "
           f"{'IC>0':>7s} {'|IC|/run':>9s} {'detik':>7s}")
    print(hdr)
    print("-" * len(hdr))

    per_arm = {}
    for name, rs in sorted(arms.items()):
        facs = [f for r in rs for f in (r.get("factors") or [])]
        ok = [f for f in facs if f.get("ic") is not None]
        # "hidup" = punya IC DAN benar-benar membedakan saham (>2 nilai unik/hari).
        # Faktor konstan/NaN-total tidak layak masuk statistik mutu sinyal.
        alive = [f for f in ok if (f.get("n_unique") or 0) > 2]
        sem_bad = sum(1 for f in facs if f.get("sem_ok") is False)
        gate_ok = sum(1 for f in facs if f.get("passed_gate"))
        n_prod = sum(1 for r in rs if (r.get("factors") or []))   # run yang MENGHASILKAN
        # per-RUN mean |IC| atas faktor HIDUP (unit analisis yang benar)
        per_run = []
        for r in rs:
            v = [abs(f["ic"]) for f in (r.get("factors") or [])
                 if f.get("ic") is not None and (f.get("n_unique") or 0) > 2]
            if v:
                per_run.append(st.mean(v))
        dur = [r.get("duration_s", 0) for r in rs]
        per_arm[name] = {"runs": rs, "facs": facs, "ok": ok, "per_run": per_run,
                         "alive": alive}
        print(f"{name:<46s} {len(rs):>4d} {n_prod:>2d}/{len(rs):<1d} {len(facs):>5d} "
              f"{gate_ok/max(len(facs),1):>5.0%} {sem_bad/max(len(facs),1):>5.0%} "
              f"{len(alive):>3d}/{len(facs):<2d} "
              f"{(st.mean(f['ic'] for f in alive) if alive else float('nan')):>+8.4f} "
              f"{(st.mean(abs(f['ic']) for f in alive) if alive else float('nan')):>7.4f} "
              f"{(max((abs(f['ic']) for f in alive), default=float('nan'))):>7.4f} "
              f"{sum(1 for f in alive if f['ic']>0):>3d}/{len(alive):<3d} "
              f"{(st.mean(per_run) if per_run else float('nan')):>9.4f} "
              f"{(st.mean(dur) if dur else 0):>7.0f}")

    # ── degenerasi / operasional ────────────────────────────────────────────
    print(f"\n{'lengan':<46s} {'construct_s':>11s} {'kv_len':>7s} {'rep':>5s} "
          f"{'unparse':>8s} {'repair':>7s} {'err':>4s}")
    for name, d in sorted(per_arm.items()):
        rs = d["runs"]
        con = [t for r in rs for t in (r.get("agent_trace") or []) if t["agent"] == "construct"]
        unparse = sum(1 for r in rs
                      if any("unparseable" in e.get("msg", "") for e in (r.get("events") or [])))
        rep = sum(1 for r in rs if r.get("repaired"))
        err = sum(1 for r in rs if r.get("error"))
        print(f"{name:<46s} "
              f"{(st.mean(t['s'] for t in con) if con else float('nan')):>11.1f} "
              f"{(st.mean(t['kv_len'] for t in con) if con else float('nan')):>7.0f} "
              f"{(st.mean(t['rep_ratio'] for t in con) if con else float('nan')):>5.2f} "
              f"{unparse:>4d}/{len(rs):<3d} {rep:>3d}/{len(rs):<3d} {err:>4d}")

    # ── A6: biaya per FAKTOR DITERIMA ───────────────────────────────────────
    # Satu-satunya metrik yang menghukum arsitektur boros secara adil. Biaya
    # "per run" menyembunyikannya: lengan yang menghasilkan 30 faktor lolos gate
    # dalam 67 s jauh lebih murah daripada lengan 99 s untuk 11 faktor, walau
    # keduanya "1 run".
    print(f"\nA6 — biaya per faktor diterima\n"
          f"{'lengan':<46s} {'lolos':>6s} {'detik/run':>10s} {'token/run':>10s} "
          f"{'detik/fak':>10s} {'token/fak':>10s}")
    for name, d in sorted(per_arm.items()):
        rs = d["runs"]
        n_pass = sum(len(r.get("passing") or []) for r in rs)
        secs = sum(r.get("duration_s", 0) for r in rs)
        costs = [cost_of_run(r) for r in rs]
        toks = sum(c for c, _ in costs if c == c)
        exact = all(e for _, e in costs)
        m = "" if exact else "~"
        print(f"{name:<46s} {n_pass:>6d} {secs/max(len(rs),1):>10.1f} "
              f"{m + format(toks/max(len(rs),1), '.0f'):>10s} "
              f"{(secs/n_pass if n_pass else float('nan')):>10.1f} "
              f"{(m + format(toks/n_pass, '.0f') if n_pass else 'nan'):>10s}")

    # ── uji beda antar-lengan pada unit RUN ─────────────────────────────────
    names = sorted(per_arm)
    if len(names) > 1:
        print("\nWelch t pada mean|IC| per-run (unit analisis = run):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                t, df = welch(per_arm[a]["per_run"], per_arm[b]["per_run"])
                z, p = mannwhitney(per_arm[a]["per_run"], per_arm[b]["per_run"])
                print(f"  {a}  vs  {b}: t={t:+.2f} df={df:.1f} | MW z={z:+.2f} p={p:.3f}")

    # ── lantai acak ─────────────────────────────────────────────────────────
    bl = OUT / args.baseline
    if bl.exists():
        rnd_rows = json.loads(bl.read_text())
        rnd = [abs(r["ic"]) for r in rnd_rows
               if r.get("ic") is not None and (r.get("n_unique") or 0) > 2]
        print(f"\nvs lantai acak ({len(rnd)} ekspresi HIDUP, mean|IC|={st.mean(rnd):.4f}, "
              f"max={max(rnd):.4f}) — Mann-Whitney pada |IC| per-ekspresi hidup:")
        for name, d in sorted(per_arm.items()):
            llm = [abs(f["ic"]) for f in d["alive"]]
            if llm:
                z, p = mannwhitney(llm, rnd)
                print(f"  {name:<44s} n={len(llm):>3d} mean|IC|={st.mean(llm):.4f} "
                      f"z={z:+.2f} p={p:.3f} "
                      f"{'(tak beda)' if p > 0.05 else '(BEDA)'}")

    if args.clusters:
        print("\nKlaster sinyal (|Spearman deret IC| > 0.7) — cakupan pencarian:")
        for name, d in sorted(per_arm.items()):
            exprs = [f["expression"] for f in d["alive"]]
            k = signal_clusters(exprs, series_files)
            print(f"  {name:<44s} faktor hidup={len(exprs):>3d} klaster={k}")

    # ── A7: kesetiaan rantai ────────────────────────────────────────────────
    if args.a7 or args.a7_full:
        from lab.chain_fidelity import annotate_runs, looks_degenerate
        annotate_runs(runs, rank_equiv=args.a7_full)
        print(f"\nA7 — kesetiaan rantai\n"
              f"{'lengan':<46s} {'expr':>5s} {'var_rec':>8s} {'var_prec':>9s} "
              f"{'horizon':>8s} {'palette':>8s} {'raw≈':>7s} {'hip.rusak':>10s}")
        for name, d in sorted(per_arm.items()):
            facs = [f for f in d["facs"] if f.get("expression")]

            def _m(key, fs=facs):
                v = [f[key] for f in fs if f.get(key) is not None]
                return st.mean(v) if v else float("nan")

            hz = [f["horizon_ok"] for f in facs if f.get("horizon_ok") is not None]
            rs_ = d["runs"]
            bad = sum(1 for r in rs_ if looks_degenerate(r.get("hypothesis") or ""))
            print(f"{name:<46s} {len(facs):>5d} {_m('var_recall'):>8.2f} "
                  f"{_m('var_precision'):>9.2f} "
                  f"{(sum(hz)/len(hz) if hz else float('nan')):>8.2f} "
                  f"{_m('palette_compliance'):>8.2f} {_m('raw_equiv_max'):>7.2f} "
                  f"{bad:>4d}/{len(rs_):<5d}")


if __name__ == "__main__":
    main()
