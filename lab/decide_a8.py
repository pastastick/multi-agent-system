"""Terapkan aturan keputusan A8 (RENCANA_PERBAIKAN §Tahap 3a) secara MEKANIS.

Aturannya ditulis SEBELUM datanya ada. Skrip ini membacanya kembali dan
menghitungnya, supaya keputusan "design dipertahankan / dipangkas / diganti"
tidak bergantung pada mata yang membaca tabel — termasuk mata saya sendiri.

    python lab/decide_a8.py --comm-mode kv
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

QL = Path(__file__).resolve().parent.parent
if str(QL) not in sys.path:
    sys.path.insert(0, str(QL))

OUT = QL / "lab" / "out"

_FUNC_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,})\s*\(")
# Lengan yang TERDAFTAR DI MUKA di RENCANA_PERBAIKAN §Tahap 3a. Hanya lengan ini
# yang boleh masuk aturan keputusan formal.
ARMS = ("full", "nodesign", "direct", "innovate", "innovate_fid")
# Lengan yang lahir DARI temuan saat menjalankan A8 (guided decoding baru bisa
# dipakai setelah bug lintas-versi lm-format-enforcer diperbaiki). Dilaporkan,
# tetapi TIDAK dipakai untuk memutuskan — menambah lengan setelah melihat data
# lalu memakainya sebagai dasar keputusan adalah cara paling halus untuk menipu
# diri sendiri. Ia menjadi dasar untuk RONDE BERIKUTNYA yang didaftarkan ulang.
EXTRA_ARMS = ("full_guided", "innovate_guided")


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    if va + vb == 0:
        return float("nan"), float("nan")
    t = (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df


def historical_functions() -> set[str]:
    """Fungsi DSL yang pernah dipakai sistem SEBELUM eksperimen A8 (korpus
    G2/G3/G4/G6 + 2x2 prompt). Dipakai untuk menghitung berapa fungsi BARU yang
    dibuka tiap lengan — ukuran paling langsung dari "keluar dari kemonotonan",
    dan tidak bisa dipalsukan dengan menulis ulang idiom yang sama."""
    used: set[str] = set()
    for p in OUT.glob("frontend_*.json"):
        if p.name.startswith("frontend_a8_"):
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        for r in doc.get("runs", []):
            for f in (r.get("factors") or []):
                used.update(_FUNC_RE.findall(f.get("expression", "")))
    return used


def load_arm(comm: str, arm: str) -> list[dict] | None:
    p = OUT / f"frontend_a8_{comm}_{arm}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["runs"]


def summarise(runs: list[dict]) -> dict:
    facs = [f for r in runs for f in (r.get("factors") or [])]
    alive = [f for f in facs if f.get("ic") is not None and (f.get("n_unique") or 0) > 2]
    per_run = []
    for r in runs:
        v = [abs(f["ic"]) for f in (r.get("factors") or [])
             if f.get("ic") is not None and (f.get("n_unique") or 0) > 2]
        if v:
            per_run.append(st.mean(v))
    used = set()
    for f in facs:
        used.update(_FUNC_RE.findall(f.get("expression", "")))
    n_pass = sum(len(r.get("passing") or []) for r in runs)
    return {
        "n_runs": len(runs),
        "n_producing": sum(1 for r in runs if (r.get("factors") or [])),
        "n_expr": len(facs),
        "n_pass": n_pass,
        "gate_rate": n_pass / max(len(facs), 1),
        "n_alive": len(alive),
        "per_run": per_run,
        "mean_ic_run": st.mean(per_run) if per_run else float("nan"),
        "lib_coverage": len(used),
        "secs_per_pass": (sum(r.get("duration_s", 0) for r in runs) / n_pass
                          if n_pass else float("nan")),
        "toks_per_pass": (sum(t.get("n_in_tok", 0) + t.get("n_out_tok", 0)
                              for r in runs for t in (r.get("agent_trace") or []))
                          / n_pass if n_pass else float("nan")),
    }


def clusters(runs: list[dict], comm: str, arm: str) -> int | None:
    try:
        import pandas as pd
    except ImportError:
        return None
    p = OUT / f"icseries_a8_{comm}_{arm}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    exprs = [f["expression"] for r in runs for f in (r.get("factors") or [])
             if f.get("ic") is not None and (f.get("n_unique") or 0) > 2]
    cols = [e for e in dict.fromkeys(exprs) if e in df.columns]
    if not cols:
        return None
    corr = df[cols].dropna(how="all").corr(method="spearman").abs()
    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if v == v and v > 0.7:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return len({find(c) for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comm-mode", dest="comm", default="kv")
    a = ap.parse_args()

    hist = historical_functions()
    S, C, NEW = {}, {}, {}
    for arm in ARMS + EXTRA_ARMS:
        runs = load_arm(a.comm, arm)
        if runs is None:
            if arm in ARMS:
                print(f"[!] lengan {arm} belum ada — keputusan tak bisa dituntaskan")
            continue
        S[arm] = summarise(runs)
        C[arm] = clusters(runs, a.comm, arm)
        used = set()
        for r in runs:
            for f in (r.get("factors") or []):
                used.update(_FUNC_RE.findall(f.get("expression", "")))
        NEW[arm] = sorted(used - hist)

    hdr = (f"{'lengan':<14s} {'run':>5s} {'expr':>5s} {'gate':>6s} {'hidup':>6s} "
           f"{'|IC|/run':>9s} {'pustaka':>8s} {'baru':>5s} {'klaster':>8s} "
           f"{'dtk/fak':>8s} {'tok/fak':>8s}")
    def row(arm: str) -> None:
        s = S[arm]
        print(f"{arm:<14s} {s['n_producing']:>2d}/{s['n_runs']:<2d} {s['n_expr']:>5d} "
              f"{s['gate_rate']:>5.0%} {s['n_alive']:>6d} {s['mean_ic_run']:>9.4f} "
              f"{s['lib_coverage']:>8d} {len(NEW[arm]):>5d} {str(C.get(arm)):>8s} "
              f"{s['secs_per_pass']:>8.1f} {s['toks_per_pass']:>8.0f}")

    print(f"\nA8 ({a.comm}) — ringkasan  "
          f"(korpus lama memakai {len(hist)} fungsi)\n{hdr}\n" + "-" * len(hdr))
    for arm in ARMS:
        if arm in S:
            row(arm)
    extra = [x for x in EXTRA_ARMS if x in S]
    if extra:
        print("- " * (len(hdr) // 2))
        print("lengan lanjutan (LUAR pendaftaran; hanya dilaporkan, tidak memutuskan):")
        for arm in extra:
            row(arm)
    print("\nFungsi yang BELUM PERNAH dipakai sistem sebelum A8:")
    for arm in ARMS + EXTRA_ARMS:
        if arm in NEW:
            print(f"  {arm:<16s} {' '.join(NEW[arm]) or '(tak ada)'}")

    if "full" not in S or "nodesign" not in S:
        sys.exit("\nlengan full/nodesign belum lengkap.")
    # Penjaga: `gpu_suite` menyekor IC di CPU SETELAH semua lengan selesai, jadi
    # sebelum itu `per_run` kosong dan setiap perbandingan menghasilkan nan.
    # Tanpa penjaga ini skrip akan mencetak putusan yang terlihat yakin padahal
    # dihitung dari ketiadaan — kegagalan yang jauh lebih buruk daripada diam.
    # Hanya lengan TERDAFTAR yang menghalangi keputusan; lengan lanjutan boleh
    # belum lengkap karena memang tidak dipakai memutuskan.
    belum = [x for x in ARMS if x in S and not S[x]["per_run"]]
    if belum:
        sys.exit(f"\nBELUM BISA MEMUTUSKAN: lengan {belum} belum di-skor IC-nya "
                 f"(jalankan `lab/frontend_probe.py --score-only --tag a8_...`, "
                 f"atau tunggu gpu_suite menyelesaikan tahap skoring).")

    print("\n" + "=" * 78)
    print("GERBANG 1 — apakah `design` berkontribusi?")
    t, df = welch(S["full"]["per_run"], S["nodesign"]["per_run"])
    delta = S["full"]["mean_ic_run"] - S["nodesign"]["mean_ic_run"]
    g1 = (not (delta > 0)) or (abs(t) < 1 if t == t else True)
    print(f"  mean|IC|/run: full {S['full']['mean_ic_run']:.4f} vs "
          f"nodesign {S['nodesign']['mean_ic_run']:.4f}  (selisih {delta:+.4f}, "
          f"Welch t={t:+.2f})")
    print(f"  → kriteria 1 (tak unggul terarah / |t|<1): {'TERPENUHI' if g1 else 'tidak'}")

    mahal = (S["full"]["secs_per_pass"] > S["nodesign"]["secs_per_pass"]
             and S["full"]["toks_per_pass"] > S["nodesign"]["toks_per_pass"])
    unggul_a1 = delta > 0
    unggul_a3 = (C.get("full") or 0) > (C.get("nodesign") or 0)
    g2 = mahal and not (unggul_a1 or unggul_a3)
    print(f"  biaya: full {S['full']['secs_per_pass']:.1f}s/{S['full']['toks_per_pass']:.0f}tok "
          f"vs nodesign {S['nodesign']['secs_per_pass']:.1f}s/{S['nodesign']['toks_per_pass']:.0f}tok "
          f"per faktor diterima")
    print(f"  → kriteria 2 (lebih mahal tanpa unggul A1/A3): {'TERPENUHI' if g2 else 'tidak'}")

    design_gagal = g1 or g2
    print(f"\n  PUTUSAN: `design` {'TIDAK berkontribusi' if design_gagal else 'BERKONTRIBUSI'}")

    if not design_gagal:
        print("\n  → pertahankan rantai produksi. `innovate` tetap dilaporkan sebagai\n"
              "    arah lanjutan, bukan pengganti.")
    if "innovate" not in S:
        return

    print("\n" + "=" * 78)
    print("GERBANG 3 — apakah `innovate` layak mengisi slot itu?")
    wins = []
    if S["innovate"]["mean_ic_run"] > S["full"]["mean_ic_run"]:
        wins.append(f"mean|IC|/run ({S['innovate']['mean_ic_run']:.4f} > "
                    f"{S['full']['mean_ic_run']:.4f})")
    if (C.get("innovate") or 0) > (C.get("full") or 0):
        wins.append(f"klaster sinyal ({C.get('innovate')} > {C.get('full')})")
    if S["innovate"]["lib_coverage"] > S["full"]["lib_coverage"]:
        wins.append(f"cakupan pustaka ({S['innovate']['lib_coverage']} > "
                    f"{S['full']['lib_coverage']})")
    andal = S["innovate"]["n_producing"] >= S["full"]["n_producing"] - 1
    for w in wins:
        print(f"  unggul: {w}")
    if not wins:
        print("  unggul: TIDAK ADA")
    print(f"  keandalan: {S['innovate']['n_producing']}/{S['innovate']['n_runs']} run "
          f"menghasilkan (rujukan {S['full']['n_producing']}/{S['full']['n_runs']}) "
          f"→ {'lolos' if andal else 'GAGAL'}")

    if "innovate_fid" in S:
        print("\n  Pemisahan efek (agen vs klem kesetiaan):")
        print(f"    innovate (klem OFF): |IC|/run {S['innovate']['mean_ic_run']:.4f}, "
              f"pustaka {S['innovate']['lib_coverage']}, klaster {C.get('innovate')}")
        print(f"    innovate (klem ON) : |IC|/run {S['innovate_fid']['mean_ic_run']:.4f}, "
              f"pustaka {S['innovate_fid']['lib_coverage']}, klaster {C.get('innovate_fid')}")

    print("\n" + "=" * 78)
    if design_gagal and wins and andal:
        print(f"KEPUTUSAN: GANTI `design` dengan `innovate`. Dasar: {'; '.join(wins)}.")
    elif design_gagal and wins and not andal:
        # Kasus yang benar-benar terjadi: `innovate` MENANG pada sumbu mutu &
        # cakupan tetapi KALAH pada keandalan. Membedakannya dari "tidak menang
        # sama sekali" itu penting — yang satu berarti idenya salah, yang lain
        # berarti idenya benar tapi implementasinya belum stabil, dan keduanya
        # menuntut langkah lanjut yang berbeda.
        print("KEPUTUSAN: BELUM GANTI. `design` gugur di gerbang 1, dan `innovate` "
              "unggul pada:\n  - " + "\n  - ".join(wins) +
              "\ntetapi GAGAL syarat keandalan "
              f"({S['innovate']['n_producing']}/{S['innovate']['n_runs']} vs "
              f"{S['full']['n_producing']}/{S['full']['n_runs']}).\n"
              "Jadi yang terbukti bukan 'ide inovasi salah', melainkan "
              "'implementasinya belum stabil'.\nLangkah yang sah: perbaiki penyebab "
              "kolaps, DAFTARKAN ULANG lengannya, jalankan lagi.\n"
              "Memangkas `design` sekarang (B13) sah secara aturan, tetapi menunda "
              "sampai pengganti\nyang stabil ada akan menghindari dua perubahan "
              "arsitektur berturut-turut.")
    elif design_gagal:
        print("KEPUTUSAN: PANGKAS `design` (B13, jadi proposal→construct). `innovate` "
              "tidak melampaui rujukan pada sumbu mana pun,\nsehingga tak ada alasan "
              "mengisi slot itu — dan gerbang 1/2 sudah menyatakan slotnya tak "
              "membayar biayanya.")
    else:
        print("KEPUTUSAN: PERTAHANKAN `design`.")
    print("\nCatatan wajib: n=6 per lengan. Sumbu |IC| di sini TIDAK punya daya uji; "
          "yang menopang keputusan\nadalah sumbu bervarians rendah (cakupan pustaka, "
          "klaster, laju gate, biaya).")


if __name__ == "__main__":
    main()
