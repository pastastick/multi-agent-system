"""G5 — laju tolak gate (khususnya gate SEMANTIK baru) pada run nyata.

AUDIT_KRITIS §8/G5 menetapkan ambang keputusan: bila laju tolak > ~60%, prompt
DSL harus diperbaiki LEBIH DULU (§S6) sebelum gate diperketat, karena kalau
tidak putaran repair akan membengkak. Skrip ini menghitung laju itu, memecahnya
per SEBAB, dan melaporkan berapa sering repair terpanggil serta berhasil.

    python lab/gate_report.py --glob 'frontend_g*.json'
    python lab/gate_report.py --glob 'frontend_px_*.json' --by model _prompts
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"

# sebab gate dinormalkan ke kelas; urutan pengecekan mengikuti gate produksi
# (latent_mas/pipeline.py::_build_regulator_gate).
CLASSES = [
    ("kosong", r"^empty expression"),
    ("tak-parsable", r"^unparsable"),
    ("arity", r"^arity:"),
    ("variabel", r"^variable:"),
    ("degenerate-arg", r"^degenerate:"),
    ("SEMANTIK", r"^semantics:"),
    ("regulator-SL/ER/dup", r"^regulator"),
]
# sub-kelas cacat semantik (pesan dari validate_semantics)
SEM_SUB = [
    ("window degenerate", r"got window .* degenerate"),
    ("kondisi non-boolean", r"is a continuous score, not a"),
    ("ambang pada persentil", r"returns a percentile between 0 and 1"),
    ("ambang absolut \\$volume", r"absolute number"),
]


def classify(reason: str) -> str:
    for name, pat in CLASSES:
        if re.search(pat, reason or ""):
            return name
    return "lain"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=["frontend_*.json"])
    ap.add_argument("--by", nargs="+", default=["tag"])
    a = ap.parse_args()

    runs = []
    for pat in a.glob:
        for p in sorted(OUT.glob(pat)):
            doc = json.loads(p.read_text())
            for r in doc["runs"]:
                r["_prompts"] = Path(r.get("prompts", "")).name
                r.setdefault("tag", doc["args"].get("tag"))
                runs.append(r)
    if not runs:
        sys.exit(f"tak ada file cocok: {a.glob}")

    arms: dict[str, list[dict]] = {}
    for r in runs:
        arms.setdefault(" | ".join(f"{k}={r.get(k)}" for k in a.by), []).append(r)

    print(f"\n{len(runs)} run, {len(arms)} lengan\n")
    hdr = (f"{'lengan':<40s} {'kand':>5s} {'tolak':>6s} {'SEM':>5s} {'arity':>6s} "
           f"{'var':>4s} {'reg':>4s} {'parse':>6s} {'repair':>8s} {'sukses':>7s}")
    print(hdr)
    print("-" * len(hdr))

    for name, rs in sorted(arms.items()):
        gl = [g for r in rs for g in (r.get("gate_log") or [])
              if g.get("repaired_by") is None]          # kandidat asli construct
        cnt = Counter(classify(g["reason"]) for g in gl if not g["ok"])
        n = len(gl)
        rej = sum(1 for g in gl if not g["ok"])
        n_rep = sum(1 for r in rs if (r.get("repair_attempts") or 0) > 0)
        n_rep_ok = sum(1 for r in rs if r.get("repaired"))
        print(f"{name:<40s} {n:>5d} {rej/max(n,1):>5.0%} "
              f"{cnt['SEMANTIK']:>5d} {cnt['arity']:>6d} {cnt['variabel']:>4d} "
              f"{cnt['regulator-SL/ER/dup']:>4d} {cnt['tak-parsable']:>6d} "
              f"{n_rep:>4d}/{len(rs):<3d} {n_rep_ok:>3d}/{max(n_rep,1):<3d}")

    # rincian sebab, seluruh korpus
    gl = [g for r in runs for g in (r.get("gate_log") or []) if g.get("repaired_by") is None]
    rej = [g for g in gl if not g["ok"]]
    print(f"\nSeluruh korpus: {len(gl)} kandidat, {len(rej)} ditolak "
          f"({len(rej)/max(len(gl),1):.0%})")
    for name, _ in CLASSES + [("lain", "")]:
        c = sum(1 for g in rej if classify(g["reason"]) == name)
        if c:
            print(f"  {name:<24s} {c:>4d}  ({c/max(len(rej),1):>4.0%} dari tolakan)")

    sem = [g for g in rej if classify(g["reason"]) == "SEMANTIK"]
    if sem:
        print(f"\n  rincian {len(sem)} tolakan SEMANTIK:")
        for name, pat in SEM_SUB:
            c = sum(1 for g in sem if re.search(pat, g["reason"]))
            if c:
                print(f"    {name:<26s} {c:>4d}")

    # cacat yang LOLOS gate (gate semantik meleset) — dihitung dari skor CPU
    facs = [f for r in runs for f in (r.get("factors") or [])]
    passed = [f for f in facs if f.get("passed_gate")]
    leak_sem = [f for f in passed if f.get("sem_ok") is False]
    leak_dead = [f for f in passed if f.get("ic") is not None
                 and (f.get("n_unique") or 0) <= 2]
    leak_err = [f for f in passed if f.get("eval_error")]
    print(f"\nLolos gate: {len(passed)} ekspresi — dari situ "
          f"{len(leak_sem)} masih cacat-semantik, {len(leak_dead)} praktis mati "
          f"(<=2 nilai unik/hari), {len(leak_err)} gagal dieksekusi.")
    for f in leak_dead[:8]:
        print(f"    MATI: {f['expression'][:90]}")
    for f in leak_err[:5]:
        print(f"    ERROR: {str(f.get('eval_error'))[:70]} | {f['expression'][:70]}")


if __name__ == "__main__":
    main()
