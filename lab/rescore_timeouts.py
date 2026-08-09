"""Pulihkan faktor yang hilang dari statistik |IC| karena ANGGARAN WAKTU skoring.

Masalah yang ditangani (KESIMPULAN §11.2). Skoring CPU memberi 90 detik per
ekspresi supaya satu operator lambat tak menyandera sweep. Ekspresi yang
melewatinya tercatat tanpa IC dan karena itu keluar dari statistik mutu. Ketika
dihitung per lengan, kerugian itu ternyata **tidak acak**: lengan ber-rantai
`innovate` kehilangan 26–30% ekspresinya, lengan ber-rantai `design` 0–6%.
Sebabnya justru alasan `innovate` diadopsi — ia menjangkau `REGRESI`/`REGBETA`
(joblib per-instrumen) dan statistik momen bergulir (`TS_MAD`/`TS_KURT`/
`TS_SKEW`/`TS_MEDIAN`), yang keduanya mahal. Akibatnya perbandingan headline
`innovate_guided` vs `full` berdiri di atas dua himpunan yang tak setara.

Alat ini menutup selisih itu: cari semua ekspresi yang kena anggaran, skor ulang
dengan anggaran besar dan worker joblib lebih banyak, lalu laporkan |IC| per
lengan SEBELUM dan SESUDAH pemulihan.

Sengaja dipisah dari `rescore_all.py`: yang itu harus memakai anggaran yang SAMA
dengan mesin GPU supaya verifikasi silangnya sah. Yang ini justru melanggar
anggaran itu dengan sengaja, dan hasilnya dilaporkan sebagai koreksi terpisah.

    PYTHONPATH=backend LAB_MAX_WORKERS=8 .venv/bin/python lab/rescore_timeouts.py
    ... --budget 1800 --max 10        # sebagian dulu (aman diulang)
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
RECOVERED = OUT / "timeout_recovered.json"


def is_timeout(f: dict) -> bool:
    return "Timeout" in (f.get("eval_error") or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1800,
                    help="anggaran detik per ekspresi (default 1800 = 30 menit)")
    ap.add_argument("--max", type=int, default=0, help="berhenti setelah N ekspresi (0 = semua)")
    ap.add_argument("--apply", action="store_true",
                    help="tulis balik hasil ke frontend_*.json (default: laporan saja)")
    args = ap.parse_args()

    import pandas as pd
    from lab.core import Lab
    from lab.frontend_probe import _time_budget

    docs = {}
    todo: dict[str, list[str]] = {}          # expr → daftar tag
    for path in sorted(OUT.glob("frontend_*.json")):
        try:
            docs[path] = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        tag = path.stem[len("frontend_"):]
        for r in docs[path]["runs"]:
            for f in r.get("factors", []) or []:
                if f.get("expression") and is_timeout(f):
                    todo.setdefault(f["expression"], []).append(tag)

    print(f"[timeout] {len(todo)} ekspresi unik kena anggaran waktu, "
          f"tersebar di {len({t for v in todo.values() for t in v})} tag")

    prev = json.loads(RECOVERED.read_text()) if RECOVERED.exists() else {}
    pending = [e for e in todo if e not in prev]
    if args.max:
        pending = pending[:args.max]
    print(f"[timeout] sudah dipulihkan sebelumnya: {len(prev)} · "
          f"dikerjakan sekarang: {len(pending)} · anggaran {args.budget}s\n", flush=True)

    lab = Lab(mode="fast")
    series: dict[str, pd.Series] = {}
    for i, e in enumerate(pending, 1):
        t0 = time.time()
        try:
            with _time_budget(args.budget):
                res, ser = lab.ic_full(e)
        except TimeoutError:
            res, ser = None, None
        dt = time.time() - t0
        if res is None or res.ic is None:
            prev[e] = {"ic": None, "detik": round(dt, 1),
                       "error": "timeout" if res is None else res.error}
            print(f"  [{i}/{len(pending)}] {dt:6.0f}s  GAGAL LAGI  {e[:78]}", flush=True)
        else:
            prev[e] = {"ic": res.ic, "tstat": res.tstat, "n_unique": res.n_unique,
                       "n_days": res.n_days, "detik": round(dt, 1), "error": None}
            if ser is not None:
                series[e] = ser
            print(f"  [{i}/{len(pending)}] {dt:6.0f}s  |IC|={abs(res.ic):.4f}  "
                  f"{e[:70]}", flush=True)
        RECOVERED.write_text(json.dumps(prev, indent=2, default=str))

    # ── laporan: |IC| per lengan sebelum vs sesudah pemulihan ─────────────
    print("\n=== |IC| per-ekspresi hidup, SEBELUM vs SESUDAH pemulihan ===")
    print(f"{'lengan':30s} {'n0':>4s} {'mean|IC|0':>10s} {'n1':>4s} {'mean|IC|1':>10s} {'pulih':>6s}")
    rows = []
    for path, doc in docs.items():
        tag = path.stem[len("frontend_"):]
        base, extra = [], []
        for r in doc["runs"]:
            for f in r.get("factors", []) or []:
                e = f.get("expression", "")
                if not e:
                    continue
                if f.get("ic") is not None and (f.get("n_unique") or 0) > 2:
                    base.append(abs(float(f["ic"])))
                elif is_timeout(f) and prev.get(e, {}).get("ic") is not None:
                    rec = prev[e]
                    if (rec.get("n_unique") or 0) > 2:
                        extra.append(abs(float(rec["ic"])))
        if not base:
            continue
        m0 = sum(base) / len(base)
        allv = base + extra
        m1 = sum(allv) / len(allv)
        rows.append({"tag": tag, "n0": len(base), "mean_abs_ic0": m0,
                     "n1": len(allv), "mean_abs_ic1": m1, "pulih": len(extra)})
        if len(base) >= 8:
            print(f"{tag:30s} {len(base):4d} {m0:10.5f} {len(allv):4d} {m1:10.5f} "
                  f"{len(extra):6d}")

    if args.apply:
        for path, doc in docs.items():
            changed = False
            for r in doc["runs"]:
                for f in r.get("factors", []) or []:
                    rec = prev.get(f.get("expression", ""))
                    if rec and is_timeout(f) and rec.get("ic") is not None:
                        f.update({k: rec[k] for k in
                                  ("ic", "tstat", "n_unique", "n_days") if k in rec})
                        f["eval_error"] = None
                        f["recovered_from_timeout"] = True
                        changed = True
            if changed:
                path.write_text(json.dumps(doc, indent=2, default=str))
        print("\n[timeout] hasil ditulis balik ke frontend_*.json "
              "(ditandai `recovered_from_timeout`)")
        if series:
            for path in OUT.glob("icseries_*.parquet"):
                tag = path.stem[len("icseries_"):]
                doc = docs.get(OUT / f"frontend_{tag}.json")
                if doc is None:
                    continue
                want = {e for r in doc["runs"] for f in (r.get("factors") or [])
                        for e in [f.get("expression", "")] if e in series}
                if not want:
                    continue
                df = pd.read_parquet(path)
                for e in want:
                    df[e] = series[e]
                df.to_parquet(path)
            print("[timeout] deret IC ditambahkan ke icseries_*.parquet terkait")

    (OUT / "timeout_report.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nlaporan → {OUT / 'timeout_report.json'} · rincian → {RECOVERED}")


if __name__ == "__main__":
    main()
