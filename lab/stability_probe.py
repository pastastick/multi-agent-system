"""A11 — stabilitas jangka panjang: VRAM puncak, pertumbuhan KV per hop, kebocoran.

Sumbu ini menopang klaim "sistem ini bisa dijalankan berkelanjutan", yang selama
ini hanya punya catatan anekdotal (B1, T6) dan tak pernah jadi metrik rutin.
Tiga hal diukur dalam SATU proses yang menjalankan beberapa trajectory
berturut-turut — karena kebocoran hanya terlihat lintas-run, bukan di dalam satu
run:

  VRAM puncak per run   `torch.cuda.max_memory_allocated()` di-reset tiap run.
  Pertumbuhan KV/hop    panjang KV setelah tiap agen — apakah linear terhadap
                        jumlah hop, dan berapa kemiringannya.
  Kebocoran lintas-run  VRAM yang MASIH terpakai setelah run selesai + gc +
                        empty_cache. Kalau angka ini naik monoton, ada objek
                        (KV cache, tensor log) yang tak pernah dilepas.

Backtest & skoring CPU sengaja TIDAK dijalankan: yang diuji di sini sumber daya,
bukan mutu faktor, dan skoring hanya akan menambah waktu tanpa menambah
informasi.

    python lab/stability_probe.py --runs 6 --comm-mode kv --latent-steps 10
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics as st
import sys
import time
from pathlib import Path
from types import SimpleNamespace

QL = Path(__file__).resolve().parent.parent
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
for p in (str(QL), str(QL / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lab.frontend_probe import (  # noqa: E402
    DIRECTIONS, OUT, build_backend, run_once,
)

MB = 1024 ** 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--comm-mode", dest="comm_mode", default="kv",
                    choices=["kv", "kv_and_text", "text", "summary"])
    ap.add_argument("--latent-steps", type=int, default=10)
    ap.add_argument("--latent-mode", default="gumbel")
    ap.add_argument("--directions", default="d0,d1")
    ap.add_argument("--tag", default="a11")
    a = ap.parse_args()

    import torch

    cfg = SimpleNamespace(
        model=a.model, comm_mode=a.comm_mode, latent_steps=a.latent_steps,
        latent_mode=a.latent_mode, latent_temp=0.7, no_realign=False,
        temperature=0.8, max_new_tokens=4096, max_repair=3, prompts="",
        tag=a.tag, holdout=False, chain="", free_form=None, early_stop_cos=None,
    )
    import os
    os.environ["LATENT_STEP_MODE"] = a.latent_mode
    os.environ["LATENT_STEP_TEMP"] = str(cfg.latent_temp)

    OUT.mkdir(parents=True, exist_ok=True)
    backend = build_backend(cfg)
    prompts_path = QL / "backend" / "latent_mas" / "prompts.yaml"
    dirs = [d.strip() for d in a.directions.split(",") if d.strip()]

    # Garis dasar SESUDAH bobot dimuat: yang menarik adalah pertumbuhan di ATAS
    # bobot, bukan ukuran bobotnya.
    gc.collect(); torch.cuda.empty_cache()
    base_mb = torch.cuda.memory_allocated() / MB
    print(f"[a11] {a.model} comm={a.comm_mode} ls={a.latent_steps} "
          f"runs={a.runs} | VRAM setelah muat bobot = {base_mb:.0f} MB")

    rows = []
    for i in range(a.runs):
        d = dirs[i % len(dirs)]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        r = run_once(backend, cfg, DIRECTIONS[d], i, prompts_path)
        peak_mb = torch.cuda.max_memory_allocated() / MB
        # Residu = memori yang MASIH dipegang setelah run dibereskan.
        gc.collect(); torch.cuda.empty_cache()
        resid_mb = torch.cuda.memory_allocated() / MB
        hops = [(t["agent"], t["kv_len"]) for t in r.get("agent_trace", [])]
        rows.append({
            "run": i, "direction": d, "dur_s": round(time.time() - t0, 1),
            "peak_mb": round(peak_mb, 1),
            "peak_over_base_mb": round(peak_mb - base_mb, 1),
            "resident_mb": round(resid_mb, 1),
            "leak_vs_base_mb": round(resid_mb - base_mb, 1),
            "kv_per_hop": hops,
            "n_factors": len(r.get("factors") or []),
            "error": bool(r.get("error")),
        })
        print(f"  run {i} dir={d:8s} {rows[-1]['dur_s']:6.1f}s "
              f"puncak={peak_mb:7.0f} MB (+{peak_mb-base_mb:6.0f} di atas bobot) "
              f"residu={resid_mb:7.0f} MB (bocor {resid_mb-base_mb:+6.1f}) "
              f"KV={[h[1] for h in hops]} n_fac={rows[-1]['n_factors']}",
              flush=True)

    leaks = [r["leak_vs_base_mb"] for r in rows]
    peaks = [r["peak_over_base_mb"] for r in rows]
    # Kemiringan kebocoran lintas-run (regresi linear sederhana). Yang penting
    # bukan nilai mutlaknya melainkan apakah ia naik MONOTON — itu tanda objek
    # yang tak pernah dilepas.
    n = len(leaks)
    slope = 0.0
    if n > 1:
        mx = (n - 1) / 2
        my = st.mean(leaks)
        den = sum((i - mx) ** 2 for i in range(n))
        slope = sum((i - mx) * (leaks[i] - my) for i in range(n)) / den if den else 0.0

    print(f"\n  VRAM puncak di atas bobot : {st.mean(peaks):.0f} MB "
          f"[{min(peaks):.0f}–{max(peaks):.0f}]")
    print(f"  residu setelah run        : {st.mean(leaks):+.1f} MB "
          f"[{min(leaks):+.1f}–{max(leaks):+.1f}]")
    print(f"  kemiringan residu/run     : {slope:+.2f} MB/run "
          f"→ {'ADA indikasi kebocoran' if slope > 5 else 'tak ada kebocoran terdeteksi'}")

    # Pertumbuhan KV per hop, dirata-rata lintas run.
    per_agent: dict[str, list[int]] = {}
    for r in rows:
        for agent, kv in r["kv_per_hop"]:
            per_agent.setdefault(agent, []).append(kv)
    print("\n  KV kumulatif per hop (rata-rata):")
    prev = 0
    for agent, vals in per_agent.items():
        m = st.mean(vals)
        print(f"    {agent:<12s} {m:8.0f} tok   (+{m - prev:7.0f} dari hop sebelumnya)")
        prev = m

    doc = {"_meta": {"model": a.model, "comm_mode": a.comm_mode,
                     "latent_steps": a.latent_steps, "runs": a.runs,
                     "base_mb": round(base_mb, 1)},
           "_summary": {"peak_over_base_mb_mean": round(st.mean(peaks), 1),
                        "leak_mb_mean": round(st.mean(leaks), 1),
                        "leak_slope_mb_per_run": round(slope, 2),
                        "kv_per_hop_mean": {k: round(st.mean(v), 1)
                                            for k, v in per_agent.items()}},
           "runs": rows}
    path = OUT / f"stability_{a.tag}.json"
    path.write_text(json.dumps(doc, indent=2, default=str))
    print(f"\ntersimpan → {path}")


if __name__ == "__main__":
    main()
