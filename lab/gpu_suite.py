"""Jalankan seluruh matriks eksperimen GPU dalam SATU proses.

Model 8B ≈ 16 GB dibaca dari volume jaringan; memuatnya ulang tiap invokasi
memakan beberapa menit dan itu biaya terbesar kalau tiap lengan dijalankan
sebagai proses sendiri. `_MODEL_CACHE` di `llm/client.py` di-key oleh
(model_name, device), jadi backend baru per lengan (latent_steps / mode langkah
laten / use_realign berbeda) hanya membangun ulang _CoreEngine — bobotnya
dipakai bersama.

Tiap lengan disimpan SEGERA setelah selesai (`lab/out/frontend_<tag>.json`),
jadi rencana yang terputus di tengah tidak kehilangan lengan yang sudah jadi
dan bisa dilanjutkan dengan --skip-existing.

    python lab/gpu_suite.py --plan g2
    python lab/gpu_suite.py --plan g4 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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
    DIRECTIONS, OUT, build_backend, run_once, score_expressions,
)

BASE = dict(
    model="Qwen/Qwen3-8B", comm_mode="kv", latent_steps=60, latent_mode="raw",
    latent_temp=0.7, no_realign=False, temperature=0.8, max_new_tokens=4096,
    max_repair=3, prompts="", tag="", holdout=False, chain="", free_form=None,
    early_stop_cos=None,     # B6; None = default engine (0,999), 1.0 = mati
)

# Rantai LAMA (sebelum B16), dipatok eksplisit di G2/G3/G4/G6/prompt di bawah.
# Default FrontEndPipeline berubah 2026-08-07 (proposal,design,construct ->
# proposal,innovate,construct — lihat lab/HASIL_A8.md). Rencana G-series ini
# mengukur latent_steps/comm_mode/realign/prompt SEBAGAI VARIABEL; membiarkan
# `chain` ikut default baru akan diam-diam mengganti apa yang dibandingkan bila
# rencana ini dijalankan ulang nanti untuk memperluas seed. Dipatok agar hasil
# lama & baru tetap bisa disandingkan pada sumbu yang sama.
_LEGACY_CHAIN = "proposal,design,construct"


def plan_g2(a) -> list[dict]:
    """G2 — uji cepat latent_steps. Set {5,10,20,40} atas permintaan; 60 ikut
    sebagai REFERENSI konfigurasi produksi saat ini (experiment.yaml)."""
    return [dict(comm_mode="kv", latent_steps=ls, chain=_LEGACY_CHAIN,
                 tag=f"g2_kv_ls{ls}")
            for ls in (5, 10, 20, 40, 60)]


def plan_g3(a) -> list[dict]:
    """G3 — mode langkah laten pada latent_steps yang dipilih (default 10)."""
    ls = a.ls
    return [dict(comm_mode="kv", latent_steps=ls, latent_mode=m, latent_temp=t,
                 chain=_LEGACY_CHAIN,
                 tag=f"g3_kv_ls{ls}_{m}{'' if m == 'raw' else f'T{t}'}")
            for m, t in (("raw", 0.7), ("gumbel", 0.7), ("sample", 1.0))]


def plan_g4(a) -> list[dict]:
    """G4 — replikasi lintas seed untuk tiap comm_mode.

    Lengan `kv_and_text` dijalankan pada DUA nilai latent_steps: a.ls (hasil G2)
    dan 60 (nilai produksi di experiment.yaml). Lengan kv@60 tidak diulang di
    sini karena sudah ada 6 run darinya di G2. `text` tak memakai jalur laten
    sama sekali, jadi latent_steps tidak relevan untuknya."""
    return [dict(comm_mode="text", latent_steps=0, chain=_LEGACY_CHAIN, tag="g4_text"),
            dict(comm_mode="kv_and_text", latent_steps=a.ls, chain=_LEGACY_CHAIN,
                 tag=f"g4_kv_and_text_ls{a.ls}"),
            dict(comm_mode="kv", latent_steps=a.ls, chain=_LEGACY_CHAIN,
                 tag=f"g4_kv_ls{a.ls}"),
            dict(comm_mode="kv_and_text", latent_steps=60, chain=_LEGACY_CHAIN,
                 tag="g4_kv_and_text_ls60")]


def plan_g6(a) -> list[dict]:
    """G6 — ablasi use_realign. Bermakna HANYA pada backbone tidak-tied
    (Qwen3-8B); pada Qwen3-4B kedua cabang identik (lab/realign_probe.py)."""
    return [dict(comm_mode="kv", latent_steps=a.ls, no_realign=b, chain=_LEGACY_CHAIN,
                 tag=f"g6_kv_ls{a.ls}_realign{'OFF' if b else 'ON'}")
            for b in (False, True)]


def plan_prompt(a) -> list[dict]:
    """Pertanyaan tambahan — MODEL vs PROMPT sebagai penyebab ekspresi buruk.
    comm_mode=text supaya jalur laten tidak ikut menjadi variabel."""
    v1 = str(QL / "backend" / "latent_mas" / "prompts_v1.yaml")
    short = a.model.split("/")[-1].replace("Qwen3-", "")
    return [dict(comm_mode="text", latent_steps=0, model=a.model, prompts=pp,
                 chain=_LEGACY_CHAIN, tag=f"px_{short}_{nm}")
            for nm, pp in (("v0", ""), ("v1", v1))]


def plan_a8(a) -> list[dict]:
    """A8 — ABLASI AGEN: apakah rantai 3-agen layak dipertahankan, dan apakah
    `design` sebaiknya diganti agen inovasi.

    Lima lengan, satu variabel berubah antar-lengan yang berpasangan:
      full          proposal->design->construct        rantai produksi
      nodesign      proposal->construct                design DIPOTONG
      direct        construct sendirian                arah langsung ke builder
      innovate      proposal->innovate->construct      design DIGANTI (+ klem
                                                       kesetiaan dilepas)
      innovate_fid  proposal->innovate->construct      design DIGANTI, klem
                                                       kesetiaan TETAP
    Pasangan (innovate, innovate_fid) memisahkan efek "ganti agen" dari efek
    "lepas klem kesetiaan" — tanpa itu keduanya berubah bersamaan dan tak ada
    klaim kausal yang bisa dipertahankan.

    Dijalankan pada konfigurasi Tahap 1 (ls dari --ls, gumbel), karena itulah
    keadaan sistem yang sekarang berlaku.
    """
    ls = a.ls
    return [
        dict(comm_mode=a.comm_mode, latent_steps=ls, latent_mode="gumbel",
             chain="proposal,design,construct", tag=f"a8_{a.comm_mode}_full"),
        dict(comm_mode=a.comm_mode, latent_steps=ls, latent_mode="gumbel",
             chain="proposal,construct", tag=f"a8_{a.comm_mode}_nodesign"),
        dict(comm_mode=a.comm_mode, latent_steps=ls, latent_mode="gumbel",
             chain="construct", tag=f"a8_{a.comm_mode}_direct"),
        dict(comm_mode=a.comm_mode, latent_steps=ls, latent_mode="gumbel",
             chain="proposal,innovate,construct", free_form=True,
             tag=f"a8_{a.comm_mode}_innovate"),
        dict(comm_mode=a.comm_mode, latent_steps=ls, latent_mode="gumbel",
             chain="proposal,innovate,construct", free_form=False,
             tag=f"a8_{a.comm_mode}_innovate_fid"),
    ]


PLANS = {"g2": plan_g2, "g3": plan_g3, "g4": plan_g4, "g6": plan_g6,
         "prompt": plan_prompt, "a8": plan_a8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=sorted(PLANS))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--directions", default="d0,d1")
    ap.add_argument("--ls", type=int, default=10, help="latent_steps untuk g3/g4/g6/a8")
    ap.add_argument("--comm-mode", dest="comm_mode", default="kv",
                    choices=["kv", "kv_and_text", "text", "summary"],
                    help="medium untuk plan a8")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    dirs = [d.strip() for d in a.directions.split(",") if d.strip()]
    arms = PLANS[a.plan](a)

    print(f"[suite] rencana={a.plan}  lengan={len(arms)}  "
          f"run/lengan={len(dirs)*len(seeds)}  total={len(arms)*len(dirs)*len(seeds)}")

    for i, over in enumerate(arms, 1):
        cfg = SimpleNamespace(**{**BASE, "model": a.model, **over})
        path = OUT / f"frontend_{cfg.tag}.json"
        if a.skip_existing and path.exists():
            print(f"\n[{i}/{len(arms)}] {cfg.tag}: sudah ada, dilewati")
            continue

        # dibaca _CoreEngine saat konstruksi → set SEBELUM build_backend
        os.environ["LATENT_STEP_MODE"] = cfg.latent_mode
        os.environ["LATENT_STEP_TEMP"] = str(cfg.latent_temp)

        t_arm = time.time()
        print(f"\n[{i}/{len(arms)}] === {cfg.tag} === comm={cfg.comm_mode} "
              f"ls={cfg.latent_steps} step_mode={cfg.latent_mode} "
              f"realign={not cfg.no_realign} prompts={Path(cfg.prompts).name or 'v0'}",
              flush=True)
        backend = build_backend(cfg)
        prompts_path = (Path(cfg.prompts) if cfg.prompts
                        else QL / "backend" / "latent_mas" / "prompts.yaml")

        runs = []
        for d in dirs:
            for s in seeds:
                r = run_once(backend, cfg, DIRECTIONS[d], s, prompts_path)
                r.update({"direction": d, "seed": s, "comm_mode": cfg.comm_mode,
                          "latent_steps": cfg.latent_steps,
                          "latent_mode": cfg.latent_mode,
                          "model": cfg.model, "use_realign": not cfg.no_realign,
                          "prompts": str(prompts_path), "tag": cfg.tag,
                          "arm": cfg.tag.split("_", 2)[-1]})
                runs.append(r)
                print(f"    dir={d} seed={s}: {r['duration_s']:>6.1f}s "
                      f"n_factors={len(r.get('factors') or [])} "
                      f"n_pass={len(r.get('passing') or [])} "
                      f"repaired={r.get('repaired')} err={bool(r.get('error'))}",
                      flush=True)
                path.write_text(json.dumps({"args": vars(cfg), "runs": runs},
                                           indent=2, default=str))

        del backend
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        print(f"    lengan selesai dalam {time.time()-t_arm:.0f}s → {path.name}",
              flush=True)

    print("\n[suite] skoring semua lengan di CPU ...", flush=True)
    for over in arms:
        cfg = SimpleNamespace(**{**BASE, "model": a.model, **over})
        path = OUT / f"frontend_{cfg.tag}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        score_expressions(doc["runs"],
                          series_path=OUT / f"icseries_{cfg.tag}.parquet")
        path.write_text(json.dumps(doc, indent=2, default=str))
        facs = [f for r in doc["runs"] for f in (r.get("factors") or [])]
        ok = [f for f in facs if f.get("ic") is not None]
        print(f"  {cfg.tag:<28s} expr={len(facs):>3d} ber-IC={len(ok):>3d} "
              f"cacat={sum(1 for f in facs if f.get('sem_ok') is False):>3d} "
              f"lolos-gate={sum(1 for f in facs if f.get('passed_gate')):>3d}")


if __name__ == "__main__":
    main()
