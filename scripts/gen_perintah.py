#!/usr/bin/env python3
"""Turunkan daftar perintah run dari `configs/matriks.yaml`.

Kenapa generator, bukan runner yang membaca YAML langsung: satu proses per sel
adalah keputusan sadar (2–3 sel muat paralel di satu A40 — docs/HASIL_TAHAP0.md
§8.7), dan sebuah runner yang menyapu matriks di dalam satu proses justru
membuang setengah kartu. Generator memberi kontrol penjadwalan ke shell tanpa
kehilangan satu sumber kebenaran untuk isi matriksnya.

Ia juga menegakkan satu aturan yang gampang terlewat kalau perintah ditulis
tangan: pada `comm_mode=text` TIDAK ADA langkah laten, jadi keempat nilai
Sumbu A menghasilkan sel yang identik. Menjalankannya empat kali membakar GPU
untuk empat salinan angka yang sama — dan lebih buruk, empat salinan itu akan
terlihat seperti empat pengamatan independen di tabel. Di sini `text`
dijalankan SEKALI.

    python scripts/gen_perintah.py --arm bench             # perintah lengan benchmark
    python scripts/gen_perintah.py --arm factor            # perintah lengan faktor
    python scripts/gen_perintah.py --arm bench --parallel 3 > jalankan.sh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from paths import CONFIGS  # noqa: E402

import yaml  # noqa: E402

PY = "PYTHONPATH=backend python"


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf8"))


def bench_commands(cfg: dict) -> list[str]:
    b = cfg["bench"]
    model, ls = cfg["model"], cfg["latent_steps"]
    base = (f"{PY} backend/bench/run_bench.py --model {model} "
            f"--latent-steps {ls} --latent-temp {cfg['latent_temp']} "
            f"--limit {b['limit']} --sample-seed {b['sample_seed']} "
            f"--temperature {b['temperature']} --top-p {b['top_p']} "
            f"--max-new-tokens {b['max_new_tokens']}")
    cmds = []
    for task in b["tasks"].values():
        for seed in b["seeds"]:
            tag = f"s{seed}"
            # Lantai: agen tunggal. Tak ada rantai → tak ada handoff → satu sel.
            if cfg.get("baseline"):
                cmds.append(f"{base} --task {task} --seed {seed} --baseline "
                            f"--latent-mode raw --comm-mode kv --tag {tag}")
            # Baseline teks: SATU sel, bukan empat (lihat docstring).
            cmds.append(f"{base} --task {task} --seed {seed} "
                        f"--comm-mode {cfg['comm_mode_tanpa_laten']} "
                        f"--latent-mode raw --tag {tag}")
            # Matriks penuh Sumbu A × Sumbu B untuk medium ber-KV.
            modes = list(cfg["latent_modes"])
            if cfg.get("kontrol_latent_mode"):
                modes.append(cfg["kontrol_latent_mode"])
            for comm in cfg["comm_modes"]:
                for mode in modes:
                    extra = (f" --latent-beta {cfg['latent_beta']}"
                             if mode == "moi" else "")
                    cmds.append(f"{base} --task {task} --seed {seed} "
                                f"--comm-mode {comm} --latent-mode {mode}"
                                f"{extra} --tag {tag}")
    return cmds


def factor_commands(cfg: dict) -> list[str]:
    f = cfg["factor"]
    seeds = ",".join(str(s) for s in f["seeds"])
    dirs = ",".join(f["directions"])
    base = (f"{PY} backend/factor/run_factor.py --model {cfg['model']} "
            f"--latent-steps {cfg['latent_steps']} "
            f"--latent-temp {cfg['latent_temp']} "
            f"--seeds {seeds} --directions {dirs} --chain {f['chain']} "
            f"--max-repair {f['max_repair']}")
    # `guided_decoding` di matriks.yaml bersifat DOKUMENTASI: run_factor.py
    # tidak mengeksposnya sebagai flag — nilainya ditentukan `json_schema:` di
    # backend/prompts/factor.yaml (B11/B16). Dicatat di sini supaya setelan yang
    # berlaku saat run tercatat bersama sel-selnya, bukan supaya diteruskan.
    cmds = [f"{base} --comm-mode {cfg['comm_mode_tanpa_laten']} "
            f"--latent-mode raw --tag text"]
    modes = list(cfg["latent_modes"])
    if cfg.get("kontrol_latent_mode"):
        modes.append(cfg["kontrol_latent_mode"])
    for comm in cfg["comm_modes"]:
        for mode in modes:
            cmds.append(f"{base} --comm-mode {comm} --latent-mode {mode} "
                        f"--tag {comm}_{mode}")
    return cmds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=["bench", "factor", "all"])
    ap.add_argument("--config", default=str(CONFIGS / "matriks.yaml"))
    ap.add_argument("--parallel", type=int, default=0,
                    help="bila >0, keluarkan skrip shell yang menjalankan N sel "
                         "bersamaan dengan jeda 30 detik antar-start")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    cmds: list[str] = []
    if args.arm in ("bench", "all"):
        cmds += bench_commands(cfg)
    if args.arm in ("factor", "all"):
        cmds += factor_commands(cfg)

    if args.parallel <= 0:
        print(f"# {len(cmds)} sel dari {args.config}")
        for c in cmds:
            print(c)
        return

    print("#!/usr/bin/env bash")
    print("set -u")
    print(f"# {len(cmds)} sel, {args.parallel} proses paralel.")
    print("# Jeda 30 dtk antar-start: fase muat model dari network storage")
    print("# rebutan I/O kalau serentak (docs/HASIL_TAHAP0.md §8.7).")
    for i in range(0, len(cmds), args.parallel):
        batch = cmds[i:i + args.parallel]
        print(f"\n# ── batch {i // args.parallel + 1} ──")
        for j, c in enumerate(batch):
            print(f"{c} &")
            if j < len(batch) - 1:
                print("sleep 30")
        print("wait")


if __name__ == "__main__":
    main()
