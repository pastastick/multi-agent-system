#!/usr/bin/env python3
"""
experiments/run_mining.py
====================
Jalankan loop mining LatentMAS BARU end-to-end untuk SATU direction —
front-end laten → backtest substrat → feedback. Untuk GPU/Qlib-test loop baru
SEBELUM menyentuh pipeline/loop.py atau menghapus kode lama.

Contoh
------
  python experiments/run_mining.py \
      --direction "overnight gap reversal on high-volume stocks" \
      --iterations 3

  # tanpa backtest (cepat, hanya cek front-end + bridge build):
  python experiments/run_mining.py --direction "..." --iterations 1 --no-backtest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run new LatentMAS mining loop end-to-end")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-backtest", action="store_true",
                    help="lewati runner.develop (hanya front-end + bridge build)")
    ap.add_argument("--library", default=None, help="path factor library JSON (opsional)")
    ap.add_argument("--console", default="INFO")
    # ── Diagnostik latent (untuk bisect collapse) ────────────────────────────
    # Default 0/off = sama dengan run_mining yang BERHASIL. Naikkan untuk
    # mereproduksi collapse dari factor_mining (yang pakai steps=10, realign, knn).
    ap.add_argument("--latent-steps", type=int, default=20,
                    help="virtual-token reasoning per call (factor_mining pakai 10)")
    ap.add_argument("--use-realign", action="store_true",
                    help="aktifkan realigner (factor_mining: on)")
    ap.add_argument("--knn", action="store_true",
                    help="aktifkan KNN KV-filter (factor_mining: on)")
    args = ap.parse_args()

    from llm.client import LocalLLMBackend
    from latent_mas.runlog import get_run_logger
    from latent_mas.mining_loop import MiningLoop
    from pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING as SETTING

    rl = get_run_logger(run_name="mining", console_level=args.console)
    rl.info("new mining loop", direction=args.direction, iterations=args.iterations,
            latent_steps=args.latent_steps, use_realign=args.use_realign, knn=args.knn)

    backend = LocalLLMBackend(
        model_name=args.model, device=args.device,
        latent_steps=args.latent_steps, use_realign=args.use_realign,
        knn_enabled=args.knn,
    )

    loop = MiningLoop.from_settings(backend, SETTING, runlog=rl, use_local=True)
    loop.library_path = args.library

    if args.no_backtest:
        # ganti runner.develop dengan no-op agar bisa cek front-end + bridge
        loop.runner.develop = lambda exp, use_local=True: exp  # type: ignore
        rl.warn("backtest disabled (--no-backtest)")

    results = loop.run(direction=args.direction, n_iterations=args.iterations)

    print("\n===== MINING RESULTS =====")
    for i, r in enumerate(results):
        print(f"\n--- iteration {i+1} ---")
        print(f"hypothesis : {r.hypothesis}")
        print(f"expression : {r.expression}")
        print(f"repaired   : {r.repaired}   backtest_ok: {r.backtest_ok}   replace_sota: {r.replace_sota}")
        if r.feedback:
            print("feedback   :", json.dumps(r.feedback, indent=2, default=str)[:800])

    rl.finalize()


if __name__ == "__main__":
    main()
