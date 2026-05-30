#!/usr/bin/env python3
"""
experiments/inspect_kv.py
====================
Probe isi sebuah KV-cache tersimpan dengan agent `introspect`.

Idenya (dari diskusi): ambil KV (mis. KV judger atau consistency), suruh model
menuliskan kembali pertanyaan/tugas + daftar fungsi yang "terlihat" di memorinya.
Ini cara memperkirakan APA yang sebenarnya tersimpan di KV — tidak terbatas
pada output teks normal.

Contoh
------
  python experiments/inspect_kv.py runs/kv_consist.pt
  python experiments/inspect_kv.py runs/kv_judger.pt --describe-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspeksi isi KV-cache via probe agent")
    ap.add_argument("kv_path", help="path .pt KV-cache")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--describe-only", action="store_true",
                    help="hanya cetak metadata KV (tanpa load model)")
    ap.add_argument("--console", default="INFO")
    args = ap.parse_args()

    from latent_mas import kv_ops

    if args.describe_only:
        kv = kv_ops.kv_load(args.kv_path, device=None)
        print(json.dumps(kv_ops.kv_describe(kv), indent=2))
        return

    from llm.client import LocalLLMBackend
    from latent_mas.runlog import get_run_logger
    from latent_mas.agent import load_agent

    rl = get_run_logger(run_name="inspect_kv", console_level=args.console)
    kv = kv_ops.kv_load(args.kv_path, device=args.device)
    rl.info("loaded KV", **kv_ops.kv_describe(kv))

    backend = LocalLLMBackend(model_name=args.model, device=args.device)
    probe = load_agent("introspect", backend, strict_vars=False, runlog=rl)

    # clone supaya KV asli tidak ter-mutasi oleh probe
    res = probe.run(past_kv=kv_ops.kv_deepcopy(kv))

    print("\n===== KV DESCRIBE =====")
    print(json.dumps(kv_ops.kv_describe(kv), indent=2))
    print("\n===== PROBE RECONSTRUCTION =====")
    print(res.text)
    rl.finalize()


if __name__ == "__main__":
    main()
