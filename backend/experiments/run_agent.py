#!/usr/bin/env python3
"""
experiments/run_agent.py
====================
Jalankan SATU agent LatentMAS secara standalone — tanpa seluruh pipeline.

Tujuan: iterasi prompt cepat + inspeksi output/KV per agent.

Contoh
------
  # Jalankan proposal saja, simpan KV-nya ke file:
  python experiments/run_agent.py proposal \
      --var direction="momentum reversal on high-volume days" \
      --save-kv runs/kv_proposal.pt

  # Lanjutkan: jalankan construct dari KV proposal:
  python experiments/run_agent.py construct --load-kv runs/kv_proposal.pt \
      --save-kv runs/kv_construct.pt

  # Jalankan judger dari KV consistency, lihat teks + hasil parse:
  python experiments/run_agent.py judger --load-kv runs/kv_consist.pt

  # Variabel bisa dari JSON file:
  python experiments/run_agent.py feedback --vars-json fixtures/fb.json

Catatan: butuh GPU + model (LocalLLMBackend). Output teks juga otomatis
tersimpan oleh backend ke debug/llm_outputs/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# pastikan backend/ ada di path saat dijalankan dari mana saja
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_vars(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--var harus key=value, dapat: {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Jalankan satu agent LatentMAS standalone")
    ap.add_argument("agent", help="nama agent (lihat latent_mas/prompts.yaml)")
    ap.add_argument("--var", action="append", default=[],
                    help="variabel prompt: key=value (boleh diulang)")
    ap.add_argument("--vars-json", help="path JSON berisi variabel prompt")
    ap.add_argument("--load-kv", help="path .pt KV-cache sebagai past_kv")
    ap.add_argument("--save-kv", help="path .pt untuk menyimpan KV output")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--latent-steps", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--show-prompt", action="store_true",
                    help="cetak system+user prompt yang dirender lalu lanjut")
    ap.add_argument("--console", default="INFO", help="level log console")
    args = ap.parse_args()

    # ── kumpulkan variabel prompt ────────────────────────────────────────────
    render_vars = {}
    if args.vars_json:
        render_vars.update(json.loads(Path(args.vars_json).read_text()))
    render_vars.update(_parse_vars(args.var))

    # ── import berat ditunda sampai argumen valid ───────────────────────────
    from llm.client import LocalLLMBackend
    from latent_mas.runlog import get_run_logger
    from latent_mas.agent import load_agent
    from latent_mas import kv_ops

    rl = get_run_logger(run_name=f"agent_{args.agent}", console_level=args.console)
    rl.info("standalone agent run", agent=args.agent, model=args.model)

    backend = LocalLLMBackend(model_name=args.model, device=args.device)
    agent = load_agent(args.agent, backend, strict_vars=False, runlog=rl)
    if args.latent_steps is not None:
        agent.spec.latent_steps = args.latent_steps
    if args.temperature is not None:
        agent.spec.temperature = args.temperature

    if args.show_prompt:
        system, user = agent.render(**render_vars)
        print("\n===== SYSTEM =====\n" + system)
        print("\n===== USER =====\n" + user + "\n")

    past_kv = None
    if args.load_kv:
        past_kv = kv_ops.kv_load(args.load_kv, device=args.device)
        rl.info("loaded past_kv", **kv_ops.kv_describe(past_kv))

    res = agent.run(past_kv=past_kv, **render_vars)

    print("\n===== OUTPUT TEXT =====")
    print(res.text if res.text is not None else "(no text — kv_only mode)")
    print("\n===== RESULT =====")
    print(json.dumps(res.describe(), indent=2, default=str))

    if args.save_kv and res.kv_cache is not None:
        path = kv_ops.kv_save(res.kv_cache, args.save_kv,
                              metadata={"agent": args.agent})
        rl.info("saved KV", path=str(path))
        print(f"\nKV saved → {path}")

    rl.finalize()


if __name__ == "__main__":
    main()
