#!/usr/bin/env python3
"""
experiments/run_evolution.py
====================
Uji kemampuan 3 agent evolution SECARA TERISOLASI — tanpa factor_mining/backtest.

Alur:
  1. Jalankan FrontEndPipeline untuk membuat 1-2 "parent" (hypothesis+expression
     + kv_judger asli, persis konteks yang dipakai loop nyata).
  2. Mutation : EvolutionOps.mutate pada parent-1 (reflection → judger).
  3. Crossover: EvolutionOps.crossover([parent-1, parent-2]) (kv_concat → judger).
  4. Cetak diagnosis + hypothesis/expression hasil, plus KV describe.

Feedback parent & ringkasan backtest disuntik manual (--feedback / --backtest-
summary) supaya kamu bisa stress-test reflection: ubah feedback → lihat apakah
mutation_reflection mendiagnosa step yang tepat.

Contoh
------
  python experiments/run_evolution.py \
    --direction  "price-volume momentum forecast" \
    --direction2 "overnight gap mean-reversion" \
    --latent-steps 10 --use-realign --knn \
    --feedback "Low IC (0.01). Expression too volume-heavy; mechanism weak under noise." \
    --backtest-summary "IC=0.010 RankIC=0.024 annualized=3.5% MDD=-9.5%"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _show(title: str, obj) -> None:
    print(f"\n{'═'*70}\n{title}\n{'═'*70}")
    print(obj)


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolated test of evolution agents")
    ap.add_argument("--direction", required=True, help="parent-1 direction")
    ap.add_argument("--direction2", default=None,
                    help="parent-2 direction (untuk crossover; default = direction)")
    ap.add_argument("--mode", choices=["mutation", "crossover", "both"], default="both")
    ap.add_argument("--feedback", default="Low IC. The factor mechanism is weak and "
                    "too correlated with raw volume; needs a different signal family.",
                    help="synthetic parent feedback (untuk mutation_reflection)")
    ap.add_argument("--backtest-summary",
                    default="IC=0.010 RankIC=0.024 annualized_return=3.5% max_drawdown=-9.5%",
                    help="synthetic parent backtest metrics")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--latent-steps", type=int, default=0)
    ap.add_argument("--use-realign", action="store_true")
    ap.add_argument("--knn", action="store_true")
    ap.add_argument("--save-parents", default=None,
                    help="dir untuk simpan kv_judger tiap parent (.pt)")
    ap.add_argument("--console", default="INFO")
    args = ap.parse_args()

    from llm.client import LocalLLMBackend
    from latent_mas.runlog import get_run_logger
    from latent_mas.pipeline import FrontEndPipeline, EvolutionOps
    from latent_mas import kv_ops

    rl = get_run_logger(run_name="evolution", console_level=args.console)
    rl.info("evolution isolation test", mode=args.mode,
            latent_steps=args.latent_steps, use_realign=args.use_realign, knn=args.knn)

    backend = LocalLLMBackend(
        model_name=args.model, device=args.device,
        latent_steps=args.latent_steps, use_realign=args.use_realign,
        knn_enabled=args.knn,
    )
    front = FrontEndPipeline(backend, runlog=rl)
    # EvolutionOps berbagi agents yang sama (seperti di loop).
    evo = EvolutionOps(backend, runlog=rl, agents=front.agents)

    # ── 1. buat parent-1 (dan parent-2 untuk crossover) ──────────────────────
    rl.info("building parent-1 via front-end")
    p1 = front.run(direction=args.direction)
    _show("PARENT-1 (front-end)", json.dumps({
        "hypothesis": p1.hypothesis, "expression": p1.expression,
        "repaired": p1.repaired, "gate_error": p1.gate_error,
        "kv_judger": kv_ops.kv_describe(p1.kv_judger),
    }, indent=2, default=str))

    need_p2 = args.mode in ("crossover", "both")
    p2 = None
    if need_p2:
        d2 = args.direction2 or args.direction
        rl.info("building parent-2 via front-end", direction=d2)
        p2 = front.run(direction=d2)
        _show("PARENT-2 (front-end)", json.dumps({
            "hypothesis": p2.hypothesis, "expression": p2.expression,
            "kv_judger": kv_ops.kv_describe(p2.kv_judger),
        }, indent=2, default=str))

    if args.save_parents:
        kv_ops.kv_save(p1.kv_judger, Path(args.save_parents) / "parent1_kv.pt")
        if p2 is not None:
            kv_ops.kv_save(p2.kv_judger, Path(args.save_parents) / "parent2_kv.pt")
        rl.info("saved parent KVs", dir=args.save_parents)

    # ── 2. mutation ──────────────────────────────────────────────────────────
    if args.mode in ("mutation", "both"):
        rl.info("running mutation (reflection → judger)")
        evo_out = evo.mutate(
            parent_kv_feedback=p1.kv_judger,
            parent_hypothesis=p1.hypothesis,
            parent_expression=p1.expression,
            parent_feedback=args.feedback,
            backtest_summary=args.backtest_summary,
        )
        _show("MUTATION RESULT", json.dumps({
            "parent_expr": p1.expression,
            "mutated_hypothesis": getattr(evo_out, "hypothesis", None),
            "mutated_expression": getattr(evo_out, "expression", None),
            "raw_text": getattr(evo_out, "raw_text", "")[:300],
            "ok": evo_out is not None and bool(getattr(evo_out, "expression", "")),
        }, indent=2, default=str))

    # ── 3. crossover ─────────────────────────────────────────────────────────
    if args.mode in ("crossover", "both") and p2 is not None:
        rl.info("running crossover (kv_concat → judger)")
        cross_out = evo.crossover(parent_kvs=[p1.kv_judger, p2.kv_judger])
        _show("CROSSOVER RESULT", json.dumps({
            "parent1_expr": p1.expression,
            "parent2_expr": p2.expression,
            "crossover_hypothesis": getattr(cross_out, "hypothesis", None),
            "crossover_expression": getattr(cross_out, "expression", None),
            "raw_text": getattr(cross_out, "raw_text", "")[:300],
            "ok": cross_out is not None and bool(getattr(cross_out, "expression", "")),
        }, indent=2, default=str))

    rl.finalize()


if __name__ == "__main__":
    main()
