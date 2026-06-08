#!/usr/bin/env python3
"""
experiments/run_evolution.py
====================
Uji kemampuan 3 agent evolution SECARA TERISOLASI — tanpa factor_mining/backtest.

Alur (GUIDANCE → re-entry front-end, sesuai produksi):
  1. Jalankan FrontEndPipeline untuk membuat 1-2 "parent" (hypothesis+expression).
  2. Mutation : front.run_evolution(kind="mutation") — agent guidance kv_only
     menetapkan arah refine (seed None, parent sbg TEKS) → menyemai propose→judger.
  3. Crossover: front.run_evolution(kind="crossover") — agent guidance kv_only
     menetapkan arah fusi 2 parent (teks, seed None) → menyemai propose→judger.
  4. Cetak hypothesis/expression hasil + KV describe (per ronde harus terbatas).

Feedback parent & ringkasan backtest disuntik manual (--feedback / --backtest-
summary) — keduanya masuk ke TEKS parent: ubah feedback → lihat apakah mutation
judger merevisi ke arah yang tepat.

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

# Ekspresi invalid yang natural + PASTI ditolak gate. Catatan: gate
# (factors.coder.expr_parser.parse_expression) hanya cek balance kurung,
# operator invalid, dan grammar — TIDAK cek arity/nama fungsi. Jadi kurung
# tak seimbang adalah kegagalan deterministik yang jelas "repairable"
# (model tinggal menutup kurung). (Ekspresi 3-arg seperti
# "TS_ZSCORE($volume, $return, 5)" justru LOLOS gate — arity tak dicek.)
DEFAULT_BAD_EXPR = "TS_ZSCORE($volume, 20"


def _show(title: str, obj) -> None:
    print(f"\n{'═'*70}\n{title}\n{'═'*70}")
    print(obj)


def _parent_block(out, *, feedback: str = "", backtest: str = "", label: str = "") -> str:
    """Format FrontEndOutput jadi TEKS parent untuk run_evolution (mirror
    loop._format_parents_text, tapi dari FrontEndOutput bukan StrategyTrajectory)."""
    exprs = out.expressions or ([out.expression] if out.expression else [])
    parts = []
    if out.hypothesis:
        parts.append(f"Hypothesis: {out.hypothesis}")
    if exprs:
        parts.append("Expression(s):\n" + "\n".join(f"  - {e}" for e in exprs))
    if backtest:
        parts.append(f"Backtest: {backtest}")
    if feedback:
        parts.append(f"Feedback: {feedback}")
    block = "\n".join(parts) or "(empty)"
    return f"[{label}]\n{block}" if label else block


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolated test of evolution agents")
    ap.add_argument("--direction", required=True, help="parent-1 direction")
    ap.add_argument("--direction2", default=None,
                    help="parent-2 direction (untuk crossover; default = direction)")
    ap.add_argument("--mode", choices=["mutation", "crossover", "both"], default="both")
    ap.add_argument("--feedback", default="Low IC. The factor mechanism is weak and "
                    "too correlated with raw volume; needs a different signal family.",
                    help="synthetic parent feedback (masuk ke teks parent mutation)")
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
    ap.add_argument("--test-repair", nargs="?", const=DEFAULT_BAD_EXPR, default=None,
                    metavar="BAD_EXPR",
                    help="paksa gate failure: jalankan _gate_and_repair pada ekspresi "
                         "sengaja invalid (default: kurung tak seimbang yang ditolak "
                         "gate) untuk memvalidasi agent repair secara terisolasi")
    ap.add_argument("--console", default="INFO")
    args = ap.parse_args()

    from llm.client import LocalLLMBackend
    from latent_mas.runlog import get_run_logger
    from latent_mas.pipeline import FrontEndPipeline
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

    # ── 1. buat parent-1 (dan parent-2 untuk crossover) ──────────────────────
    rl.info("building parent-1 via front-end")
    p1 = front.run(direction=args.direction)
    _show("PARENT-1 (front-end)", json.dumps({
        "hypothesis": p1.hypothesis, "expression": p1.expression,
        "repaired": p1.repaired, "gate_error": p1.gate_error,
        "kv_judger": kv_ops.kv_describe(p1.kv_judger),
    }, indent=2, default=str))

    # ── 1b. (opsional) validasi repair dengan gate failure paksa ─────────────
    # Berangkat dari kv_consist parent-1 yang pristine (sama seperti loop nyata),
    # suntik ekspresi invalid → _gate_and_repair harus memulihkannya.
    if args.test_repair is not None:
        bad = args.test_repair
        gate_ok, gate_err = front.gate(bad)
        rl.info("forcing gate failure to test repair", bad_expr=bad,
                gate_ok=gate_ok, gate_err=gate_err)
        fixed, repaired, attempts, err, _kv = front._gate_and_repair(bad, p1.kv_consist)
        _show("REPAIR VALIDATION", json.dumps({
            "injected_bad_expr": bad,
            "gate_rejected": not gate_ok,
            "gate_error": gate_err,
            "repaired": repaired,
            "attempts": attempts,
            "final_expression": fixed,
            "final_gate_ok": front.gate(fixed)[0] if fixed else False,
            "recovered": repaired and front.gate(fixed)[0],
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

    # ── 2. mutation: JUDGER-ONLY, seed=None, parent sebagai teks ─────────────
    if args.mode in ("mutation", "both"):
        rl.info("running mutation (judger-only: revise target)")
        parent_text = _parent_block(p1, feedback=args.feedback,
                                    backtest=args.backtest_summary)
        child = front.run_evolution(kind="mutation", parent_text=parent_text,
                                    n_parents=1, direction=args.direction)
        _show("MUTATION RESULT", json.dumps({
            "parent_expr": p1.expression,
            "mutated_hypothesis": child.hypothesis,
            "mutated_expressions": child.expressions,
            "repaired": child.repaired,
            "gate_error": child.gate_error,
            "kv_judger": kv_ops.kv_describe(child.kv_judger),
            "ok": bool(child.expressions),
        }, indent=2, default=str))

    # ── 3. crossover: JUDGER-ONLY recombination, seed=None, parents sebagai teks
    if args.mode in ("crossover", "both") and p2 is not None:
        rl.info("running crossover (judger-only: recombination)")
        parent_text = "\n\n".join([
            _parent_block(p1, backtest=args.backtest_summary, label="Parent 1"),
            _parent_block(p2, backtest=args.backtest_summary, label="Parent 2"),
        ])
        child = front.run_evolution(kind="crossover", parent_text=parent_text,
                                    n_parents=2, direction=args.direction)
        _show("CROSSOVER RESULT", json.dumps({
            "parent1_expr": p1.expression,
            "parent2_expr": p2.expression,
            "crossover_hypothesis": child.hypothesis,
            "crossover_expressions": child.expressions,
            "repaired": child.repaired,
            "gate_error": child.gate_error,
            "kv_judger": kv_ops.kv_describe(child.kv_judger),
            "ok": bool(child.expressions),
        }, indent=2, default=str))

    rl.finalize()


if __name__ == "__main__":
    main()
