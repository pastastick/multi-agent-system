#!/usr/bin/env python3
"""
experiments/inspect_chain.py
============================
Jalankan rantai front-end LATEN (proposal → construct → consistency → judger)
SEKALI, lalu PROBE isi KV-cache di SETIAP checkpoint dengan agent `introspect`
(Memory Probe). Tujuan: melihat "apa yang DIPIKIRKAN" tiap agent di laten —
apakah sesuai harapan — BUKAN hanya output teks akhir judger.

Kenapa perlu: proposal/construct/consistency bermode kv_only (tak men-decode
teks), jadi satu-satunya cara membaca isinya = re-attend KV via probe. Lihat
try/probe.py & memory latent_kvcache_mechanics: probe = satu-satunya introspeksi
KV yang feasible (W_K/W_V many-to-one, tak bisa di-decode langsung).

KV HYGIENE (golden rule kv_ops): tiap stage MENGEKSTEND cache stage sebelumnya
IN-PLACE. Maka KV tiap checkpoint DI-DEEPCOPY SEGERA setelah diproduksi (sebelum
stage berikutnya memutasinya), dan probe jalan di atas deepcopy → rantai asli
tak terkontaminasi.

Bisa menyemai rantai seperti evolution (--seed-from mutation|crossover) untuk
membandingkan "apa yang dipikirkan proposal" saat di-seed guidance vs murni.

Contoh
------
  V=/workspace/project/multi-agent-system/.venv/bin/python
  HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 \
    $V experiments/inspect_chain.py --direction "high-volume reversal"
  # rantai yang di-seed mutation director:
  $V experiments/inspect_chain.py --direction "..." --seed-from mutation
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _wrap(s: str, width: int = 100) -> str:
    out = []
    for line in (s or "").splitlines():
        out.append(textwrap.fill(line, width=width) if line.strip() else "")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction", default="high-volume days precede short-horizon reversal")
    ap.add_argument("--seed-from", choices=["none", "mutation", "crossover"], default="none",
                    help="seed rantai dari guidance director (default none = jalur original)")
    ap.add_argument("--feedback", default="Low RankIC (~0.01). Mechanism too volume-"
                    "dominated and fragile under noise; needs a distinct signal family.")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--latent-steps", type=int, default=10, help="DEFAULT 10 (produksi)")
    ap.add_argument("--use-realign", action="store_true")
    ap.add_argument("--knn", action="store_true")
    ap.add_argument("--console", default="WARNING")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    from llm.client import LocalLLMBackend
    from latent_mas.runlog import get_run_logger
    from latent_mas.agent import load_all_agents
    from latent_mas import kv_ops
    from latent_mas.operator_families import diversity_hint

    rl = get_run_logger(run_name="inspect_chain", console_level=args.console)
    out_dir = Path(args.out_dir) if args.out_dir else Path(rl.dir) / "chain_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[chain] loading {args.model} (latent_steps={args.latent_steps}) …", flush=True)
    backend = LocalLLMBackend(
        model_name=args.model, device=args.device, latent_steps=args.latent_steps,
        use_realign=args.use_realign, knn_enabled=args.knn,
    )
    agents = load_all_agents(backend, runlog=rl)

    def probe(label: str, kv) -> str:
        """introspect di atas DEEPCOPY → tak mengganggu KV rantai."""
        if kv is None:
            return "(no KV)"
        res = agents["introspect"].run(past_kv=kv_ops.kv_deepcopy(kv))
        desc = kv_ops.kv_describe(kv)
        txt = res.text or "(empty)"
        block = (f"\n{'═'*92}\n▼ {label.upper()}  [{desc}]\n{'═'*92}\n"
                 + _wrap(txt))
        print(block, flush=True)
        (out_dir / f"{label}.txt").write_text(f"# {label}  {desc}\n\n{txt}")
        return txt

    # ── seed (opsional): guidance KV dari mutation/crossover director ─────────
    seed_kv = None
    if args.seed_from != "none":
        # parent teks sintetis untuk director (seperti exp_judger_quality)
        parent = (f"Hypothesis: high volume precedes reversal\n"
                  f"Expression(s):\n  - TS_ZSCORE($volume, 5) > 0.5\n"
                  f"Backtest: RankIC=0.012\nFeedback: {args.feedback}")
        if args.seed_from == "mutation":
            rg = agents["mutation"].run(past_kv=None, target_text=parent,
                                        direction=args.direction)
        else:
            two = f"[Parent 1]\n{parent}\n\n[Parent 2]\n{parent}"
            rg = agents["crossover"].run(past_kv=None, parents_text=two, n_parents=2,
                                         direction=args.direction)
        seed_kv = rg.kv_cache
        probe(f"guidance_{args.seed_from}", seed_kv)

    # ── rantai front-end; DEEPCOPY tiap KV SEGERA (sebelum stage berikut memutasi) ─
    dhint = diversity_hint([])
    r_prop = agents["proposal"].run(past_kv=seed_kv, direction=args.direction,
                                    market_context="", prior_feedback="", negative_hint="")
    kv_prop = kv_ops.kv_deepcopy(r_prop.kv_cache)

    r_con = agents["construct"].run(past_kv=r_prop.kv_cache, diversity_hint=dhint)
    kv_con = kv_ops.kv_deepcopy(r_con.kv_cache)

    r_cons = agents["consistency"].run(past_kv=r_con.kv_cache)
    kv_cons = kv_ops.kv_deepcopy(r_cons.kv_cache)

    # judger: men-decode teks (kv_and_text). Probe KV consistency yang DILIHAT judger.
    r_judge = agents["judger"].run(past_kv=kv_ops.kv_deepcopy(kv_cons),
                                   direction=args.direction, diversity_hint=dhint)

    # ── probe tiap checkpoint ────────────────────────────────────────────────
    probe("proposal", kv_prop)
    probe("construct", kv_con)
    probe("consistency", kv_cons)

    print(f"\n{'═'*92}\n▼ JUDGER (decoded text — bukan probe)\n{'═'*92}")
    print(_wrap(r_judge.text or "(empty)"), flush=True)
    (out_dir / "judger_decoded.txt").write_text(r_judge.text or "")

    print(f"\n[chain] probe per-stage + judger → {out_dir}")


if __name__ == "__main__":
    main()
