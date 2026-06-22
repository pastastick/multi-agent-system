"""prod/run.py — CLI entry pipeline produksi. Lihat DESIGN.md §7-8.

  # dry-run (tanpa GPU): render node + cek wiring transfer (no-crop/concat/decode)
  python -m prod.run --generations 3 --latent-steps 10 --transfer kv --dry-run

  # A/B baseline (full-text, kv none, latent 0)
  python -m prod.run --generations 3 --transfer text --dry-run

  # GPU: loop evolusi nyata (butuh runner.py / backtest_fn — F3)
  python -m prod.run --generations 3 --latent-steps 10 --transfer kv
"""
from __future__ import annotations

import argparse
import json

from .config import RunConfig
from .pipeline import EvolutionPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="QuantaLatent production pipeline")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--latent-steps", type=int, default=10)
    ap.add_argument("--transfer", choices=["kv", "text"], default="kv")
    ap.add_argument("--director", choices=["mutation", "crossover"], default="mutation")
    ap.add_argument("--backtest", choices=["mock", "real"], default="mock")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--quiet", action="store_true", help="matikan log terminal")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = RunConfig(
        generations=args.generations,
        latent_steps=args.latent_steps,
        transfer_mode=args.transfer,
        director=args.director,
        backtest_mode=args.backtest,
        decode_temperature=args.temp,
        verbose=not args.quiet,
    )

    if args.dry_run:
        pipe = EvolutionPipeline(cfg)
        res = pipe.dry_run()
        n_missing = sum(int(n["missing_vars"]) for n in res["nodes"])
        print(f"[prod dry] transfer={cfg.transfer_mode} ls={cfg.effective_latent_steps} "
              f"nodes={len(res['nodes'])} ok={res['ok']} missing_vars={n_missing}")
        for n in res["nodes"]:
            flag = " <<MISSING" if n["missing_vars"] else ""
            print(f"  {n['order']:02d} {n['id']:<16} transfer={n['transfer']:<12} "
                  f"no_crop={n['no_crop']}{flag}")
        print(f"[prod dry] artifacts → {pipe.run_dir}")
        return

    # GPU path
    from common import get_backend, get_latent_backend  # type: ignore
    use_latent = cfg.effective_latent_steps > 0 and cfg.transfer_mode == "kv"
    backend = (get_latent_backend(latent_steps_init=max(cfg.latent_steps, 10))
               if use_latent else get_backend())
    pipe = EvolutionPipeline(cfg, backend=backend)
    res = pipe.run()
    print(json.dumps({k: res[k] for k in ("mode", "transfer", "ok", "err")},
                     ensure_ascii=False))
    print(f"[prod] artifacts → {pipe.run_dir}")


if __name__ == "__main__":
    main()
