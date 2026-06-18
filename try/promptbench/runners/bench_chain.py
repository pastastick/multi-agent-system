"""
promptbench/runners/bench_chain.py
==================================
Phase B — driver RANTAI multi-agent via KV-cache (LatentMAS working-memory transfer).

Tujuan: setelah Phase A memilih varian+setting terbaik PER AGENT, uji apakah
varian-varian itu tetap baik saat DIRANGKAI lewat KV-cache — dan deteksi dini
bila KV menimbun / collapse di sepanjang rantai.

Rantai didefinisikan di `chains/chain_manifest.yaml` (lihat juga chains/registry.py).
Tiap job = (chain × latent_steps × rep).

KEBENARAN TRANSFER KV (inti Phase B) — lihat juga komentar di run_chain():
  • `latent_pass()` MEMUTASI DynamicCache `past_key_values` IN-PLACE (HF) lalu
    mengembalikannya sebagai output. Jadi kalau KV langkah-i dioper langsung ke
    langkah-(i+1), KV langkah-i ikut berubah → snapshot batas jadi salah & bisa
    terjadi kontaminasi antar-cabang.
  • SOLUSI: clone-on-transfer. Sebelum diumpankan, KV di-`kv_deepcopy()`. Maka
    output tiap langkah membeku (frozen) dan hanya SALINAN yang dimutasi langkah
    berikutnya. Pertumbuhan KV = murni (prompt + latent_steps) langkah itu —
    "latent working memory transfer" yang DISENGAJA, bukan timbunan tak sengaja.
  • Mode kv_and_text sudah meng-CROP answer-token sebelum KV diteruskan (engine),
    jadi token jawaban tidak bocor ke hilir — konsisten filosofi latent comms.
  • Realignment W_a aktif otomatis di latent backend (use_realign=True) untuk
    latent_steps>0; latent_steps=0 = rantai prompt murni tanpa virtual token.

Paralelisme: ProcessPoolExecutor (spawn), dipisah fase TEXT (ls==0 → get_backend)
vs LATENT (ls>0 → get_latent_backend, use_realign). Tiap worker = 1 model.

Jalankan:
  # dry-run (tanpa GPU): render tiap langkah + cek wiring KV + skor placeholder
  python -m try.promptbench.runners.bench_chain --chains pc_2agent,pcj_judger \
      --latent-steps 0,20 --reps 1 --dry-run

  # runpod (GPU):
  python -m try.promptbench.runners.bench_chain \
      --chains front_end_full,front_end_feedback \
      --latent-steps 0,10,20,40 --reps 5 --workers 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, Undefined

_THIS = Path(__file__).resolve()
PROMPTBENCH = _THIS.parent.parent
REPO = PROMPTBENCH.parent.parent

DECODE_TEMPERATURE = 0.7
DEFAULT_MAX_NEW = 512


class _Vis(Undefined):
    def __str__(self): return f"[[MISSING:{self._undefined_name}]]"


# ════════════════════════════════════════════════════════════════════════════
# enumerasi job
# ════════════════════════════════════════════════════════════════════════════

def enumerate_jobs(chains: List[str], latent_steps: List[int], reps: int,
                   overrides: Dict[str, str]) -> List[dict]:
    jobs = []
    for chain in chains:
        for ls in latent_steps:
            for r in range(reps):
                jobs.append({
                    "chain": chain, "latent_steps": ls, "rep": r,
                    "overrides": overrides,
                })
    return jobs


# ════════════════════════════════════════════════════════════════════════════
# rendering (dipakai dry-run & artifact)
# ════════════════════════════════════════════════════════════════════════════

def _load_spec(path: str, agent: str) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return (raw.get("agents") or {})[agent]


def _render(spec: dict, fixtures: dict) -> tuple[str, str]:
    env = Environment(undefined=_Vis)
    system = env.from_string(spec.get("system", "")).render(**fixtures).strip()
    user = env.from_string(spec.get("user", "")).render(**fixtures).strip()
    return system, user


# ════════════════════════════════════════════════════════════════════════════
# eksekusi satu rantai
# ════════════════════════════════════════════════════════════════════════════

def run_chain_job(job: dict, *, dry_run: bool, temp: float) -> dict:
    """Jalankan satu (chain × ls × rep). Lazy-import backend → dry-run bebas GPU."""
    from ..chains import resolve_chain
    from ..artifacts import config_slug, phaseB_step_artifact, phaseB_chain_json, \
        write_step_artifact, write_json
    from ..fixtures_pb import FIXTURES
    from ..diagnostics import (BoundaryRecord, detect_kv_growth,
                               detect_text_collapse, summarize_chain_health)
    from ..scoring import score_chain

    chain = resolve_chain(job["chain"], overrides=job.get("overrides") or {})
    ls = job["latent_steps"]
    rep = job["rep"]
    cfg = config_slug(ls)

    out: Dict[str, Any] = {
        "chain": chain.name, "config": cfg, "latent_steps": ls, "rep": rep,
        "terminal_agent": chain.steps[-1].agent,
        "ok": False, "score": 0.0, "healthy": None, "err": None,
        "steps": [], "elapsed_s": 0.0,
    }

    # ── DRY-RUN: render tiap langkah, cek wiring KV, skor placeholder ────────
    if dry_run:
        prev_len = 0
        all_ok = True
        for st in chain.steps:
            spec = _load_spec(st.variant_path, st.agent)
            system, user = _render(spec, FIXTURES)
            missing = "[[MISSING:" in (system + user)
            all_ok = all_ok and not missing
            eff_ls = st.latent_steps if st.latent_steps is not None else ls
            est_prompt = len((system + " " + user).split())
            # transfer none → mulai 0; chain → lanjut prev_len
            base = prev_len if st.transfer == "chain" else (prev_len if st.transfer == "concat" else 0)
            seq = base + est_prompt + eff_ls
            rec = {
                "step": st.idx, "agent": st.agent, "variant": st.variant_short,
                "mode": ("kv_and_text" if st.is_terminal and st.native_mode == "kv_only"
                         else st.native_mode),
                "transfer": st.transfer, "latent_steps": eff_ls,
                "n_prompt_tokens": est_prompt, "kv_seq_len": seq,
                "prev_seq_len": base, "missing_vars": missing,
            }
            out["steps"].append(rec)
            art = phaseB_step_artifact(chain.name, cfg, rep, st.idx, st.agent)
            write_step_artifact(
                art, header={"chain": chain.name, "config": cfg, "rep": rep,
                             "step": st.idx, "agent": st.agent,
                             "variant": st.variant_short, "transfer": st.transfer,
                             "mode": rec["mode"], "dry_run": True},
                system=system, user=user, response="(dry_run — tidak call LLM)",
                kv_report=rec,
            )
            prev_len = seq
        out["ok"] = all_ok
        out["score"] = 1.0 if all_ok else 0.0
        out["healthy"] = all_ok
        write_json(phaseB_chain_json(chain.name, cfg, rep),
                   {**out, "desc": chain.desc, "dry_run": True})
        return out

    # ── REAL RUN (GPU) ──────────────────────────────────────────────────────
    try:
        sys.path.insert(0, str(REPO / "backend"))
        from ...common import get_backend, get_latent_backend
        from latent_mas.agent import load_agent
        from latent_mas.kv_ops import kv_deepcopy, kv_concat, kv_describe, kv_seq_len

        use_latent = ls > 0 or any(
            (s.latent_steps or 0) > 0 for s in chain.steps)
        backend = get_latent_backend(latent_steps_init=max(ls, 10)) if use_latent \
            else get_backend()

        kv_by_step: Dict[int, Any] = {}     # idx → output KV (FROZEN snapshot)
        boundaries: List[BoundaryRecord] = []
        terminal_text: Optional[str] = None
        terminal_out_tokens = 0
        t0 = time.time()

        for st in chain.steps:
            eff_ls = st.latent_steps if st.latent_steps is not None else ls

            # ── transfer KV: CLONE-ON-TRANSFER (anti penimbunan) ────────────
            if st.transfer == "chain":
                src = kv_by_step.get(st.idx - 1)
                input_kv = kv_deepcopy(src)          # salinan independen → src membeku
            elif st.transfer == "concat":
                # Phase C: gabung KV semua langkah sebelumnya layer-wise
                # (LatentMAS Eq.4). Tiap sumber di-clone agar tak termutasi.
                srcs = [kv_deepcopy(kv_by_step[i]) for i in sorted(kv_by_step)]
                input_kv = kv_concat(srcs) if srcs else None
            else:  # none
                input_kv = None

            prev_len = kv_seq_len(input_kv)

            # ── load agent + atur mode/ls ───────────────────────────────────
            ag = load_agent(st.agent, backend, strict_vars=False,
                            path=Path(st.variant_path))
            ag.spec.latent_steps = eff_ls
            force_decode = st.is_terminal and ag.spec.mode == "kv_only"
            if force_decode:
                ag.spec.mode = "kv_and_text"
            if ag.spec.mode == "kv_and_text":
                if ag.spec.temperature is None:
                    ag.spec.temperature = temp
                if ag.spec.max_new_tokens is None:
                    ag.spec.max_new_tokens = DEFAULT_MAX_NEW

            # ── run (engine meng-crop answer-token utk kv_and_text) ─────────
            with open(os.devnull, "w") as dn, redirect_stdout(dn):
                res = ag.run(past_kv=input_kv, **FIXTURES)
            out_kv = res.kv_cache
            kv_by_step[st.idx] = out_kv               # frozen (next step deepcopy-nya)

            desc = kv_describe(out_kv)
            rec = BoundaryRecord(
                step_idx=st.idx, agent=st.agent, mode=ag.spec.mode,
                latent_steps=eff_ls,
                n_prompt_tokens=int(getattr(res, "n_input_tokens", 0) or 0),
                kv_seq_len=int(getattr(res, "kv_seq_len", 0) or kv_seq_len(out_kv)),
                kv_size_mb=float(desc.get("size_mb", -1.0)),
                transfer=st.transfer, prev_seq_len=prev_len,
            )
            boundaries.append(rec)

            # ── artifact per langkah ────────────────────────────────────────
            spec = _load_spec(st.variant_path, st.agent)
            system, user = _render(spec, FIXTURES)
            step_detail = None
            step_text = res.text if res.text is not None else "(kv_only — tidak di-decode)"
            if st.is_terminal:
                with open(os.devnull, "w") as dn, redirect_stdout(dn):
                    step_detail = score_chain.score_terminal(st.agent, res.text or "")
                terminal_text = res.text or ""
                terminal_out_tokens = int(getattr(res, "n_output_tokens", 0) or 0)

            write_step_artifact(
                phaseB_step_artifact(chain.name, cfg, rep, st.idx, st.agent),
                header={"chain": chain.name, "config": cfg, "rep": rep,
                        "step": st.idx, "agent": st.agent,
                        "variant": st.variant_short, "transfer": st.transfer,
                        "mode": ag.spec.mode, "latent_steps": eff_ls,
                        "terminal": st.is_terminal},
                system=system, user=user, response=step_text,
                kv_report=rec.to_dict(), score_detail=step_detail,
            )
            out["steps"].append({**rec.to_dict(),
                                 "text_len": len(res.text) if res.text else 0})

        out["elapsed_s"] = round(time.time() - t0, 2)

        # ── skor terminal + collapse ────────────────────────────────────────
        term = chain.steps[-1].agent
        with open(os.devnull, "w") as dn, redirect_stdout(dn):
            term_detail = score_chain.score_terminal(term, terminal_text or "")
        out["score"] = float(term_detail.get("score", 0.0))
        out["score_detail"] = term_detail
        out["ok"] = bool(terminal_text and terminal_text.strip())

        kv_health = detect_kv_growth(boundaries)
        txt_health = detect_text_collapse(
            terminal_text, output_tokens=terminal_out_tokens,
            max_new_tokens=DEFAULT_MAX_NEW,
            parser_ok=term_detail.get("parser_ok"),
        )
        health = summarize_chain_health(kv_health, txt_health)
        out["healthy"] = health["healthy"]
        out["collapse"] = {"kv": kv_health, "text": txt_health, "health": health}

        write_json(phaseB_chain_json(chain.name, cfg, rep),
                   {**out, "desc": chain.desc})

        # ── lepas KV agar VRAM segera bebas untuk job berikutnya ────────────
        kv_by_step.clear()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    except Exception as e:  # noqa: BLE001
        out["err"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-1500:]
    return out


# ════════════════════════════════════════════════════════════════════════════
# orkestrasi + agregasi
# ════════════════════════════════════════════════════════════════════════════

def _run_phase(jobs: List[dict], workers: int, dry_run: bool, temp: float) -> List[dict]:
    if not jobs:
        return []
    if workers <= 1 or dry_run:
        return [run_chain_job(j, dry_run=dry_run, temp=temp) for j in jobs]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_chain_job, j, dry_run=dry_run, temp=temp): j for j in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


def aggregate_and_write(results: List[dict]) -> Path:
    from ..artifacts import PHASE_B
    import statistics

    groups: Dict[tuple, List[dict]] = {}
    for r in results:
        groups.setdefault((r["chain"], r["config"]), []).append(r)

    rows = []
    for (chain, cfg), items in sorted(groups.items()):
        scores = [it.get("score", 0.0) for it in items]
        n = len(items)
        rows.append({
            "chain": chain, "config": cfg,
            "terminal_agent": items[0].get("terminal_agent", "?"),
            "n": n,
            "score_mean": round(statistics.fmean(scores), 3),
            "score_std": round(statistics.pstdev(scores), 3) if n > 1 else 0.0,
            "ok_rate": round(sum(int(it.get("ok", False)) for it in items) / n, 3),
            "healthy_rate": round(
                sum(int(bool(it.get("healthy"))) for it in items) / n, 3),
            "kv_flag_rate": round(sum(
                int(bool((it.get("collapse") or {}).get("kv", {}).get("flags")))
                for it in items) / n, 3),
            "text_collapse_rate": round(sum(
                int(bool((it.get("collapse") or {}).get("text", {}).get("collapsed")))
                for it in items) / n, 3),
            "err_rate": round(sum(int(bool(it.get("err"))) for it in items) / n, 3),
        })

    rows.sort(key=lambda x: (x["chain"], x["config"]))
    PHASE_B.mkdir(parents=True, exist_ok=True)
    cols = ["chain", "config", "terminal_agent", "n", "score_mean", "score_std",
            "ok_rate", "healthy_rate", "kv_flag_rate", "text_collapse_rate", "err_rate"]
    csv_path = PHASE_B / "scoreboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = ["# Phase B scoreboard (rantai KV)\n",
          f"_generated {time.strftime('%Y-%m-%d %H:%M')}_\n",
          "| chain | config | terminal | n | score | std | ok | healthy | kv_flag | txt_collapse | err |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['chain']} | {r['config']} | {r['terminal_agent']} | {r['n']} | "
                  f"{r['score_mean']} | {r['score_std']} | {r['ok_rate']} | "
                  f"{r['healthy_rate']} | {r['kv_flag_rate']} | "
                  f"{r['text_collapse_rate']} | {r['err_rate']} |")
    (PHASE_B / "scoreboard.md").write_text("\n".join(md), encoding="utf-8")
    return csv_path


# ════════════════════════════════════════════════════════════════════════════
def _parse_overrides(s: Optional[str]) -> Dict[str, str]:
    """`--pick judger=working,proposal=git_optimalisasi` → dict."""
    out: Dict[str, str] = {}
    if not s:
        return out
    for kv in s.split(","):
        kv = kv.strip()
        if "=" in kv:
            a, v = kv.split("=", 1)
            out[a.strip()] = v.strip()
    return out


def main():
    from ..chains import all_chain_names

    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default=",".join(all_chain_names()),
                    help="nama chain (default: semua di chain_manifest.yaml)")
    ap.add_argument("--latent-steps", default="0,10,20,40")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pick", default=None,
                    help="override varian: 'judger=working,proposal=git_optimalisasi'")
    ap.add_argument("--temp", type=float, default=DECODE_TEMPERATURE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chains = [c.strip() for c in args.chains.split(",") if c.strip()]
    lsteps = [int(x) for x in args.latent_steps.split(",") if x.strip() != ""]
    overrides = _parse_overrides(args.pick)
    jobs = enumerate_jobs(chains, lsteps, args.reps, overrides)
    text_jobs = [j for j in jobs if j["latent_steps"] == 0]
    lat_jobs = [j for j in jobs if j["latent_steps"] > 0]

    print(f"[bench_chain] chains={chains} latent_steps={lsteps} reps={args.reps} "
          f"workers={args.workers} dry_run={args.dry_run} overrides={overrides}")
    print(f"[bench_chain] total jobs={len(jobs)} (text={len(text_jobs)} latent={len(lat_jobs)})")

    results = []
    if text_jobs:
        print(f"[bench_chain] === TEXT phase (ls=0): {len(text_jobs)} jobs ===")
        results += _run_phase(text_jobs, args.workers, args.dry_run, args.temp)
    if lat_jobs:
        print(f"[bench_chain] === LATENT phase (use_realign): {len(lat_jobs)} jobs ===")
        results += _run_phase(lat_jobs, args.workers, args.dry_run, args.temp)

    n_ok = sum(int(r.get("ok", False)) for r in results)
    n_err = sum(int(bool(r.get("err"))) for r in results)
    n_unhealthy = sum(int(r.get("healthy") is False) for r in results)
    print(f"[bench_chain] done: {len(results)} results, ok={n_ok}, "
          f"err={n_err}, unhealthy={n_unhealthy}")
    csv_path = aggregate_and_write(results)
    print(f"[bench_chain] scoreboard → {csv_path}")


if __name__ == "__main__":
    main()
