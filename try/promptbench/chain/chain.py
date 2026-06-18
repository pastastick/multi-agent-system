"""
promptbench/chain/chain.py
==========================
Phase B (desain STAGES) — rantai multi-agent via KV-cache, BERTAHAP. Tujuan:
lihat apakah akumulasi KV lintas-agent menurunkan kualitas output (terutama
ekspresi judger) relatif ke skor isolasi Phase A. Bila ya → bug ada di
chaining/KV, bukan prompt.

Dua desain Phase B HIDUP BERDAMPINGAN (mengukur hal berbeda):
  • STAGES (file ini)         : prefix yang makin panjang pada latent_steps TETAP
    (top per-agent dari scoreboard) → mengisolasi kontribusi MARGINAL tiap agent
    + perbandingan skip-consistency (s2 vs s3) + evolution-seed (s5/s6).
  • CHAINS (runners/bench_chain.py) : skenario tematik × GRID latent_steps × rep.

INFRASTRUKTUR BERSAMA (konsolidasi 2026-06-18) — keduanya memakai:
  diagnostics/collapse.py  (detektor KV-growth + teks),
  artifacts.py             (path & format artefak nested),
  scoring/score_chain.py   (skor terminal + parser_hook fallback + trace).

DISIPLIN KV = LINEAR PENUH (keputusan user 2026-06-18): tiap langkah meng-chain
KV langkah sebelumnya via kv_deepcopy (clone-on-transfer → snapshot batas beku,
tak ada penimbunan tak sengaja). Termasuk feedback yang MENG-CHAIN dari judger
(bukan dicabang dari kv_consist). Pertumbuhan KV = murni (prompt + latent_steps).

Tahap (prefix makin panjang; TIP = langkah terakhir, selalu di-decode & diskor):
  s1_pc      proposal → construct[tip]
  s2_pcj     proposal → construct → judger[tip]              (consistency dilewati)
  s3_pccj    proposal → construct → consistency → judger[tip] (front-end penuh)
  s4_full_fb proposal → construct → consistency → judger → feedback[tip]
  s5_mut     mutation → front-end penuh → judger[tip]
  s6_cross   crossover → front-end penuh → judger[tip]

Metode input agent = SERAGAM **FIXTURES (impl 2): tiap agent dirender dari
fixtures deterministik (load_agent strict_vars=False mengabaikan var tak terpakai).

Jalankan (GPU):
  python -m try.promptbench.chain.chain --stages s1_pc,s2_pcj,s3_pccj,s4_full_fb --reps 3
  python -m try.promptbench.chain.chain --stages all --reps 3 \
      --pick judger=gate_deterministik --ls judger=20 --uniform-ls 20 --workers 3
  python -m try.promptbench.chain.chain --dry-run   # tanpa GPU: cek wiring & picks
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
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jinja2 import Environment, Undefined

_THIS = Path(__file__).resolve()
PROMPTBENCH = _THIS.parent.parent
REPO = PROMPTBENCH.parent.parent
VARIANTS_DIR = PROMPTBENCH / "variants"
SCOREBOARD_CSV = PROMPTBENCH / "results" / "phaseA" / "scoreboard.csv"

DECODE_TEMPERATURE = 0.7
DEFAULT_MAX_NEW = 30000   # selaras dgn runners/bench_chain.py (keputusan user)
DEFAULT_LS = 20           # latent_steps default utk agent tanpa pick Phase A

FRONTEND = ["proposal", "construct", "consistency"]

# singkatan agent utk slug config (mirror artifacts._AGENT_ABBREV)
_ABBREV = {"proposal": "p", "construct": "c", "consistency": "co",
           "judger": "j", "feedback": "fb", "mutation": "mut", "crossover": "cr"}

# fallback bila scoreboard.csv belum ada (mis. dry-run di mesin tanpa Phase A).
# Diambil dari scoreboard 2026-06-16 (baris teratas tiap agent).
HARDCODED_PICKS: Dict[str, Tuple[str, int]] = {
    "proposal":    ("proposal__working__258abdbbccea", 60),
    "construct":   ("construct__working__56396e7d44b0", 0),
    "consistency": ("consistency__authored_claude_latentpaper__88133558d50d", 0),
    "judger":      ("judger__git_gate_deterministik__6ce003675450", 20),
}


class _Vis(Undefined):
    def __str__(self): return f"[[MISSING:{self._undefined_name}]]"


# ════════════════════════════════════════════════════════════════════════════
# pemilihan varian
# ════════════════════════════════════════════════════════════════════════════

def load_manifest() -> dict:
    return yaml.safe_load((VARIANTS_DIR / "variants_manifest.yaml").read_text()) or {}


def _top_from_scoreboard() -> Dict[str, Tuple[str, int]]:
    """Baca scoreboard.csv → {agent: (variant_id, latent_steps)} skor tertinggi."""
    if not SCOREBOARD_CSV.exists():
        return {}
    best: Dict[str, Tuple[float, str, int]] = {}
    with SCOREBOARD_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            agent = row["agent"]
            try:
                sc = float(row.get("score_mean") or 0)
                ls = int(row.get("latent_steps") or 0)
            except ValueError:
                continue
            if agent not in best or sc > best[agent][0]:
                best[agent] = (sc, row["variant_id"], ls)
    return {a: (vid, ls) for a, (sc, vid, ls) in best.items()}


def _manifest_first(agent: str, man: dict) -> Optional[str]:
    entries = man.get(agent, [])
    return entries[0]["variant_id"] if entries else None


def resolve_picks(
    man: dict,
    pick_overrides: Dict[str, str],
    ls_overrides: Dict[str, int],
    uniform_ls: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    """Tentukan {agent: {variant_id, variant_path, latent_steps}} untuk semua agent
    yang dipakai chain. Prioritas variant_id: override > scoreboard top > hardcoded
    > manifest[0]. Prioritas latent_steps: uniform_ls > override > pick-derived > DEFAULT_LS.
    """
    sb = _top_from_scoreboard()
    needed = FRONTEND + ["judger", "feedback", "repair", "mutation", "crossover"]
    out: Dict[str, Dict[str, Any]] = {}
    for agent in needed:
        vid: Optional[str] = None
        base_ls = DEFAULT_LS
        if agent in sb:
            vid, base_ls = sb[agent]
        elif agent in HARDCODED_PICKS:
            vid, base_ls = HARDCODED_PICKS[agent]
        if agent in pick_overrides:
            match = _match_variant(agent, pick_overrides[agent], man)
            if match:
                vid = match
        if vid is None:
            vid = _manifest_first(agent, man)
        if vid is None:
            continue  # agent tak punya varian → lewati

        if uniform_ls is not None:
            ls = uniform_ls
        elif agent in ls_overrides:
            ls = ls_overrides[agent]
        else:
            ls = base_ls

        out[agent] = {
            "variant_id": vid,
            "variant_path": str(_variant_path(agent, vid, man)),
            "latent_steps": ls,
        }
    return out


def _match_variant(agent: str, substr: str, man: dict) -> Optional[str]:
    for v in man.get(agent, []):
        if substr in v["variant_id"]:
            return v["variant_id"]
    return None


def _variant_path(agent: str, vid: str, man: dict) -> Path:
    for v in man.get(agent, []):
        if v["variant_id"] == vid:
            return PROMPTBENCH / v["path"]
    return VARIANTS_DIR / agent / f"{vid}.yaml"


# ════════════════════════════════════════════════════════════════════════════
# stage spec → urutan langkah linear
# ════════════════════════════════════════════════════════════════════════════
# tiap stage = (urutan agent front-end yg dijalankan, tip, evolution-seed)
STAGES: Dict[str, Dict[str, Any]] = {
    "s1_pc":      {"front": ["proposal", "construct"], "tip": "construct", "evo": None},
    "s2_pcj":     {"front": ["proposal", "construct"], "tip": "judger", "evo": None},
    "s3_pccj":    {"front": FRONTEND, "tip": "judger", "evo": None},
    "s4_full_fb": {"front": FRONTEND, "tip": "feedback", "evo": None},
    "s5_mut":     {"front": FRONTEND, "tip": "judger", "evo": "mutation"},
    "s6_cross":   {"front": FRONTEND, "tip": "judger", "evo": "crossover"},
}
STAGE_ORDER = list(STAGES.keys())


def stage_steps(stage: str) -> List[str]:
    """Urutan agent LINEAR untuk satu stage (evo→front→[judger]→[feedback]).
    Langkah terakhir = tip (selalu di-decode)."""
    spec = STAGES[stage]
    steps = list(spec["front"])
    if spec["evo"]:
        steps = [spec["evo"]] + steps
    tip = spec["tip"]
    if tip in ("judger", "feedback") and "judger" not in steps:
        steps.append("judger")
    if tip == "feedback" and "feedback" not in steps:
        steps.append("feedback")
    return steps


def _stage_config(stage: str, picks: Dict[str, Dict[str, Any]]) -> str:
    """Slug config per-stage dari ls tiap agent yang terlibat, mis. `p60_c0_j20`."""
    parts = []
    for a in stage_steps(stage):
        if a in picks:
            parts.append(f"{_ABBREV.get(a, a[:2])}{picks[a]['latent_steps']}")
    return "_".join(parts) or "ls0"


# ════════════════════════════════════════════════════════════════════════════
# rendering (artefak)
# ════════════════════════════════════════════════════════════════════════════

def _load_spec(path: str, agent: str) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return (raw.get("agents") or {})[agent]


def _render(spec: dict, fixtures: dict) -> Tuple[str, str]:
    env = Environment(undefined=_Vis)
    system = env.from_string(spec.get("system", "")).render(**fixtures).strip()
    user = env.from_string(spec.get("user", "")).render(**fixtures).strip()
    return system, user


# ════════════════════════════════════════════════════════════════════════════
# runner satu stage
# ════════════════════════════════════════════════════════════════════════════

def _stage_uses_latent(stage: str, picks: Dict[str, Dict[str, Any]]) -> bool:
    return any(picks[a]["latent_steps"] > 0 for a in stage_steps(stage) if a in picks)


def run_stage(stage: str, picks: Dict[str, Dict[str, Any]], rep: int,
              *, temp: float, dry_run: bool) -> dict:
    """Jalankan satu stage satu kali (rantai LINEAR). Tulis artefak + kembalikan dict."""
    from ..artifacts import (config_slug, phaseB_step_artifact, phaseB_chain_json,
                             write_step_artifact, write_json, variant_short)
    from ..fixtures_pb import FIXTURES
    from ..diagnostics import (BoundaryRecord, detect_kv_growth,
                               detect_text_collapse, summarize_chain_health)
    from ..scoring import score_chain

    steps = stage_steps(stage)
    terminal = steps[-1]
    cfg = _stage_config(stage, picks)

    out: Dict[str, Any] = {
        "stage": stage, "config": cfg, "rep": rep, "terminal_agent": terminal,
        "picks": {a: variant_short(picks[a]["variant_id"], a) for a in steps if a in picks},
        "ok": False, "score": 0.0, "healthy": None, "err": None,
        "steps": [], "elapsed_s": 0.0,
    }

    # ── DRY-RUN: render tiap langkah + cek wiring KV + skor placeholder ──────
    if dry_run:
        prev_len, all_ok = 0, True
        for idx, agent in enumerate(steps):
            if agent not in picks:
                all_ok = False
                continue
            p = picks[agent]
            spec = _load_spec(p["variant_path"], agent)
            system, user = _render(spec, FIXTURES)
            missing = "[[MISSING:" in (system + user)
            all_ok = all_ok and not missing
            ls = p["latent_steps"]
            transfer = "none" if idx == 0 else "chain"
            est_prompt = len((system + " " + user).split())
            base = prev_len if transfer == "chain" else 0
            seq = base + est_prompt + ls
            rec = {"step": idx, "agent": agent,
                   "variant": variant_short(p["variant_id"], agent),
                   "transfer": transfer, "latent_steps": ls,
                   "n_prompt_tokens": est_prompt, "kv_seq_len": seq,
                   "prev_seq_len": base, "missing_vars": missing}
            out["steps"].append(rec)
            write_step_artifact(
                phaseB_step_artifact(stage, cfg, rep, idx, agent),
                header={"stage": stage, "config": cfg, "rep": rep, "step": idx,
                        "agent": agent, "variant": rec["variant"],
                        "transfer": transfer, "dry_run": True},
                system=system, user=user, response="(dry_run — tidak call LLM)",
                kv_report=rec)
            prev_len = seq
        out["ok"] = all_ok
        out["score"] = 1.0 if all_ok else 0.0
        out["healthy"] = all_ok
        write_json(phaseB_chain_json(stage, cfg, rep), {**out, "dry_run": True})
        return out

    # ── REAL RUN (GPU) ──────────────────────────────────────────────────────
    try:
        sys.path.insert(0, str(REPO / "backend"))
        from ...common import get_backend, get_latent_backend
        from latent_mas.agent import load_agent
        from latent_mas.kv_ops import kv_deepcopy, kv_describe, kv_seq_len

        max_ls = max((picks[a]["latent_steps"] for a in steps if a in picks), default=0)
        use_latent = max_ls > 0
        backend = get_latent_backend(latent_steps_init=max(max_ls, 10)) if use_latent \
            else get_backend()

        prev_kv = None
        boundaries: List[BoundaryRecord] = []
        terminal_text: Optional[str] = None
        terminal_out_tokens = 0
        t0 = time.time()

        for idx, agent in enumerate(steps):
            p = picks[agent]
            ls = p["latent_steps"]
            is_terminal = (idx == len(steps) - 1)

            # transfer LINEAR clone-on-transfer (anti penimbunan)
            transfer = "none" if idx == 0 else "chain"
            input_kv = kv_deepcopy(prev_kv) if transfer == "chain" else None
            prev_len = kv_seq_len(input_kv)

            ag = load_agent(agent, backend, strict_vars=False,
                            path=Path(p["variant_path"]))
            ag.spec.latent_steps = ls
            force_decode = is_terminal and ag.spec.mode == "kv_only"
            if force_decode:
                ag.spec.mode = "kv_and_text"
            if ag.spec.mode == "kv_and_text":
                if ag.spec.temperature is None:
                    ag.spec.temperature = temp
                if ag.spec.max_new_tokens is None:
                    ag.spec.max_new_tokens = DEFAULT_MAX_NEW

            with open(os.devnull, "w") as dn, redirect_stdout(dn):
                res = ag.run(past_kv=input_kv, **FIXTURES)
            out_kv = res.kv_cache
            prev_kv = out_kv  # frozen snapshot; langkah berikut kv_deepcopy-nya

            desc = kv_describe(out_kv)
            rec = BoundaryRecord(
                step_idx=idx, agent=agent, mode=ag.spec.mode, latent_steps=ls,
                n_prompt_tokens=int(getattr(res, "n_input_tokens", 0) or 0),
                kv_seq_len=int(getattr(res, "kv_seq_len", 0) or kv_seq_len(out_kv)),
                kv_size_mb=float(desc.get("size_mb", -1.0)),
                transfer=transfer, prev_seq_len=prev_len,
            )
            boundaries.append(rec)

            spec = _load_spec(p["variant_path"], agent)
            system, user = _render(spec, FIXTURES)
            step_text = res.text if res.text is not None else "(kv_only — tidak di-decode)"
            step_detail = None
            if is_terminal:
                with open(os.devnull, "w") as dn, redirect_stdout(dn):
                    step_detail = score_chain.score_terminal(agent, res.text or "")
                terminal_text = res.text or ""
                terminal_out_tokens = int(getattr(res, "n_output_tokens", 0) or 0)

            write_step_artifact(
                phaseB_step_artifact(stage, cfg, rep, idx, agent),
                header={"stage": stage, "config": cfg, "rep": rep, "step": idx,
                        "agent": agent, "variant": variant_short(p["variant_id"], agent),
                        "transfer": transfer, "mode": ag.spec.mode,
                        "latent_steps": ls, "terminal": is_terminal},
                system=system, user=user, response=step_text,
                kv_report=rec.to_dict(), score_detail=step_detail)
            out["steps"].append({**rec.to_dict(),
                                 "text_len": len(res.text) if res.text else 0})

        out["elapsed_s"] = round(time.time() - t0, 2)

        with open(os.devnull, "w") as dn, redirect_stdout(dn):
            term_detail = score_chain.score_terminal(terminal, terminal_text or "")
        out["score"] = float(term_detail.get("score", 0.0))
        out["score_detail"] = term_detail
        out["ok"] = bool(terminal_text and terminal_text.strip())

        kv_health = detect_kv_growth(boundaries)
        txt_health = detect_text_collapse(
            terminal_text, output_tokens=terminal_out_tokens,
            max_new_tokens=DEFAULT_MAX_NEW, parser_ok=term_detail.get("parser_ok"))
        health = summarize_chain_health(kv_health, txt_health)
        out["healthy"] = health["healthy"]
        out["collapse"] = {"kv": kv_health, "text": txt_health, "health": health}

        write_json(phaseB_chain_json(stage, cfg, rep), {**out})

        prev_kv = None
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

def _run_phase(jobs: List[Tuple[str, int]], picks: Dict[str, Dict[str, Any]],
               workers: int, dry_run: bool, temp: float) -> List[dict]:
    if not jobs:
        return []
    if workers <= 1 or dry_run:
        return [run_stage(s, picks, r, temp=temp, dry_run=dry_run) for s, r in jobs]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_stage, s, picks, r, temp=temp, dry_run=dry_run): (s, r)
                for s, r in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


def aggregate_and_write(results: List[dict]) -> Path:
    from ..artifacts import PHASE_B
    import statistics

    PHASE_B.mkdir(parents=True, exist_ok=True)
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in results:
        groups.setdefault((r["stage"], r["config"]), []).append(r)

    rows = []
    for (stage, cfg), items in sorted(groups.items(),
                                      key=lambda kv: (STAGE_ORDER.index(kv[0][0])
                                                      if kv[0][0] in STAGE_ORDER else 99,
                                                      kv[0][1])):
        scores = [it.get("score", 0.0) for it in items]
        n = len(items)
        rows.append({
            "stage": stage, "config": cfg,
            "terminal_agent": items[0].get("terminal_agent", "?"), "n": n,
            "score_mean": round(statistics.fmean(scores), 3),
            "score_std": round(statistics.pstdev(scores), 3) if n > 1 else 0.0,
            "ok_rate": round(sum(int(it.get("ok", False)) for it in items) / n, 3),
            "healthy_rate": round(sum(int(bool(it.get("healthy"))) for it in items) / n, 3),
            "kv_flag_rate": round(sum(
                int(bool((it.get("collapse") or {}).get("kv", {}).get("flags")))
                for it in items) / n, 3),
            "text_collapse_rate": round(sum(
                int(bool((it.get("collapse") or {}).get("text", {}).get("collapsed")))
                for it in items) / n, 3),
            "err_rate": round(sum(int(bool(it.get("err"))) for it in items) / n, 3),
        })

    cols = ["stage", "config", "terminal_agent", "n", "score_mean", "score_std",
            "ok_rate", "healthy_rate", "kv_flag_rate", "text_collapse_rate", "err_rate"]
    csv_path = PHASE_B / "scoreboard_stages.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = ["# Phase B scoreboard (STAGES — prefix bertahap)\n",
          f"_generated {time.strftime('%Y-%m-%d %H:%M')}_\n",
          "| stage | config | terminal | n | score | std | ok | healthy | kv_flag | txt_collapse | err |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['stage']} | {r['config']} | {r['terminal_agent']} | {r['n']} | "
                  f"{r['score_mean']} | {r['score_std']} | {r['ok_rate']} | "
                  f"{r['healthy_rate']} | {r['kv_flag_rate']} | "
                  f"{r['text_collapse_rate']} | {r['err_rate']} |")
    (PHASE_B / "scoreboard_stages.md").write_text("\n".join(md), encoding="utf-8")
    return csv_path


# ════════════════════════════════════════════════════════════════════════════
def _parse_kv_overrides(items: List[str], cast=str) -> dict:
    """'agent=val' → {agent: cast(val)}."""
    out = {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
            out[k.strip()] = cast(v.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="s1_pc,s2_pcj,s3_pccj,s4_full_fb",
                    help="comma list atau 'all'")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pick", action="append", default=[],
                    help="agent=substr_variant (override pemilihan)")
    ap.add_argument("--ls", action="append", default=[],
                    help="agent=N (override latent_steps per agent)")
    ap.add_argument("--uniform-ls", type=int, default=None,
                    help="paksa SEMUA agent ke latent_steps ini")
    ap.add_argument("--temp", type=float, default=DECODE_TEMPERATURE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stages = STAGE_ORDER if args.stages == "all" else \
        [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"stage tak dikenal: {s}. Pilihan: {STAGE_ORDER}")

    man = load_manifest()
    picks = resolve_picks(
        man,
        _parse_kv_overrides(args.pick, str),
        _parse_kv_overrides(args.ls, int),
        args.uniform_ls,
    )
    print("[chain] picks:")
    for a, p in picks.items():
        print(f"  {a:12s} ls={p['latent_steps']:<3} {p['variant_id']}")

    # pisah job TEXT (stage tanpa latent) vs LATENT (ada ls>0) → backend tak reload.
    jobs = [(s, r) for s in stages for r in range(args.reps)]
    text_jobs = [(s, r) for (s, r) in jobs if not _stage_uses_latent(s, picks)]
    lat_jobs = [(s, r) for (s, r) in jobs if _stage_uses_latent(s, picks)]
    print(f"[chain] stages={stages} reps={args.reps} workers={args.workers} "
          f"dry_run={args.dry_run}")
    print(f"[chain] total jobs={len(jobs)} (text={len(text_jobs)} latent={len(lat_jobs)})")

    results = []
    if text_jobs:
        print(f"[chain] === TEXT phase (ls=0): {len(text_jobs)} jobs ===")
        results += _run_phase(text_jobs, picks, args.workers, args.dry_run, args.temp)
    if lat_jobs:
        print(f"[chain] === LATENT phase (use_realign): {len(lat_jobs)} jobs ===")
        results += _run_phase(lat_jobs, picks, args.workers, args.dry_run, args.temp)

    n_ok = sum(int(r.get("ok", False)) for r in results)
    n_err = sum(int(bool(r.get("err"))) for r in results)
    n_unhealthy = sum(int(r.get("healthy") is False) for r in results)
    print(f"[chain] done: {len(results)} runs, ok={n_ok}, err={n_err}, "
          f"unhealthy={n_unhealthy}")
    sb = aggregate_and_write(results)
    print(f"[chain] scoreboard → {sb}")


if __name__ == "__main__":
    main()
