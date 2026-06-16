"""
promptbench/chain/chain.py
==========================
Phase B — rantai multi-agent via KV-cache, BERTAHAP. Tujuan: lihat apakah
akumulasi KV lintas-agent menurunkan kualitas output (terutama ekspresi judger)
relatif ke skor isolasi Phase A. Bila ya → bug ada di chaining/KV, bukan prompt.

Disiplin KV identik `backend/latent_mas/pipeline.py`:
  seed → proposal(kv_only) → construct(kv_only) → consistency(kv_only) = kv_consist
  judger   ← deepcopy(kv_consist)   (decode hipotesis+ekspresi)
  feedback ← deepcopy(kv_consist)   (BUKAN kv_judger → anti-bias)
Evolution: mutation/crossover (kv_only, seed=None, parent=TEKS) → guidance_kv →
  seed front-end.

Tahap (prefix yang makin panjang; TIP selalu di-decode & diskor):
  s1_pc      proposal → construct[tip]
  s2_pcj     proposal → construct → judger[tip]              (consistency dilewati)
  s3_pccj    proposal → construct → consistency → judger[tip] (front-end penuh)
  s4_full_fb front-end penuh → judger → feedback[tip]         (backtest rekayasa)
  s5_mut     mutation(guidance) → front-end penuh → judger[tip]
  s6_cross   crossover(guidance) → front-end penuh → judger[tip]

Tiap batas agent → kv_shape_report + Boundary (untuk detektor collapse).
Tiap tip → parsing_hook (audit fallback) + scorer Phase A + collapse.detect.

Pemilihan varian default: top per-agent dari results/phaseA/scoreboard.csv
(score_mean tertinggi → variant_id + latent_steps). Override via CLI.

Jalankan (GPU):
  python -m try.promptbench.chain.chain --stages s1_pc,s2_pcj,s3_pccj,s4_full_fb --reps 3
  python -m try.promptbench.chain.chain --stages all --reps 3 \
      --pick judger=gate_deterministik --ls judger=20 --uniform-ls 20
  python -m try.promptbench.chain.chain --dry-run   # tanpa GPU: cek wiring & picks
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
import yaml

_THIS = Path(__file__).resolve()
PROMPTBENCH = _THIS.parent.parent
REPO = PROMPTBENCH.parent.parent
VARIANTS_DIR = PROMPTBENCH / "variants"
SCOREBOARD_CSV = PROMPTBENCH / "results" / "phaseA" / "scoreboard.csv"
RESULTS_DIR = PROMPTBENCH / "results" / "phaseB"

DECODE_TEMPERATURE = 0.7
DEFAULT_MAX_NEW = 512
DEFAULT_LS = 20            # latent_steps default utk agent tanpa pick Phase A

FRONTEND = ["proposal", "construct", "consistency"]

# fallback bila scoreboard.csv belum ada (mis. dry-run di mesin tanpa Phase A).
# Diambil dari scoreboard 2026-06-16 (baris teratas tiap agent).
HARDCODED_PICKS: Dict[str, Tuple[str, int]] = {
    "proposal":    ("proposal__working__258abdbbccea", 60),
    "construct":   ("construct__working__56396e7d44b0", 0),
    "consistency": ("consistency__authored_claude_latentpaper__88133558d50d", 0),
    "judger":      ("judger__git_gate_deterministik__6ce003675450", 20),
}


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
        # variant_id
        vid: Optional[str] = None
        base_ls = DEFAULT_LS
        if agent in sb:
            vid, base_ls = sb[agent]
        elif agent in HARDCODED_PICKS:
            vid, base_ls = HARDCODED_PICKS[agent]
        if agent in pick_overrides:
            sub = pick_overrides[agent]
            match = _match_variant(agent, sub, man)
            if match:
                vid = match
        if vid is None:
            vid = _manifest_first(agent, man)
        if vid is None:
            continue  # agent tak punya varian → lewati (mis. introspect tak dipakai)

        # latent_steps
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
    # fallback: konvensi path langsung
    return VARIANTS_DIR / agent / f"{vid}.yaml"


# ════════════════════════════════════════════════════════════════════════════
# stage spec
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


# ════════════════════════════════════════════════════════════════════════════
# runner
# ════════════════════════════════════════════════════════════════════════════

def _agent_kwargs(name: str, fx: dict) -> dict:
    """Variabel template per agent (mirror pipeline.py / mining_loop.py)."""
    if name == "proposal":
        return dict(direction=fx["direction"], market_context=fx["market_context"],
                    prior_feedback=fx["prior_feedback"], negative_hint=fx["negative_hint"])
    if name == "construct":
        return dict(diversity_hint=fx["diversity_hint"])
    if name == "consistency":
        return {}
    if name == "judger":
        return dict(direction=fx["direction"], diversity_hint=fx["diversity_hint"])
    if name == "mutation":
        return dict(target_text=fx["target_text"], direction=fx["direction"])
    if name == "crossover":
        return dict(parents_text=fx["parents_text"], n_parents=fx["n_parents"],
                    direction=fx["direction"])
    return {}


def run_stage(stage: str, picks: Dict[str, Dict[str, Any]], rep: int,
              *, temp: float, dry_run: bool) -> dict:
    """Jalankan satu stage satu kali. Mengembalikan dict hasil + artefak ditulis."""
    spec = STAGES[stage]
    out: Dict[str, Any] = {"stage": stage, "rep": rep, "ok": False, "err": None,
                           "picks": {a: picks[a]["variant_id"] for a in picks
                                     if a in _stage_agents(stage)},
                           "boundaries": [], "tip_text": None, "score": 0.0}

    if dry_run:
        out["ok"] = all(a in picks for a in _stage_agents(stage))
        out["note"] = "dry_run — wiring only"
        _save_artifact(out)
        return out

    try:
        from ..fixtures_pb import FIXTURES
        from ..scoring import score as scoremod
        from . import parsing_hook, collapse

        sys.path.insert(0, str(REPO / "backend"))
        from ...common import get_backend, get_latent_backend
        from latent_mas.agent import load_agent
        from latent_mas import kv_ops

        # backend: latent bila ADA agent dengan ls>0 di stage ini
        stage_agents = _stage_agents(stage)
        any_latent = any(picks[a]["latent_steps"] > 0 for a in stage_agents if a in picks)
        max_ls = max((picks[a]["latent_steps"] for a in stage_agents if a in picks), default=0)
        backend = get_latent_backend(latent_steps_init=max(max_ls, 10)) if any_latent else get_backend()

        def _mk(name: str, *, force_decode: bool) -> Any:
            p = picks[name]
            ag = load_agent(name, backend, strict_vars=False, path=Path(p["variant_path"]))
            ag.spec.latent_steps = p["latent_steps"]
            if force_decode and ag.spec.mode == "kv_only":
                ag.spec.mode = "kv_and_text"
            if ag.spec.mode != "kv_only":
                if ag.spec.temperature is None:
                    ag.spec.temperature = temp
                if ag.spec.max_new_tokens is None:
                    ag.spec.max_new_tokens = DEFAULT_MAX_NEW
            return ag

        boundaries: List[collapse.Boundary] = []

        def _record(label: str, res: Any) -> None:
            b = collapse.Boundary(
                label=label, kv_tokens=getattr(res, "kv_seq_len", 0) or 0,
                n_input=getattr(res, "n_input_tokens", 0) or 0,
                n_output=getattr(res, "n_output_tokens", 0) or 0,
                latent_steps=getattr(res, "latent_steps", 0) or 0,
            )
            boundaries.append(b)
            print(f"  [chain {stage}] {label}: kv={b.kv_tokens} in={b.n_input} "
                  f"out={b.n_output} ls={b.latent_steps}")

        t0 = time.time()

        # ── 0. evolution guidance (opsional) ────────────────────────────────
        seed_kv = None
        if spec["evo"]:
            evo = spec["evo"]
            r_g = _mk(evo, force_decode=False).run(past_kv=None, **_agent_kwargs(evo, FIXTURES))
            seed_kv = r_g.kv_cache
            _record(f"{evo}(guidance)", r_g)

        # ── 1. front-end sequential (in-place OK) ───────────────────────────
        front = spec["front"]
        tip = spec["tip"]
        kv = seed_kv
        kv_consist = None
        tip_text, tip_parsed_ok, tip_role = None, False, tip
        parse_trace = None

        for i, name in enumerate(front):
            is_tip = (name == tip)
            ag = _mk(name, force_decode=is_tip)
            r = ag.run(past_kv=kv, **_agent_kwargs(name, FIXTURES))
            _record(name, r)
            kv = r.kv_cache
            if name == "consistency" or (name == "construct" and "consistency" not in front):
                kv_consist = kv  # baseline = output front-end terakhir
            if is_tip:
                tip_text = r.text or ""
                parsed, parse_trace = parsing_hook.parse_with_trace(tip_text, role=name)
                tip_parsed_ok = parsed is not None and bool(getattr(parsed, "expressions", None))

        if kv_consist is None:
            kv_consist = kv

        # ── 2. judger (bila tip butuh judger / feedback) ────────────────────
        hypothesis, expr = "", ""
        if tip in ("judger", "feedback"):
            judger = _mk("judger", force_decode=True)
            r_j = judger.run(past_kv=kv_ops.kv_deepcopy(kv_consist),
                             **_agent_kwargs("judger", FIXTURES))
            _record("judger", r_j)
            j_parsed, j_trace = parsing_hook.parse_with_trace(r_j.text or "", role="judger")
            if tip == "judger":
                tip_text, parse_trace = r_j.text or "", j_trace
                tip_parsed_ok = j_parsed is not None and bool(getattr(j_parsed, "expressions", None))
            if j_parsed is not None:
                hypothesis = j_parsed.hypothesis or ""
                expr = j_parsed.expressions[0] if j_parsed.expressions else ""

        # ── 3. feedback (tip) dengan backtest rekayasa ──────────────────────
        if tip == "feedback":
            factor_block = (f"- factor: {hypothesis or FIXTURES['hypothesis']}\n"
                            f"  Expression: {expr or '(none parsed)'}")
            fb = _mk("feedback", force_decode=True)
            r_f = fb.run(past_kv=kv_ops.kv_deepcopy(kv_consist),
                         hypothesis_text=hypothesis or FIXTURES["hypothesis"],
                         factor_block=factor_block,
                         backtest_results=FIXTURES["backtest_results"],
                         sota_block=FIXTURES["sota_block"])
            _record("feedback", r_f)
            tip_text, tip_role = r_f.text or "", "feedback"
            tip_parsed_ok = True  # feedback dinilai via JSON di scorer, bukan parser ekspresi

        out["elapsed_s"] = round(time.time() - t0, 2)

        # ── skor + collapse ─────────────────────────────────────────────────
        with open(os.devnull, "w") as dn, redirect_stdout(dn):
            score_detail = scoremod.score_output(tip_role, tip_text or "")
        verdict = collapse.detect(
            tip_text or "", boundaries,
            tip_should_parse=tip_role in ("construct", "judger"),
            tip_parsed_ok=tip_parsed_ok,
        )

        out.update({
            "ok": bool(tip_text and tip_text.strip()),
            "tip_role": tip_role,
            "tip_text": tip_text,
            "score": score_detail.get("score", 0.0),
            "score_detail": score_detail,
            "parse_trace": parse_trace,
            "boundaries": [b.to_dict() for b in boundaries],
            "collapse": verdict.to_dict(),
        })
        _save_artifact(out)
    except Exception as e:  # noqa: BLE001
        out["err"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-1500:]
        _save_artifact(out)
    return out


def _stage_agents(stage: str) -> List[str]:
    spec = STAGES[stage]
    ag = list(spec["front"])
    if spec["evo"]:
        ag = [spec["evo"]] + ag
    if spec["tip"] in ("judger", "feedback") and "judger" not in ag:
        ag.append("judger")
    if spec["tip"] == "feedback" and "feedback" not in ag:
        ag.append("feedback")
    return ag


def _save_artifact(out: dict) -> None:
    d = RESULTS_DIR / out["stage"]
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{out['stage']}__rep{out['rep']}.txt"
    lines = [
        f"# stage={out['stage']} rep={out['rep']} ok={out.get('ok')} "
        f"score={out.get('score')} elapsed_s={out.get('elapsed_s')}",
        f"# picks={json.dumps(out.get('picks', {}), ensure_ascii=False)}",
        f"# tip_role={out.get('tip_role')}",
        "", "=" * 78, "KV BOUNDARIES (seed→tip)", "=" * 78,
        json.dumps(out.get("boundaries", []), indent=2, ensure_ascii=False),
        "", "=" * 78, "COLLAPSE VERDICT", "=" * 78,
        json.dumps(out.get("collapse", {}), indent=2, ensure_ascii=False),
        "", "=" * 78, "PARSE TRACE", "=" * 78,
        json.dumps(out.get("parse_trace", {}), indent=2, ensure_ascii=False),
        "", "=" * 78, "TIP RESPONSE", "=" * 78, (out.get("tip_text") or ""),
        "", "=" * 78, "SCORE DETAIL", "=" * 78,
        json.dumps(out.get("score_detail", {}), indent=2, ensure_ascii=False),
    ]
    if out.get("err"):
        lines += ["", "ERROR", out["err"], out.get("traceback", "")]
    fp.write_text("\n".join(lines), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
def _parse_kv_overrides(items: List[str], cast=str) -> dict:
    """'agent=val' → {agent: cast(val)}."""
    out = {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
            out[k.strip()] = cast(v.strip())
    return out


def aggregate_and_write(results: List[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # group by stage
    by_stage: Dict[str, List[dict]] = {}
    for r in results:
        by_stage.setdefault(r["stage"], []).append(r)

    md = ["# Phase B scoreboard (KV-chain)\n",
          f"_generated {time.strftime('%Y-%m-%d %H:%M')}_\n",
          "| stage | n | ok | score_mean | collapse_rate | tip_kv_mean | picks |",
          "|---|---|---|---|---|---|---|"]
    for stage in STAGE_ORDER:
        items = by_stage.get(stage)
        if not items:
            continue
        n = len(items)
        ok = round(sum(int(x.get("ok", False)) for x in items) / n, 2)
        sc = round(sum(x.get("score", 0.0) for x in items) / n, 3)
        col = round(sum(int((x.get("collapse") or {}).get("collapsed", False))
                        for x in items) / n, 2)
        tipkv = [(_b[-1]["kv_tokens"] if (_b := x.get("boundaries")) else 0) for x in items]
        tipkv_mean = round(sum(tipkv) / n, 1)
        picks = items[0].get("picks", {})
        md.append(f"| {stage} | {n} | {ok} | {sc} | {col} | {tipkv_mean} | "
                  f"{json.dumps(picks, ensure_ascii=False)} |")
    p = RESULTS_DIR / "scoreboard.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="s1_pc,s2_pcj,s3_pccj,s4_full_fb",
                    help="comma list atau 'all'")
    ap.add_argument("--reps", type=int, default=3)
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
    print(f"[chain] stages={stages} reps={args.reps} dry_run={args.dry_run}")

    results = []
    for stage in stages:
        for rep in range(args.reps):
            results.append(run_stage(stage, picks, rep,
                                     temp=args.temp, dry_run=args.dry_run))

    n_ok = sum(int(r.get("ok", False)) for r in results)
    n_err = sum(int(bool(r.get("err"))) for r in results)
    n_col = sum(int((r.get("collapse") or {}).get("collapsed", False)) for r in results)
    print(f"[chain] done: {len(results)} runs, ok={n_ok}, err={n_err}, collapse_flags={n_col}")
    sb = aggregate_and_write(results)
    print(f"[chain] scoreboard → {sb}")
    print(f"[chain] artifacts  → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
