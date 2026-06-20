"""
promptbench/runners/v4_eval.py
==============================
Evaluasi pipeline REDESIGN v4 sebagai POHON FAN-OUT, membandingkan dua MEDIUM
hand-off antar-agent: KV-cache (latent) vs TEKS.

TOPOLOGI (per medium, per latent_steps, per rep)
------------------------------------------------
  feedback x4     : satu per seed backtest (seeds_v4.SEEDS[0..3]).
  mutation x4     : satu per parent feedback (exploitation).
  crossover x2    : gabung 2 parent feedback (seeds_v4.CROSS_PAIRS).
  directions = 2 mutation (MUT_FORWARD) + 2 crossover  -> 4 arah.
  proposal->design->construct x4 : satu front-end independen per arah; construct
                                   TERMINAL di-decode & diskor.

DUA MEDIUM
----------
  kv   : agen tengah (mutation/crossover/proposal/design) tetap kv_only; KV
         output di-CLONE lalu dioper ke agen berikut (crossover = kv_concat 2
         parent). USER prompt cuma bilang "sudah ada di latent memory".
         Hanya feedback (entry, kv_and_text) & construct (terminal) yang decode.
  text : SETIAP agen di-decode (kv_and_text); KV TIDAK dioper (transfer none).
         Output teks agen sebelumnya disuntik ke USER prompt agen berikut
         (var: target_text / parents_text / direction / hypothesis_text /
         prior_factors). Persis filosofi c4_hybrid, tapi utk loop evolusi penuh.

feedback IDENTIK di kedua medium (titik masuk; backtest+faktor selalu teks).
Jadi perbedaan skor murni efek medium pada hop mutation/crossover->...->construct.

JALANKAN
--------
  # dry-run (tanpa GPU): render tiap node + cek wiring KV + var lengkap
  python -m try.promptbench.runners.v4_eval --media kv,text --latent-steps 20 \
      --reps 1 --dry-run

  # runpod (GPU): eksekusi nyata + skor terminal
  python -m try.promptbench.runners.v4_eval --media kv,text \
      --latent-steps 20 --reps 3 --workers 1

Hasil: results/v4_eval/<medium>/ls<N>/rep<R>/<NN_node>.txt + run.json,
       results/v4_eval/scoreboard.{csv,md} (bandingkan kv vs text per node arah).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, Undefined

_THIS = Path(__file__).resolve()
PROMPTBENCH = _THIS.parent.parent
REPO = PROMPTBENCH.parent.parent
V4_YAML = PROMPTBENCH / "variants" / "_authored" / "redesign_v4.yaml"
RESULTS = PROMPTBENCH / "results" / "v4_eval"

DECODE_TEMPERATURE = 0.7
DEFAULT_MAX_NEW = 30000
MARKET_CONTEXT = "Liquid equities, daily bars, 2018-2021 train segment."

# Dua mutation yang diteruskan jadi arah front-end (default: dua parent
# ber-RankIC standalone tertinggi = overfit_meanrev(1) & momentum_volume(2)).
MUT_FORWARD_DEFAULT = (1, 2)


class _Vis(Undefined):
    def __str__(self):  # var hilang → token jelas, terdeteksi di dry-run
        return f"[[MISSING:{self._undefined_name}]]"


# ════════════════════════════════════════════════════════════════════════════
# graph: daftar node + dependensi (medium-agnostic; transfer ditentukan medium)
# ════════════════════════════════════════════════════════════════════════════

def build_nodes(mut_forward: Tuple[int, int]) -> List[dict]:
    """Bangun daftar node berurutan-topologis. Tiap node:
        id, order, agent, stage, parents(list id), seed_idx/pair untuk entry.
    """
    from ..seeds_v4 import SEEDS, CROSS_PAIRS

    nodes: List[dict] = []

    # feedback x len(SEEDS)
    for i, _ in enumerate(SEEDS):
        nodes.append({"id": f"feedback_s{i}", "agent": "feedback",
                      "stage": "feedback", "seed_idx": i, "parents": []})

    # mutation x len(SEEDS) — satu per parent feedback
    for i, _ in enumerate(SEEDS):
        nodes.append({"id": f"mutation_p{i}", "agent": "mutation",
                      "stage": "mutation", "parents": [f"feedback_s{i}"],
                      "seed_idx": i})

    # crossover x len(CROSS_PAIRS) — gabung 2 parent feedback
    for j, (a, b) in enumerate(CROSS_PAIRS):
        nodes.append({"id": f"crossover_x{j}", "agent": "crossover",
                      "stage": "crossover",
                      "parents": [f"feedback_s{a}", f"feedback_s{b}"],
                      "pair": (a, b)})

    # arah front-end: 2 mutation (forward) + semua crossover
    directors = [f"mutation_p{mut_forward[0]}", f"mutation_p{mut_forward[1]}"]
    directors += [f"crossover_x{j}" for j in range(len(CROSS_PAIRS))]

    # proposal -> design -> construct per arah
    for k, d in enumerate(directors):
        p, dz, c = f"proposal_d{k}", f"design_d{k}", f"construct_d{k}"
        nodes.append({"id": p, "agent": "proposal", "stage": "proposal",
                      "parents": [d], "director": d})
        nodes.append({"id": dz, "agent": "design", "stage": "design",
                      "parents": [p]})
        nodes.append({"id": c, "agent": "construct", "stage": "construct",
                      "parents": [dz], "terminal": True})

    for o, n in enumerate(nodes):
        n["order"] = o
    return nodes


# ════════════════════════════════════════════════════════════════════════════
# rendering vars per node (text-handoff menyuntik teks upstream)
# ════════════════════════════════════════════════════════════════════════════

def node_vars(node: dict, medium: str, text_by_id: Dict[str, str]) -> Dict[str, str]:
    """Var Jinja untuk satu node, sesuai medium. Pada medium 'text' var upstream
    diisi dari `text_by_id` (output decode parent); pada 'kv' tidak perlu."""
    from ..seeds_v4 import SEEDS, feedback_vars, parent_text, parents_text

    v: Dict[str, str] = {"handoff": medium, "market_context": MARKET_CONTEXT}
    stage = node["stage"]

    if stage == "feedback":
        v.update(feedback_vars(SEEDS[node["seed_idx"]]))
        return v

    if medium != "text":
        return v  # kv: upstream lewat KV, tak ada var teks

    # ── medium text: suntik teks upstream ──
    if stage == "mutation":
        # entry feedback → parent ringkas: pakai teks feedback hasil decode bila
        # ada, kalau dry-run/awal pakai parent_text seed (selalu tersedia).
        fb = text_by_id.get(node["parents"][0]) or parent_text(SEEDS[node["seed_idx"]])
        v["target_text"] = fb
    elif stage == "crossover":
        a, b = node["pair"]
        ta = text_by_id.get(f"feedback_s{a}")
        tb = text_by_id.get(f"feedback_s{b}")
        if ta and tb:
            v["parents_text"] = ta + "\n---\n" + tb
        else:
            v["parents_text"] = parents_text(node["pair"])
    elif stage == "proposal":
        v["direction"] = text_by_id.get(node["director"], "")
    elif stage == "design":
        v["hypothesis_text"] = text_by_id.get(node["parents"][0], "")
    elif stage == "construct":
        v["prior_factors"] = text_by_id.get(node["parents"][0], "")
    return v


# ════════════════════════════════════════════════════════════════════════════
# dry-run: render semua node, cek var lengkap + ringkas wiring
# ════════════════════════════════════════════════════════════════════════════

def _load_spec(agent: str) -> dict:
    import yaml
    raw = yaml.safe_load(V4_YAML.read_text(encoding="utf-8"))
    return (raw.get("agents") or {})[agent]


def _render(spec: dict, fixtures: dict) -> Tuple[str, str]:
    env = Environment(undefined=_Vis)
    system = env.from_string(spec.get("system", "")).render(**fixtures).strip()
    user = env.from_string(spec.get("user", "")).render(**fixtures).strip()
    return system, user


def _transfer_for(node: dict, medium: str) -> str:
    if medium == "text":
        return "none"
    if not node["parents"]:
        return "none"
    return "concat" if len(node["parents"]) > 1 else "chain"


def run_medium(medium: str, ls: int, rep: int, nodes: List[dict], *,
               dry_run: bool, temp: float, mut_forward: Tuple[int, int]) -> dict:
    """Jalankan seluruh pohon untuk satu (medium × ls × rep)."""
    from ..artifacts import write_step_artifact, write_json

    run_dir = RESULTS / medium / f"ls{ls}" / f"rep{rep}"
    out: Dict[str, Any] = {"medium": medium, "latent_steps": ls, "rep": rep,
                           "ok": True, "nodes": [], "err": None}
    text_by_id: Dict[str, str] = {}

    # ── DRY-RUN ──────────────────────────────────────────────────────────────
    if dry_run:
        all_ok = True
        for n in nodes:
            spec = _load_spec(n["agent"])
            vars_ = node_vars(n, medium, text_by_id)
            system, user = _render(spec, vars_)
            missing = "[[MISSING:" in (system + user)
            all_ok = all_ok and not missing
            transfer = _transfer_for(n, medium)
            decode = (medium == "text") or n["stage"] in ("feedback", "construct")
            rec = {"id": n["id"], "order": n["order"], "agent": n["agent"],
                   "transfer": transfer, "decode": decode,
                   "missing_vars": missing, "parents": n["parents"]}
            out["nodes"].append(rec)
            write_step_artifact(
                run_dir / f"{n['order']:02d}_{n['id']}.txt",
                header={"medium": medium, "ls": ls, "rep": rep, "node": n["id"],
                        "agent": n["agent"], "transfer": transfer,
                        "decode": decode, "dry_run": True},
                system=system, user=user, response="(dry_run — tidak call LLM)",
                kv_report=rec,
            )
            # dry-run: isi text_by_id dgn placeholder agar downstream text render
            if decode:
                text_by_id[n["id"]] = f"(dry_run decoded output of {n['id']})"
        out["ok"] = all_ok
        write_json(run_dir / "run.json", out)
        return out

    # ── REAL RUN (GPU) ───────────────────────────────────────────────────────
    try:
        sys.path.insert(0, str(REPO / "backend"))
        from ...common import get_backend, get_latent_backend
        from latent_mas.agent import load_agent
        from latent_mas.kv_ops import kv_deepcopy, kv_concat, kv_seq_len
        from ..scoring import score_chain

        use_latent = ls > 0 and medium == "kv"
        backend = get_latent_backend(latent_steps_init=max(ls, 10)) if use_latent \
            else get_backend()

        kv_by_id: Dict[str, Any] = {}
        t0 = time.time()

        for n in nodes:
            transfer = _transfer_for(n, medium)
            # ── transfer KV (kv medium saja) ──
            if medium == "kv" and transfer == "chain":
                input_kv = kv_deepcopy(kv_by_id.get(n["parents"][0]))
            elif medium == "kv" and transfer == "concat":
                srcs = [kv_deepcopy(kv_by_id[p]) for p in n["parents"]
                        if p in kv_by_id]
                input_kv = kv_concat(srcs) if srcs else None
            else:
                input_kv = None

            decode = (medium == "text") or n["stage"] in ("feedback", "construct")

            ag = load_agent(n["agent"], backend, strict_vars=False, path=V4_YAML)
            ag.spec.latent_steps = ls if medium == "kv" else 0
            if decode and ag.spec.mode == "kv_only":
                ag.spec.mode = "kv_and_text"
            if ag.spec.mode == "kv_and_text":
                if ag.spec.temperature is None:
                    ag.spec.temperature = temp
                if ag.spec.max_new_tokens is None:
                    ag.spec.max_new_tokens = DEFAULT_MAX_NEW

            vars_ = node_vars(n, medium, text_by_id)
            with open(os.devnull, "w") as dn, redirect_stdout(dn):
                res = ag.run(past_kv=input_kv, **vars_)

            if medium == "kv":
                kv_by_id[n["id"]] = res.kv_cache
            if res.text is not None:
                text_by_id[n["id"]] = res.text

            score_detail = None
            if n.get("terminal"):
                with open(os.devnull, "w") as dn, redirect_stdout(dn):
                    score_detail = score_chain.score_terminal("construct", res.text or "")

            spec = _load_spec(n["agent"])
            system, user = _render(spec, vars_)
            write_step_artifact(
                run_dir / f"{n['order']:02d}_{n['id']}.txt",
                header={"medium": medium, "ls": ls, "rep": rep, "node": n["id"],
                        "agent": n["agent"], "transfer": transfer,
                        "decode": decode, "terminal": bool(n.get("terminal"))},
                system=system, user=user,
                response=res.text if res.text is not None else "(kv_only — tidak di-decode)",
                kv_report={"transfer": transfer,
                           "kv_seq_len": int(getattr(res, "kv_seq_len", 0) or 0)},
                score_detail=score_detail,
            )
            out["nodes"].append({
                "id": n["id"], "agent": n["agent"], "transfer": transfer,
                "terminal": bool(n.get("terminal")),
                "score": float((score_detail or {}).get("score", 0.0)) if n.get("terminal") else None,
                "parser_ok": (score_detail or {}).get("parser_ok") if n.get("terminal") else None,
            })

        out["elapsed_s"] = round(time.time() - t0, 2)
        kv_by_id.clear()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["err"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-1500:]

    write_json(run_dir / "run.json", out)
    return out


# ════════════════════════════════════════════════════════════════════════════
# agregasi: bandingkan kv vs text per node terminal (construct arah)
# ════════════════════════════════════════════════════════════════════════════

def aggregate(results: List[dict]) -> Path:
    import statistics
    RESULTS.mkdir(parents=True, exist_ok=True)

    # kumpulkan skor terminal per (medium, ls, construct-node)
    groups: Dict[tuple, List[float]] = {}
    for r in results:
        for nd in r.get("nodes", []):
            if nd.get("terminal") and nd.get("score") is not None:
                key = (r["medium"], r["latent_steps"], nd["id"])
                groups.setdefault(key, []).append(nd["score"])

    rows = []
    for (medium, ls, node), scores in sorted(groups.items()):
        rows.append({
            "medium": medium, "latent_steps": ls, "node": node, "n": len(scores),
            "score_mean": round(statistics.fmean(scores), 3),
            "score_std": round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
        })

    cols = ["medium", "latent_steps", "node", "n", "score_mean", "score_std"]
    csv_path = RESULTS / "scoreboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = ["# v4_eval scoreboard — KV vs TEXT hand-off\n",
          f"_generated {time.strftime('%Y-%m-%d %H:%M')}_\n",
          "| medium | ls | node | n | score | std |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['medium']} | {r['latent_steps']} | {r['node']} | "
                  f"{r['n']} | {r['score_mean']} | {r['score_std']} |")
    (RESULTS / "scoreboard.md").write_text("\n".join(md), encoding="utf-8")
    return csv_path


# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", default="kv,text", help="kv,text")
    ap.add_argument("--latent-steps", default="20", help="mis. 0,20,40 (kv saja yg pakai)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mut-forward", default=",".join(map(str, MUT_FORWARD_DEFAULT)),
                    help="indeks 2 mutation yg diteruskan jadi arah, mis. 1,2")
    ap.add_argument("--temp", type=float, default=DECODE_TEMPERATURE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    media = [m.strip() for m in args.media.split(",") if m.strip()]
    lsteps = [int(x) for x in args.latent_steps.split(",") if x.strip() != ""]
    mf = tuple(int(x) for x in args.mut_forward.split(","))[:2]
    nodes = build_nodes(mf)

    print(f"[v4_eval] media={media} latent_steps={lsteps} reps={args.reps} "
          f"mut_forward={mf} dry_run={args.dry_run}")
    print(f"[v4_eval] nodes/tree={len(nodes)} "
          f"(feedback+mutation+crossover+4x(proposal,design,construct))")

    results = []
    for medium in media:
        # text medium tak terpengaruh latent_steps utk hand-off; tetap pakai ls
        # untuk parameter realign internal bila >0. Looping ls apa adanya.
        for ls in lsteps:
            for rep in range(args.reps):
                r = run_medium(medium, ls, rep, nodes, dry_run=args.dry_run,
                               temp=args.temp, mut_forward=mf)
                tag = "OK" if r["ok"] else f"FAIL({r.get('err')})"
                print(f"  [{medium} ls{ls} rep{rep}] {tag} "
                      f"nodes={len(r.get('nodes', []))}")
                results.append(r)

    csv_path = aggregate(results)
    n_ok = sum(int(r["ok"]) for r in results)
    print(f"[v4_eval] done: {len(results)} runs, ok={n_ok}")
    print(f"[v4_eval] scoreboard → {csv_path}")


if __name__ == "__main__":
    main()
