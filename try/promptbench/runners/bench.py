"""
promptbench/runners/bench.py
============================
Phase 0e — driver benchmark per-agent (Phase A).

Grid  = varian_prompt × latent_steps{0,10,20,40,60,80} × R repetisi
Axis thinking DIBUANG (latent alignment > token; lihat GUIDE §2 #6). Decode temp 0.7.

Paralelisme: ProcessPoolExecutor (spawn). Job dipisah 2 fase by backend type —
  fase TEXT  (latent_steps==0) → get_backend()         (~8 GB/proses)
  fase LATENT(latent_steps>0)  → get_latent_backend()  (~12-14 GB/proses, use_realign)
Tiap worker = 1 model (singleton per proses). Default 3 worker (48 GB GPU).

Agent kv_only (proposal/construct/consistency) di-FORCE ke kv_and_text supaya
output bisa di-decode & diskor (latent_steps tetap aktif; cuma menambah tahap decode).

Jalankan:
  # dry-run (tanpa GPU): render prompt + skor placeholder, untuk validasi pipa
  python -m try.promptbench.runners.bench --agents proposal,construct,judger \
      --latent-steps 0,10 --reps 2 --dry-run

  # runpod (GPU):
  python -m try.promptbench.runners.bench --agents proposal,construct,consistency,judger \
      --latent-steps 0,10,20,40,60,80 --reps 5 --workers 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, Undefined

_THIS = Path(__file__).resolve()
PROMPTBENCH = _THIS.parent.parent
REPO = PROMPTBENCH.parent.parent
VARIANTS_DIR = PROMPTBENCH / "variants"
RESULTS_DIR = PROMPTBENCH / "results" / "phaseA"

CORE = ["proposal", "construct", "consistency", "judger"]
KV_ONLY = {"proposal", "construct", "consistency"}   # di-force kv_and_text utk skor
DECODE_TEMPERATURE = 0.7
DEFAULT_MAX_NEW = 1536


class _Vis(Undefined):
    def __str__(self): return f"[[MISSING:{self._undefined_name}]]"


# ════════════════════════════════════════════════════════════════════════════
# enumerasi job
# ════════════════════════════════════════════════════════════════════════════

def load_manifest() -> dict:
    p = VARIANTS_DIR / "variants_manifest.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def enumerate_jobs(agents: List[str], latent_steps: List[int], reps: int,
                   variant_filter: Optional[str]) -> List[dict]:
    man = load_manifest()
    jobs = []
    for agent in agents:
        for v in man.get(agent, []):
            if variant_filter and variant_filter not in v["variant_id"]:
                continue
            for ls in latent_steps:
                for r in range(reps):
                    jobs.append({
                        "agent": agent,
                        "variant_id": v["variant_id"],
                        "variant_path": str(PROMPTBENCH / v["path"]),
                        "latent_steps": ls,
                        "rep": r,
                    })
    return jobs


# ════════════════════════════════════════════════════════════════════════════
# worker
# ════════════════════════════════════════════════════════════════════════════

def _load_spec(path: str, agent: str) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return (raw.get("agents") or {})[agent]


def _render(spec: dict, fixtures: dict) -> tuple[str, str]:
    env = Environment(undefined=_Vis)
    system = env.from_string(spec.get("system", "")).render(**fixtures).strip()
    user = env.from_string(spec.get("user", "")).render(**fixtures).strip()
    return system, user


def run_job(job: dict, *, dry_run: bool, temp: float) -> dict:
    """Dijalankan di worker. Lazy-import backend & scorer agar dry-run bebas GPU."""
    agent = job["agent"]
    ls = job["latent_steps"]
    spec = _load_spec(job["variant_path"], agent)

    # import fixtures + scorer (torch-free). Relatif: 'try' adalah keyword,
    # `from try.` ilegal di statement import — pakai relative import.
    from ..fixtures_pb import FIXTURES
    from ..scoring import score as scoremod

    system, user = _render(spec, FIXTURES)
    out = {**job, "ok": False, "text": None, "score": 0.0, "err": None,
           "kv_tokens": 0, "elapsed_s": 0.0}

    if dry_run:
        # skor placeholder: render sukses → 1.0, gagal var → tandai
        out["text"] = "(dry_run — tidak call LLM)"
        out["ok"] = "[[MISSING:" not in (system + user)
        out["score"] = 1.0 if out["ok"] else 0.0
        out["missing_vars"] = "[[MISSING:" in (system + user)
        _save_artifact(out, system, user, score_detail={"dry_run": True})
        return out

    try:
        sys.path.insert(0, str(REPO / "backend"))
        from ...common import get_backend, get_latent_backend  # try.common
        from latent_mas.agent import load_agent

        backend = get_backend() if ls == 0 else get_latent_backend(latent_steps_init=max(ls, 10))
        ag = load_agent(agent, backend, strict_vars=False, path=Path(job["variant_path"]))
        # force kv_only → kv_and_text supaya bisa di-decode & diskor
        if ag.spec.mode == "kv_only":
            ag.spec.mode = "kv_and_text"
            if ag.spec.temperature is None:
                ag.spec.temperature = temp
            if ag.spec.max_new_tokens is None:
                ag.spec.max_new_tokens = DEFAULT_MAX_NEW
        ag.spec.latent_steps = ls

        t0 = time.time()
        res = ag.run(**FIXTURES)
        out["elapsed_s"] = round(time.time() - t0, 2)
        out["text"] = res.text
        out["kv_tokens"] = getattr(res, "kv_seq_len", 0)

        with open(os.devnull, "w") as dn, redirect_stdout(dn):
            detail = scoremod.score_output(agent, res.text or "")
        out["score"] = detail.get("score", 0.0)
        out["ok"] = bool(res.text and res.text.strip())
        out["score_detail"] = detail
        _save_artifact(out, system, user, score_detail=detail)
    except Exception as e:  # noqa: BLE001
        out["err"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-1200:]
    return out


def _save_artifact(out: dict, system: str, user: str, score_detail: dict) -> None:
    # Layout nested & mudah dibaca: phaseA/<agent>/<variant_short>/ls<N>/rep<R>.txt
    from ..artifacts import phaseA_artifact, write_step_artifact
    fp = phaseA_artifact(out["agent"], out["variant_id"], out["latent_steps"], out["rep"])
    write_step_artifact(
        fp,
        header={"agent": out["agent"], "variant": out["variant_id"],
                "latent_steps": out["latent_steps"], "rep": out["rep"],
                "score": out.get("score"), "ok": out.get("ok"),
                "elapsed_s": out.get("elapsed_s"), "kv_tokens": out.get("kv_tokens")},
        system=system, user=user, response=out.get("text") or "",
        score_detail=score_detail,
        error=(out["err"] + "\n" + out.get("traceback", "")) if out.get("err") else None,
    )


# ════════════════════════════════════════════════════════════════════════════
# orkestrasi + agregasi
# ════════════════════════════════════════════════════════════════════════════

def _run_phase(jobs: List[dict], workers: int, dry_run: bool, temp: float) -> List[dict]:
    if not jobs:
        return []
    if workers <= 1 or dry_run:
        return [run_job(j, dry_run=dry_run, temp=temp) for j in jobs]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_job, j, dry_run=dry_run, temp=temp): j for j in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


def aggregate_and_write(results: List[dict]) -> Path:
    from ..scoring import score as scoremod
    # group by (agent, variant_id, latent_steps)
    groups: Dict[tuple, List[dict]] = {}
    for r in results:
        groups.setdefault((r["agent"], r["variant_id"], r["latent_steps"]), []).append(r)

    new_rows: Dict[tuple, dict] = {}
    for (agent, variant, ls), items in sorted(groups.items()):
        details = [it.get("score_detail", {"score": it.get("score", 0.0)}) for it in items]
        agg = scoremod.aggregate(agent, details)
        ok_rate = round(sum(int(it.get("ok", False)) for it in items) / len(items), 3)
        new_rows[(agent, variant, ls)] = {
            "agent": agent, "variant_id": variant, "latent_steps": ls,
            "n": len(items), "ok_rate": ok_rate,
            **{k: v for k, v in agg.items() if k not in ("role", "n_rep", "families_union")},
        }

    # Merge: baca scoreboard lama, update/tambah baris baru saja
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "scoreboard.csv"
    existing: Dict[tuple, dict] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["agent"], row["variant_id"], int(row["latent_steps"]))
                existing[key] = row
    existing.update(new_rows)  # new_rows menimpa baris lama yang sama

    rows = sorted(existing.values(),
                  key=lambda x: (x["agent"], -float(x.get("score_mean") or 0), int(x["latent_steps"])))
    cols = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["agent", "variant_id", "latent_steps", "n",
                                          "ok_rate", "score_mean", "score_std"]
                                         + [c for c in cols if c not in
                                            {"agent", "variant_id", "latent_steps", "n",
                                             "ok_rate", "score_mean", "score_std"}])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # markdown ringkas: best variant per (agent, latent_steps)
    md = ["# Phase A scoreboard\n", f"_generated {time.strftime('%Y-%m-%d %H:%M')}_\n"]
    cur = None
    for r in rows:
        if r["agent"] != cur:
            cur = r["agent"]
            md.append(f"\n## {cur}\n")
            md.append("| variant | latent_steps | n | ok | score_mean | score_std | extra |")
            md.append("|---|---|---|---|---|---|---|")
        extra = {k: r[k] for k in r if k not in
                 {"agent", "variant_id", "latent_steps", "n", "ok_rate", "score_mean", "score_std"}}
        md.append(f"| {r['variant_id']} | {r['latent_steps']} | {r['n']} | {r['ok_rate']} | "
                  f"{r.get('score_mean')} | {r.get('score_std')} | {extra} |")
    (RESULTS_DIR / "scoreboard.md").write_text("\n".join(md), encoding="utf-8")
    return csv_path


# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=",".join(CORE))
    ap.add_argument("--latent-steps", default="0,10,20,40,60,80")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--variant", default=None, help="substring filter pada variant_id")
    ap.add_argument("--temp", type=float, default=DECODE_TEMPERATURE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    lsteps = [int(x) for x in args.latent_steps.split(",") if x.strip() != ""]
    jobs = enumerate_jobs(agents, lsteps, args.reps, args.variant)
    text_jobs = [j for j in jobs if j["latent_steps"] == 0]
    lat_jobs = [j for j in jobs if j["latent_steps"] > 0]

    print(f"[bench] agents={agents} latent_steps={lsteps} reps={args.reps} "
          f"workers={args.workers} dry_run={args.dry_run}")
    print(f"[bench] total jobs={len(jobs)} (text={len(text_jobs)} latent={len(lat_jobs)})")

    results = []
    if text_jobs:
        print(f"[bench] === TEXT phase (latent_steps=0): {len(text_jobs)} jobs ===")
        results += _run_phase(text_jobs, args.workers, args.dry_run, args.temp)
    if lat_jobs:
        print(f"[bench] === LATENT phase (use_realign): {len(lat_jobs)} jobs ===")
        results += _run_phase(lat_jobs, args.workers, args.dry_run, args.temp)

    n_ok = sum(int(r.get("ok", False)) for r in results)
    n_err = sum(int(bool(r.get("err"))) for r in results)
    print(f"[bench] done: {len(results)} results, ok={n_ok}, err={n_err}")
    csv_path = aggregate_and_write(results)
    print(f"[bench] scoreboard → {csv_path}")
    print(f"[bench] artifacts  → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
