"""
promptbench/runners/bench_chain_grid.py
=======================================
Phase B grid sweep — Cartesian product dari varian per agent, dijalankan via
subprocess `bench_chain` per combo.

Setiap combo mendapat subfolder unik melalui env var PHASE_B_OVERRIDE:
  results/phaseB_grid/<combo_key>/<chain>/ls<N>/rep<R>/

Setelah semua combo selesai, scoreboard master diagregasi dari semua combo:
  results/phaseB_grid/scoreboard_master.{csv,md}

Jalankan (dari root project):
  source /workspace/runpod_env.sh && \\
  source .venv/bin/activate && \\
  python -m try.promptbench.runners.bench_chain_grid \\
      --chains pc_2agent,mut_proposal,cross_proposal,pcj_judger,front_end_full,front_end_feedback \\
      --latent-steps 60 --reps 3 --workers 3
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS = Path(__file__).resolve()
RUNNERS = _THIS.parent
PROMPTBENCH = RUNNERS.parent
REPO = PROMPTBENCH.parent.parent
PYTHON = str(REPO / ".venv" / "bin" / "python")

PHASE_B_GRID = PROMPTBENCH / "results" / "phaseB_grid"

# ── Grid: agent → [variant1, variant2, ...] ────────────────────────────────
GRID: Dict[str, List[str]] = {
    "proposal":    ["git_judger_only", "git_latentmas_foundation", "working"],
    "construct":   ["git_optimalisasi", "git_judger_only", "working"],
    "judger":      ["git_judger_retry_fix", "git_judger_only", "git_optimalisasi"],
    "consistency": ["git_latentmas_foundation", "git_judger_only", "authored_claude_latentpaper"],
}

# Agen yang tidak ada di GRID menggunakan winner scoreboard Phase A
AGENTS_SORTED = sorted(GRID.keys())   # agar urutan combo_key konsisten


# ════════════════════════════════════════════════════════════════════════════
# Combo key & enumerasi
# ════════════════════════════════════════════════════════════════════════════

# Prefix pendek per agent supaya path tidak terlalu panjang
_PREFIX = {"proposal": "p", "construct": "c", "consistency": "co", "judger": "j"}


def combo_key(picks: Dict[str, str]) -> str:
    """Misal: p-working__c-git_optimalisasi__co-git_latentmas_foundation__j-git_judger_retry_fix"""
    return "__".join(f"{_PREFIX.get(a, a)}-{v}" for a, v in sorted(picks.items()))


def all_combos() -> List[Dict[str, str]]:
    combos = []
    for values in itertools.product(*[GRID[a] for a in AGENTS_SORTED]):
        combos.append(dict(zip(AGENTS_SORTED, values)))
    return combos


# ════════════════════════════════════════════════════════════════════════════
# Jalankan satu combo via subprocess
# ════════════════════════════════════════════════════════════════════════════

def run_combo(
    combo: Dict[str, str],
    *,
    chains: str,
    latent_steps: str,
    reps: int,
    workers: int,
    temp: float,
    dry_run: bool,
) -> dict:
    key = combo_key(combo)
    output_dir = PHASE_B_GRID / key
    output_dir.mkdir(parents=True, exist_ok=True)

    pick_str = ",".join(f"{a}={v}" for a, v in sorted(combo.items()))

    cmd = [
        PYTHON, "-m", "try.promptbench.runners.bench_chain",
        "--chains", chains,
        "--latent-steps", latent_steps,
        "--reps", str(reps),
        "--workers", str(workers),
        "--temp", str(temp),
        "--pick", pick_str,
    ]
    if dry_run:
        cmd.append("--dry-run")

    env = {**os.environ, "PHASE_B_OVERRIDE": str(output_dir)}

    log_path = output_dir / "run.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# combo: {key}\n# cmd: {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )

    elapsed = round(time.time() - t0, 1)
    ok = proc.returncode == 0
    scoreboard_csv = output_dir / "scoreboard.csv"

    return {
        "combo_key": key,
        "picks": combo,
        "returncode": proc.returncode,
        "ok": ok,
        "elapsed_s": elapsed,
        "scoreboard_csv": str(scoreboard_csv) if scoreboard_csv.exists() else None,
    }


# ════════════════════════════════════════════════════════════════════════════
# Agregasi master scoreboard dari semua combo
# ════════════════════════════════════════════════════════════════════════════

def aggregate_master(combo_results: List[dict]) -> Path:
    rows = []
    for cr in combo_results:
        if not cr.get("scoreboard_csv"):
            continue
        p = Path(cr["scoreboard_csv"])
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["combo_key"] = cr["combo_key"]
                for agent, variant in cr["picks"].items():
                    row[f"pick_{agent}"] = variant
                rows.append(row)

    PHASE_B_GRID.mkdir(parents=True, exist_ok=True)
    if not rows:
        return PHASE_B_GRID / "scoreboard_master.csv"

    # urutan kolom: combo_key, pick_*, lalu kolom bench_chain asli
    pick_cols = [f"pick_{a}" for a in sorted(GRID.keys())]
    base_cols = ["chain", "config", "terminal_agent", "n",
                 "score_mean", "score_std", "ok_rate", "healthy_rate",
                 "kv_flag_rate", "text_collapse_rate", "err_rate"]
    all_cols = ["combo_key"] + pick_cols + base_cols

    csv_path = PHASE_B_GRID / "scoreboard_master.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # markdown ringkas: top-20 per score_mean
    rows_sorted = sorted(rows,
                         key=lambda r: float(r.get("score_mean") or 0.0),
                         reverse=True)
    md = [
        "# Phase B Grid Scoreboard (master)\n",
        f"_generated {time.strftime('%Y-%m-%d %H:%M')}_\n",
        f"Total baris: {len(rows)} ({len(combo_results)} combo × 6 chain)\n",
        "| combo_key | chain | terminal | score | std | ok | healthy | kv_flag | txt_collapse |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_sorted[:50]:
        md.append(
            f"| {r['combo_key']} | {r['chain']} | {r.get('terminal_agent','?')} | "
            f"{r['score_mean']} | {r['score_std']} | {r['ok_rate']} | "
            f"{r['healthy_rate']} | {r['kv_flag_rate']} | {r['text_collapse_rate']} |"
        )
    (PHASE_B_GRID / "scoreboard_master.md").write_text("\n".join(md), encoding="utf-8")
    return csv_path


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint: simpan progress agar bisa dilanjutkan jika terputus
# ════════════════════════════════════════════════════════════════════════════

def _load_checkpoint() -> set:
    """Set combo_key yang sudah selesai."""
    cp = PHASE_B_GRID / "checkpoint.txt"
    if not cp.exists():
        return set()
    return {line.strip() for line in cp.read_text().splitlines() if line.strip()}


def _save_checkpoint(done: set) -> None:
    PHASE_B_GRID.mkdir(parents=True, exist_ok=True)
    (PHASE_B_GRID / "checkpoint.txt").write_text("\n".join(sorted(done)), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains",
                    default="pc_2agent,mut_proposal,cross_proposal,pcj_judger,front_end_full,front_end_feedback")
    ap.add_argument("--latent-steps", default="60")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3,
                    help="paralel workers di dalam tiap bench_chain subprocess")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="lewati combo yang sudah ada di checkpoint")
    args = ap.parse_args()

    combos = all_combos()
    n_total = len(combos)
    done_keys = _load_checkpoint() if args.resume else set()

    print(f"[grid] total combos={n_total}  chains={args.chains}")
    print(f"[grid] latent_steps={args.latent_steps}  reps={args.reps}  workers={args.workers}")
    print(f"[grid] output base: {PHASE_B_GRID}")
    if done_keys:
        print(f"[grid] resume: {len(done_keys)} combo sudah done, skip.")

    combo_results: List[dict] = []
    t_global = time.time()

    for idx, combo in enumerate(combos, 1):
        key = combo_key(combo)
        if key in done_keys:
            print(f"[grid] [{idx:02d}/{n_total}] SKIP (done): {key}")
            # tetap muat ke combo_results untuk agregasi
            scoreboard_csv = PHASE_B_GRID / key / "scoreboard.csv"
            combo_results.append({
                "combo_key": key, "picks": combo, "ok": True,
                "scoreboard_csv": str(scoreboard_csv) if scoreboard_csv.exists() else None,
            })
            continue

        print(f"\n[grid] [{idx:02d}/{n_total}] START: {key}")
        t0 = time.time()
        cr = run_combo(
            combo,
            chains=args.chains,
            latent_steps=args.latent_steps,
            reps=args.reps,
            workers=args.workers,
            temp=args.temp,
            dry_run=args.dry_run,
        )
        elapsed = round(time.time() - t0, 1)
        status = "OK" if cr["ok"] else f"FAIL(rc={cr['returncode']})"
        print(f"[grid] [{idx:02d}/{n_total}] {status} {key}  ({elapsed}s)")

        combo_results.append(cr)
        done_keys.add(key)
        _save_checkpoint(done_keys)

        # estimasi sisa
        elapsed_total = time.time() - t_global
        avg = elapsed_total / idx
        remaining = avg * (n_total - idx)
        print(f"[grid] progress {idx}/{n_total}  avg={avg:.0f}s/combo  "
              f"eta={remaining/60:.1f}min")

    print(f"\n[grid] Semua combo selesai. Agregasi master scoreboard ...")
    master_csv = aggregate_master(combo_results)
    print(f"[grid] scoreboard master → {master_csv}")

    n_ok = sum(1 for r in combo_results if r.get("ok"))
    n_fail = n_total - n_ok
    total_elapsed = time.time() - t_global
    print(f"[grid] done: {n_ok} ok, {n_fail} fail  total={total_elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
