#!/usr/bin/env bash
# run_targeted.sh — Phase A construct baru + Phase B (pc_2agent + pcj_judger × 7 judger)
# Phase A proposal dijalankan ulang agar scoreboard entry terupdate.
# Jalankan dari root repo: bash try/promptbench/runners/run_targeted.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="$ROOT/try/promptbench/results/targeted_run_logs"
mkdir -p "$LOG_DIR"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

PYTHON="$ROOT/.venv/bin/python"
BASE_PICK="construct=git_stepwise_final,proposal=authored_mixed_scoped"

JUDGER_VARIANTS=(
  authored_claude_latentpaper
  git_gate_deterministik
  git_judger_only
  git_judger_retry_fix
  git_latentmas_foundation
  git_optimalisasi
  working
)

log "===== Phase A: construct (git_stepwise_final, ls=60, reps=3) ====="
"$PYTHON" -m try.promptbench.runners.bench \
    --agents construct \
    --variant git_stepwise_final \
    --latent-steps 60 \
    --reps 3 \
    --workers 3 \
    2>&1 | tee "$LOG_DIR/phaseA_construct.log"

log "===== Phase A: proposal (authored_mixed_scoped, ls=60, reps=3) ====="
"$PYTHON" -m try.promptbench.runners.bench \
    --agents proposal \
    --variant authored_mixed_scoped \
    --latent-steps 60 \
    --reps 3 \
    --workers 3 \
    2>&1 | tee "$LOG_DIR/phaseA_proposal.log"

log "===== Phase B: pc_2agent (construct+proposal baru, ls=60, reps=3) ====="
"$PYTHON" -m try.promptbench.runners.bench_chain \
    --chains pc_2agent \
    --pick "$BASE_PICK" \
    --latent-steps 60 \
    --reps 3 \
    --workers 3 \
    2>&1 | tee "$LOG_DIR/phaseB_pc2agent.log"

log "===== Phase B: pcj_judger — semua judger variant (ls=60, reps=3 each) ====="
for judger in "${JUDGER_VARIANTS[@]}"; do
    log "  judger=$judger …"
    "$PYTHON" -m try.promptbench.runners.bench_chain \
        --chains pcj_judger \
        --pick "$BASE_PICK,judger=$judger" \
        --latent-steps 60 \
        --reps 3 \
        --workers 3 \
        2>&1 | tee "$LOG_DIR/phaseB_pcj_judger__${judger}.log"
    log "  judger=$judger selesai."
done

log "===== Semua selesai. Log di $LOG_DIR ====="
