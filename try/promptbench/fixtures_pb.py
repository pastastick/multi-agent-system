"""
promptbench/fixtures_pb.py
==========================
Input ter-mock untuk benchmark per-agent latent_mas. Variabel di-key sesuai
template di backend/latent_mas/prompts.yaml (BUKAN fixtures.py lama yang untuk
pipeline factors/ rdagent).

Dipakai Phase A (independen): tiap agent dirender dengan superset dict ini.
Agent yang butuh konteks hulu (construct/judger/consistency) diberi konteks
sebagai TEKS deterministik supaya bisa diuji terisolasi (latennya nanti diuji
via KV-chain di Phase B).
"""

from __future__ import annotations

# ── direction (proposal/judger/mutation/crossover) ──────────────────────────
DIRECTION = (
    "Find a cross-sectional alpha factor from daily OHLCV that predicts next-day "
    "returns, exploiting short-horizon liquidity and volatility dynamics."
)

MARKET_CONTEXT = "Liquid equities, daily bars, 2018-2021 train segment."

PRIOR_FEEDBACK = (
    "Prior round: a volume-zscore momentum factor had standalone RankIC 0.018 "
    "(ICIR 0.22) — weak but positive; the volatility-gated variant was noisier."
)

NEGATIVE_HINT = (
    "AVOID failed mechanisms from earlier rounds: pure low-volume + volatility-spike "
    "mean-reversion (over-mined, unstable)."
)

# ── construct/judger upstream-as-text (untuk uji independen) ─────────────────
DIVERSITY_HINT = (
    "Diversity: recent factors leaned on cross_sectional + time_series families. "
    "Consider an under-used family — smoothing (EMA/DECAYLINEAR), regression "
    "(REGBETA/REGRESI), or conditional gating (COUNT/SUMIF) — if the mechanism fits."
)

# Hipotesis teks (di chain nyata ada di KV; di Phase A independen disuntik teks)
HYPOTHESIS_TEXT = (
    "When 5-day cross-sectional volume z-score is elevated while the intraday range "
    "($high-$low)/$close stays compressed, next-day returns tend to reverse."
)

# ── repair ───────────────────────────────────────────────────────────────────
FORMER_EXPRESSION = "RANK($volume, 5) - TS_CORR($close, $close, 10)"
ERROR_LOG = (
    "arity: RANK takes 1 arg (cross-sectional, no window); "
    "degenerate: TS_CORR needs two DIFFERENT series, got $close with itself."
)
VALUE_FEEDBACK = "IC = 0.006, near zero; distribution degenerate."
ATTEMPT_MODE = "minimal"

# ── feedback (factor_block / backtest_results / sota_block) ──────────────────
FACTOR_BLOCK = (
    "[A] Standalone per-factor metrics:\n"
    "  Factor 1: TS_ZSCORE($volume,5) - RANK(($high-$low)/$close)  "
    "[standalone RankIC=0.031, ICIR=0.41]\n"
    "  Factor 2: REGBETA($return, $volume, 20)  [standalone RankIC=0.009, ICIR=0.10]"
)
BACKTEST_RESULTS = (
    "[B] Combined LightGBM RankIC=0.052, ICIR=0.48, MaxDrawdown=0.17 "
    "(near baseline floor; supplementary only)."
)
SOTA_BLOCK = (
    "[C] SOTA FactorIC_mean=0.025; this round FactorIC_mean=0.020 → "
    "Replace Best Result: no (deterministic)."
)

# ── evolution (mutation/crossover parents-as-text) ───────────────────────────
TARGET_TEXT = (
    "[Parent] HYPOTHESIS: elevated 5-day volume z-score with compressed range "
    "predicts next-day reversal.\n"
    "EXPRESSION 1: TS_ZSCORE($volume,5) - RANK(($high-$low)/$close)  "
    "[standalone RankIC=0.031, ICIR=0.41]\n"
    "EXPRESSION 2: REGBETA($return, $volume, 20)  [standalone RankIC=0.009, ICIR=0.10]\n"
    "METRICS: combined RankIC=0.052, MaxDrawdown=0.17\n"
    "FEEDBACK: PARTIALLY supports; factor 2 weak (window too short for beta)."
)
PARENTS_TEXT = (
    "[Parent A] momentum on $return: DELTA($close,10)/TS_STD($return,10) "
    "[RankIC=0.028]\n"
    "[Parent B] volume pressure: RANK($volume) * SIGN(DELTA($volume,5)) "
    "[RankIC=0.022]"
)
N_PARENTS = 2

# ── superset dict: dipakai LatentAgent.render(**FIXTURES) ─────────────────────
FIXTURES = {
    "direction": DIRECTION,
    "market_context": MARKET_CONTEXT,
    "prior_feedback": PRIOR_FEEDBACK,
    "negative_hint": NEGATIVE_HINT,
    "diversity_hint": DIVERSITY_HINT,
    "hypothesis": HYPOTHESIS_TEXT,
    "target_hypothesis": HYPOTHESIS_TEXT,
    "former_expression": FORMER_EXPRESSION,
    "error_log": ERROR_LOG,
    "value_feedback": VALUE_FEEDBACK,
    "attempt_mode": ATTEMPT_MODE,
    "factor_block": FACTOR_BLOCK,
    "backtest_results": BACKTEST_RESULTS,
    "sota_block": SOTA_BLOCK,
    "target_text": TARGET_TEXT,
    "parents_text": PARENTS_TEXT,
    "n_parents": N_PARENTS,
}
