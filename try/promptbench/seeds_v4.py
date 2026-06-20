"""
promptbench/seeds_v4.py
=======================
SEED BACKTEST untuk evaluasi pipeline REDESIGN v4 (feedback -> mutation/crossover
-> proposal -> design -> construct).

KENAPA FILE INI ADA
-------------------
Loop evolusi v4 dimulai dari agent `feedback`, yang butuh INPUT backtest. Sebelum
folder `try/outputs/` dihapus (mulai-ulang eksperimen), backtest nyata dari
trajektori lama DIPANEN ke sini supaya seed-nya durable & dapat di-version.

Sumber (dipanen 2026-06-20 dari try/outputs/, sebelum dihapus):
  - try/outputs/evolution_rekayasa/mutation_rekayasa_weak_momentum_*    -> SEED 1
  - try/outputs/evolution_rekayasa/mutation_rekayasa_overfit_meanrev_*  -> SEED 2
  - try/outputs/evolution_rekayasa/crossover_rekayasa_momentum_x_meanrev_* (Parent 1) -> SEED 3
  - try/outputs/proposal_feedback/feedback_*  +  fixtures_pb.TARGET_TEXT -> SEED 4

Angka Qlib lama (IC / annualized_return / information_ratio / max_drawdown)
DITERJEMAHKAN ke format metrik v4:
  - Block A = standalone RankIC + ICIR per faktor (metrik penentu, model-free).
  - Block B = combined LightGBM RankIC + MaxDrawdown (dekat baseline floor; konteks).
  - Block C = keputusan replace-best deterministik (FactorIC_mean vs SOTA).
RankIC <- IC; MaxDrawdown <- max_drawdown apa adanya; ICIR diperkirakan dari
kestabilan (IR + besar drawdown). Empat seed sengaja BERBEDA mekanisme DAN beda
jenis kelemahan struktural, supaya feedback/mutation/crossover punya bahan kaya.

BENTUK DATA
-----------
Tiap seed = dict dengan:
  name              : id pendek scenario
  hypothesis        : hipotesis parent (1 kalimat)
  factor_block      : Block A (string) — dipakai feedback (var `factor_block`)
  backtest_results  : Block B (string) — dipakai feedback (var `backtest_results`)
  sota_block        : Block C (string) — dipakai feedback (var `sota_block`)
  parent_text       : ringkas parent (hypothesis+expr+metrics+feedback) untuk
                      hand-off TEKS ke mutation/crossover (var `target_text` /
                      bagian dari `parents_text`).
CROSS_PAIRS = pasangan indeks seed untuk 2 run crossover (gabung 2 parent).
"""

from __future__ import annotations

from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# SEED 1 — weak momentum (noisy, tanpa normalisasi cross-sectional)
#   asal: mutation_rekayasa_weak_momentum (IC 0.005, AR 0.045, IR 0.18, MDD 0.185)
# ─────────────────────────────────────────────────────────────────────────────
SEED_WEAK_MOMENTUM = {
    "name": "weak_momentum",
    "hypothesis": (
        "Short-term price momentum over three days predicts next-day "
        "cross-sectional returns."
    ),
    "factor_block": (
        "[A] Standalone per-factor metrics:\n"
        "  Factor 1: Momentum_Raw_3D = TS_MEAN($return, 3)  "
        "[standalone RankIC=0.005, ICIR=0.08]"
    ),
    "backtest_results": (
        "[B] Combined LightGBM RankIC=0.041, MaxDrawdown=0.185 "
        "(near baseline floor; supplementary only). Annualised return 0.045, "
        "information ratio 0.18 — return thin relative to drawdown."
    ),
    "sota_block": (
        "[C] SOTA FactorIC_mean=0.025; this round FactorIC_mean=0.005 -> "
        "Replace Best Result: no (deterministic)."
    ),
    "parent_text": (
        "[Parent: weak_momentum]\n"
        "HYPOTHESIS: short-term three-day price momentum predicts next-day "
        "cross-sectional returns.\n"
        "EXPRESSION 1: TS_MEAN($return, 3)  [standalone RankIC=0.005, ICIR=0.08]\n"
        "METRICS: combined RankIC=0.041, MaxDrawdown=0.185, IR=0.18\n"
        "FEEDBACK: REFUTES as built — raw three-day return with no "
        "cross-sectional normalization is pure noise; window too short."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# SEED 2 — overfit mean-reversion (RankIC kuat tapi MDD katastrofik, tanpa gate)
#   asal: mutation_rekayasa_overfit_meanrev (IC 0.041, AR 0.182, IR 0.71, MDD 0.312)
# ─────────────────────────────────────────────────────────────────────────────
SEED_OVERFIT_MEANREV = {
    "name": "overfit_meanrev",
    "hypothesis": (
        "Stocks with strongly negative five-day cumulative return revert and "
        "recover within two days — fade short-term losers cross-sectionally."
    ),
    "factor_block": (
        "[A] Standalone per-factor metrics:\n"
        "  Factor 1: MeanReversion_5D = RANK(TS_MEAN($return, 5)) * (-1)  "
        "[standalone RankIC=0.041, ICIR=0.22]"
    ),
    "backtest_results": (
        "[B] Combined LightGBM RankIC=0.060, MaxDrawdown=0.312 "
        "(high RankIC but the drawdown is catastrophic). Annualised return 0.182, "
        "information ratio 0.71 in-sample — collapses in trending regimes."
    ),
    "sota_block": (
        "[C] SOTA FactorIC_mean=0.025; this round FactorIC_mean=0.041 -> "
        "Replace Best Result: yes (deterministic)."
    ),
    "parent_text": (
        "[Parent: overfit_meanrev]\n"
        "HYPOTHESIS: fade short-term losers — strongly negative five-day return "
        "reverts within two days, cross-sectionally.\n"
        "EXPRESSION 1: RANK(TS_MEAN($return, 5)) * (-1)  "
        "[standalone RankIC=0.041, ICIR=0.22]\n"
        "METRICS: combined RankIC=0.060, MaxDrawdown=0.312, IR=0.71\n"
        "FEEDBACK: PARTIALLY supports — strong cross-sectional reversal signal but "
        "no regime gate; pure contrarian collapses when momentum persists."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# SEED 3 — momentum x volume confirmation (kontrol risiko bagus tapi sempit)
#   asal: crossover_rekayasa Parent 1 (IC 0.035, AR 0.115, IR 0.63, MDD 0.084)
# ─────────────────────────────────────────────────────────────────────────────
SEED_MOMENTUM_VOLUME = {
    "name": "momentum_volume",
    "hypothesis": (
        "Stocks with positive short-term momentum and increasing trading volume "
        "outperform over the next five days."
    ),
    "factor_block": (
        "[A] Standalone per-factor metrics:\n"
        "  Factor 1: Momentum_Volume_Confirm_5D = "
        "RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))  "
        "[standalone RankIC=0.035, ICIR=0.48]"
    ),
    "backtest_results": (
        "[B] Combined LightGBM RankIC=0.052, MaxDrawdown=0.084 "
        "(strong risk control near baseline floor). Annualised return 0.115, "
        "information ratio 0.63 — works mainly in trending, high-volume regimes."
    ),
    "sota_block": (
        "[C] SOTA FactorIC_mean=0.025; this round FactorIC_mean=0.035 -> "
        "Replace Best Result: yes (deterministic)."
    ),
    "parent_text": (
        "[Parent: momentum_volume]\n"
        "HYPOTHESIS: positive short-term momentum confirmed by rising volume "
        "predicts five-day out-performance.\n"
        "EXPRESSION 1: RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))  "
        "[standalone RankIC=0.035, ICIR=0.48]\n"
        "METRICS: combined RankIC=0.052, MaxDrawdown=0.084, IR=0.63\n"
        "FEEDBACK: SUPPORTS but narrow — good drawdown control, yet SIGN discards "
        "volume magnitude and the edge only fires in trending high-volume regimes."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# SEED 4 — volume-zscore x range-compression reversal (leg kedua lemah/kolinear)
#   asal: fixtures_pb.TARGET_TEXT + feedback_* (RankIC 0.031/0.009, MDD 0.17)
# ─────────────────────────────────────────────────────────────────────────────
SEED_VOLUME_RANGE_REVERSAL = {
    "name": "volume_range_reversal",
    "hypothesis": (
        "When five-day cross-sectional volume z-score is elevated while intraday "
        "range stays compressed, next-day returns tend to reverse."
    ),
    "factor_block": (
        "[A] Standalone per-factor metrics:\n"
        "  Factor 1: TS_ZSCORE($volume, 5) - RANK(($high - $low) / $close)  "
        "[standalone RankIC=0.031, ICIR=0.41]\n"
        "  Factor 2: REGBETA($return, $volume, 20)  "
        "[standalone RankIC=0.009, ICIR=0.10]"
    ),
    "backtest_results": (
        "[B] Combined LightGBM RankIC=0.052, MaxDrawdown=0.170 "
        "(near baseline floor; supplementary only)."
    ),
    "sota_block": (
        "[C] SOTA FactorIC_mean=0.025; this round FactorIC_mean=0.020 -> "
        "Replace Best Result: no (deterministic)."
    ),
    "parent_text": (
        "[Parent: volume_range_reversal]\n"
        "HYPOTHESIS: elevated five-day volume z-score with compressed intraday "
        "range predicts next-day reversal.\n"
        "EXPRESSION 1: TS_ZSCORE($volume, 5) - RANK(($high - $low) / $close)  "
        "[standalone RankIC=0.031, ICIR=0.41]\n"
        "EXPRESSION 2: REGBETA($return, $volume, 20)  "
        "[standalone RankIC=0.009, ICIR=0.10]\n"
        "METRICS: combined RankIC=0.052, MaxDrawdown=0.170\n"
        "FEEDBACK: PARTIALLY supports — the volume-zscore leg carries the signal; "
        "the range leg is collinear and adds little; window too short for a stable beta."
    ),
}

# Urutan tetap → indeks 0..3 dipakai runner untuk feedback x4.
SEEDS: List[Dict[str, str]] = [
    SEED_WEAK_MOMENTUM,        # 0
    SEED_OVERFIT_MEANREV,      # 1
    SEED_MOMENTUM_VOLUME,      # 2
    SEED_VOLUME_RANGE_REVERSAL # 3
]

# Dua run crossover (gabung 2 parent), pasangan komplementer:
#   (1,2) overfit_meanrev x momentum_volume : gate volume P2 menambal MDD P1.
#   (0,3) weak_momentum x volume_range_rev   : normalisasi P3 menambal noise P0.
CROSS_PAIRS: List[tuple] = [(1, 2), (0, 3)]


def feedback_vars(seed: Dict[str, str]) -> Dict[str, str]:
    """Var yang dibutuhkan agent feedback untuk satu seed."""
    return {
        "factor_block": seed["factor_block"],
        "backtest_results": seed["backtest_results"],
        "sota_block": seed["sota_block"],
        "hypothesis": seed["hypothesis"],
    }


def parent_text(seed: Dict[str, str]) -> str:
    """Ringkas parent untuk hand-off TEKS (mutation `target_text`)."""
    return seed["parent_text"]


def parents_text(pair: tuple) -> str:
    """Gabung dua parent untuk hand-off TEKS crossover (`parents_text`)."""
    a, b = pair
    return SEEDS[a]["parent_text"] + "\n---\n" + SEEDS[b]["parent_text"]
