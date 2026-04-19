"""
Mock data untuk setiap test case.

Tujuan: sediakan input kontekstual yang DALAM KONDISI NORMAL didapat dari agent
sebelumnya. Misal untuk 'construct', hipotesis yang seharusnya keluar dari
'propose' kita set manual di sini sesuai skema outputnya.

Silakan edit isinya (skenario, hipotesis, faktor, dll.) untuk mencoba input
yang berbeda tanpa menyentuh kode test.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


# ═════════════════════════════════════════════════════════════════════════
# Scenario — dipakai di hampir semua Jinja template via {{ scenario }}
# ═════════════════════════════════════════════════════════════════════════

SCENARIO_DESC = """\
Background: Penelitian mining alpha factor di pasar saham Indonesia (IDX).
Data daily OHLCV tersedia untuk ±800 saham liquid. Target: menghasilkan
ekspresi faktor yang menggeneralisir di out-of-sample dengan IC stabil.

Source Data:
  - Variabel yang tersedia di ekspresi faktor: $open, $close, $high, $low, $volume, $return
  - Granularitas: daily

Interface:
  - Ekspresi faktor ditulis dengan fungsi yang tersedia di function library.
  - Output faktor adalah DataFrame ber-index (datetime, instrument), satu kolom nilai faktor.

Output Format:
  - Hasil backtest dievaluasi dengan IC, IR, annualized return, max drawdown.
"""

# Versi ringkas (dipakai di beberapa call di 'feature' filter)
SCENARIO_DESC_COMPACT = (
    "Indonesia equity factor mining, daily OHLCV, variables: "
    "$open $close $high $low $volume $return. Output: DataFrame (datetime, instrument) → factor value."
)


# ═════════════════════════════════════════════════════════════════════════
# Hasil Propose — dipakai sebagai input untuk Construct & Feedback
# ═════════════════════════════════════════════════════════════════════════

# Bentuk ini sesuai schema di hypothesis_output_format yg dipelajari LLM:
HYPOTHESIS_DICT = {
    "hypothesis": "Saham dengan momentum positif jangka pendek dan volume meningkat cenderung out-perform 5 hari ke depan.",
    "concise_knowledge": "If short-term return > 0 AND volume growth > 0, THEN expected forward return is positive.",
    "concise_observation": "Kombinasi price momentum dan volume confirmation historis menghasilkan signal prediktif.",
    "concise_justification": "Price-volume confirmation adalah indikator klasik dari trend strength dalam behavioral finance.",
    "concise_specification": "Gunakan window 5-20 hari, RANK lintas saham, syarat volume pct-change > 0.",
}

# Representasi string yang dikirim ke Construct sebagai {{target_hypothesis}}
HYPOTHESIS_STR = f"""Hypothesis: {HYPOTHESIS_DICT['hypothesis']}
Concise Observation: {HYPOTHESIS_DICT['concise_observation']}
Concise Justification: {HYPOTHESIS_DICT['concise_justification']}
Concise Knowledge: {HYPOTHESIS_DICT['concise_knowledge']}
Concise Specification: {HYPOTHESIS_DICT['concise_specification']}
"""


# ═════════════════════════════════════════════════════════════════════════
# Trace — dipakai untuk {{hypothesis_and_feedback}} (Jinja loop trace.hist)
# ═════════════════════════════════════════════════════════════════════════

# Template Jinja iterates: trace.hist[-10:] → (hypothesis, experiment, feedback)
# experiment.sub_workspace_list[0].code_dict.get("model.py") → kode
# feedback.observations / .hypothesis_evaluation / .new_hypothesis / .reason / .decision

def _make_workspace(code: str) -> SimpleNamespace:
    return SimpleNamespace(
        sub_workspace_list=[SimpleNamespace(code_dict={"model.py": code})],
    )


def _make_feedback(obs: str, eva: str, new_h: str, reason: str, decision: bool) -> SimpleNamespace:
    return SimpleNamespace(
        observations=obs,
        hypothesis_evaluation=eva,
        new_hypothesis=new_h,
        reason=reason,
        decision=decision,
    )


# Round 1: gagal (IC rendah), Round 2: berhasil (IC naik tapi masih perlu improvement)
TRACE_HIST = [
    (
        "Hypothesis: Simple 10-day price momentum predicts forward return.",
        _make_workspace("expr = \"TS_MEAN($return, 10)\""),
        _make_feedback(
            obs="IC = 0.012, sangat rendah. Max drawdown 15%.",
            eva="Momentum saja tidak cukup; butuh konfirmasi volume.",
            new_h="Kombinasikan momentum dengan volume growth.",
            reason="Volume menambah konteks conviction dari price move.",
            decision=False,
        ),
    ),
    (
        "Hypothesis: Price momentum × volume confirmation untuk prediksi 5-day forward return.",
        _make_workspace("expr = \"RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))\""),
        _make_feedback(
            obs="IC = 0.035, meningkat dari baseline. IR masih 0.4.",
            eva="Arah benar, tapi struktur faktor bisa lebih halus.",
            new_h="Tambah normalisasi dengan ZSCORE pada volume, perpanjang window.",
            reason="Normalisasi mengurangi noise; window lebih panjang memperbaiki stabilitas.",
            decision=True,
        ),
    ),
]


class _MockScen:
    """Scenario mock — Jinja template pakai .background dan get_scenario_all_desc()."""
    background = SCENARIO_DESC

    def get_scenario_all_desc(self, task=None, filtered_tag=None, simple_background=None):
        return SCENARIO_DESC


class _MockTrace:
    """Trace mock — punya .scen dan .hist sesuai core.Trace.

    Jinja `hypothesis_and_feedback` iterate: for hypothesis, experiment, feedback in trace.hist[-10:]
    """
    def __init__(self, hist):
        self.scen = _MockScen()
        self.hist = hist


TRACE = _MockTrace(TRACE_HIST)
EMPTY_TRACE = _MockTrace([])


# ═════════════════════════════════════════════════════════════════════════
# Factor Task — dipakai Coder + Evaluator
# ═════════════════════════════════════════════════════════════════════════

FACTOR_TASK_INFO = """\
Factor Name: Momentum_Volume_Confirm_5D
Factor Description: Ranking momentum 5-hari dikalikan sign dari volume growth 5-hari.
Factor Formulation: RANK(TS\\_MEAN(return, 5)) \\cdot SIGN(TS\\_PCTCHANGE(volume, 5))
Variables:
  - $return: daily return
  - $volume: daily volume
"""

FACTOR_TASK_DESCRIPTION = "Momentum_Volume_Confirm_5D: ranking momentum 5 hari dikalikan sign volume growth 5 hari."

# Ekspresi 'lama' (gagal) — dipakai sebagai former_expression di coder retry
FORMER_EXPRESSION = "TS_MEAN($return, 5)"

# Feedback dari runner/evaluator ke ekspresi lama
FORMER_FEEDBACK = (
    "Execution OK, tetapi nilai faktor tidak ter-normalisasi lintas saham.\n"
    "Value feedback: IC = 0.008, terlalu rendah; distribusi nilai skewed karena "
    "tidak ada ranking cross-sectional. Pertimbangkan RANK() dan kombinasi dengan volume."
)

# Generated code (mock) — dipakai FactorCodeEvaluator
FACTOR_CODE = '''\
import pandas as pd
from qlib.contrib.alpha_expr_engine.expr_engine import ExprEngine

expr = "RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))"

def factor_func(df: pd.DataFrame) -> pd.DataFrame:
    engine = ExprEngine()
    return engine.evaluate(expr, df)
'''

# Execution feedback (mock) — hasil eksekusi kode, misal berhasil tanpa exception
EXECUTION_FEEDBACK_OK = (
    "Execution completed without exception.\n"
    "Output shape: (252 * 800, 1). Non-null ratio: 0.97."
)
EXECUTION_FEEDBACK_FAIL = (
    "ValueError: NaN encountered in output at >5% of rows.\n"
    "Traceback (most recent call last):\n"
    '  File "factor.py", line 7, in factor_func\n'
    "    return engine.evaluate(expr, df)\n"
    "ValueError: 'TS_PCTCHANGE' returned infinite values — denominator is zero."
)

VALUE_FEEDBACK = (
    "Correlation with gt = 0.42. IC = 0.035. "
    "Distribusi nilai faktor secara umum konsisten dengan ground truth, "
    "namun ada bias pada saham small-cap."
)


# ═════════════════════════════════════════════════════════════════════════
# Backtest result — dipakai Feedback ({{combined_result}})
# ═════════════════════════════════════════════════════════════════════════

COMBINED_RESULT_STR = """\
metric                                                       Current Result  SOTA Result  Bigger columns name
1day.excess_return_without_cost.max_drawdown                        0.0842       0.1120  Current Result
1day.excess_return_without_cost.information_ratio                   0.6310       0.4820  Current Result
1day.excess_return_without_cost.annualized_return                   0.1150       0.0890  Current Result
IC                                                                  0.0351       0.0280  Current Result
"""

TASK_DETAILS = [
    {
        "factor_name": "Momentum_Volume_Confirm_5D",
        "factor_description": "Ranking momentum 5-hari × sign volume growth 5-hari.",
        "factor_formulation": r"RANK(TS_MEAN(\$return, 5)) \cdot SIGN(TS_PCTCHANGE(\$volume, 5))",
        "variables": {"$return": "daily return", "$volume": "daily volume"},
        "factor_implementation": True,
    }
]


# ═════════════════════════════════════════════════════════════════════════
# External agents (data yang dikumpulkan di phase 1 search)
# ═════════════════════════════════════════════════════════════════════════

MAKRO_DATA_TEXT = """\
[1] Fed maintains rate at 5.25-5.50%; dovish Q&A but CPI still above 3%.
[2] ECB signals gradual easing as eurozone HICP falls to 2.1%.
[3] PBoC injects 500B CNY via MLF; Yuan weakens to 7.25/USD.
[4] US payroll surprise +240k, unemployment 3.9%; wage growth 4.2% YoY.
[5] Crude oil stable at 78/bbl; geopolitical premium easing.
"""

NEWS_DATA_TEXT = """\
[1] Tech earnings beat: NVDA +8%, AAPL +2% after results.
[2] Indonesia budget deficit widened; Rupiah weakens to 16200.
[3] IHSG turun 1.2% on foreign outflow; banking sector tertekan.
[4] OPEC+ extends production cuts through Q2.
[5] BI pertahankan suku bunga 6.25%; rupiah stabil jangka pendek.
"""

FUNDAMENTAL_DATA_TEXT = """\
[1] BBCA Q1 earnings +12% YoY; NIM stabil 5.6%.
[2] TLKM revenue flat; margin tertekan oleh capex data center.
[3] Banking sector P/B median 2.1×; consumer 4.3×.
[4] Coal miners ADRO, PTBA: dividend yield 12%+ di harga sekarang.
[5] Earnings revision breadth: positif di finansial, negatif di tech.
"""

TECHNICAL_DATA_TEXT = """\
[1] IHSG 50-day MA menembus 200-day MA ke bawah (death cross).
[2] BBCA RSI(14) = 62, uptrend intact.
[3] Volume rata-rata turun 15% MoM; sinyal likuiditas lemah.
[4] Bollinger band width menyempit di indeks → potensi breakout.
[5] Sektor komoditas menunjukkan rotasi positif dalam 30 hari terakhir.
"""

# Summary per agent — dipakai Manager untuk sintesa
AGENT_SUMMARIES = {
    "MAKRO": "Fed dovish di margin; BI hold. Tekanan pada Rupiah dan IHSG; sektor banking sensitif ke rate differential.",
    "NEWS":  "Foreign outflow tekan IHSG; sentimen global relatif bearish; sektor komoditas ter-support oleh OPEC+.",
    "FUNDAMENTAL": "Earnings revision positif di finansial, negatif di tech; coal miners kasih yield tinggi.",
    "TECHNICAL": "Death cross di IHSG; BB width menyempit — setup breakout; volume rata-rata turun.",
}


# ═════════════════════════════════════════════════════════════════════════
# Evolution (mutation & crossover) — parent trajectory summary
# ═════════════════════════════════════════════════════════════════════════

PARENT_HYPOTHESIS = HYPOTHESIS_DICT["hypothesis"]

PARENT_FACTORS_STR = (
    "- Momentum_Volume_Confirm_5D: RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))\n"
    "  Description: Ranking momentum 5-hari dikalikan sign volume growth 5-hari.\n"
)

PARENT_METRICS_STR = (
    "- IC: 0.0351\n"
    "- annualized_return: 0.1150\n"
    "- information_ratio: 0.6310\n"
    "- max_drawdown: 0.0842\n"
)

PARENT_FEEDBACK_STR = (
    "Faktor bekerja, tetapi terlalu bergantung pada signal momentum jangka pendek. "
    "Disarankan eksplorasi dimensi likuiditas atau volatility regime untuk diversifikasi."
)

# Crossover butuh multiple parents
PARENT_SUMMARIES_STR = """\
### Parent 1: Original Round
**Direction ID**: dir_0
**Hypothesis**: Momentum × volume confirmation untuk prediksi 5-day forward return.
**Factors**:
- RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))
**Metrics**:
- IC: 0.035, IR: 0.63
**Feedback**:
Bekerja tapi sempit; butuh diversifikasi ke dimensi lain.
---
### Parent 2: Mutation Round
**Direction ID**: dir_1
**Hypothesis**: Volatility regime-switch menggunakan rolling variance sebagai prediktor mean-reversion.
**Factors**:
- ZSCORE(TS_STD($return, 20)) * (-1) * TS_RANK($close, 20)
**Metrics**:
- IC: 0.028, IR: 0.55
**Feedback**:
Kurang stabil di regime tinggi-volatility; perlu filter tambahan.
---
"""
