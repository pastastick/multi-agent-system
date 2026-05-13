"""
Inputs/variables needed by the ORIGINAL prompts in the backend.

The prompts themselves are NOT defined here — they are loaded verbatim from the
backend YAML files (factors/prompts/prompts.yaml, factors/coder/prompts.yaml,
factors/coder/qa_prompts.yaml, pipeline/prompts/*.yaml) by the test modules.

What you CAN freely change in this file:
  * Any value under a "── EDIT ──" section.
  * Language of the content (all values are English by default).
  * Number of trace rounds, factor expressions, hypotheses, etc.

What you should NOT change:
  * The field names / dict keys — backend templates refer to them by name.
  * The type of objects (e.g. FactorTask instance, list of tuples for trace).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Backend must be on sys.path so we can import real Scenario / FactorTask.
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from backend.factors.coder.factor import FactorTask                    # real task class


# ═════════════════════════════════════════════════════════════════════════
#  ── EDIT ──  SCENARIO
#  Scenario sections are loaded verbatim from rdagent's qlib YAML
#  (the SAME YAML production uses through `T("...").r()`). We skip the
#  QlibAlphaAgentScenario() constructor because it requires a qlib data
#  folder; instead we read the YAML and stitch the scenario text with
#  the SAME layout as `QlibAlphaAgentScenario.get_scenario_all_desc()`
#  at backend/factors/experiment.py:155.
#
#  What to edit freely:
#    - SOURCE_DATA_DESC  (the "source data" blurb)
#    - RUNTIME_ENV_DESC  (the "runtime environment" blurb)
#    - Or replace the whole RealScenario class with an inline string.
# ═════════════════════════════════════════════════════════════════════════

import importlib.util as _ilu
import yaml as _yaml
from jinja2 import Environment as _Env, StrictUndefined as _SU


def _find_rdagent_prompts() -> Path:
    spec = _ilu.find_spec("rdagent")
    if spec and spec.submodule_search_locations:
        candidate = Path(next(iter(spec.submodule_search_locations))) / "scenarios" / "qlib" / "experiment" / "prompts.yaml"
        if candidate.exists():
            return candidate
    # fallback: hardcoded legacy path
    fallback = Path(
        "/root/projects/first-experiment/ai-agent/.venv/lib/python3.10/"
        "site-packages/rdagent/scenarios/qlib/experiment/prompts.yaml"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "rdagent prompts.yaml tidak ditemukan. "
        "Pastikan rdagent terinstall di environment aktif (pip install rdagent)."
    )


_RDAGENT_PROMPTS = _find_rdagent_prompts()

with _RDAGENT_PROMPTS.open() as _f:
    _RDAGENT_YAML = _yaml.safe_load(_f)

RUNTIME_ENV_DESC = "Python 3.10, venv-managed, pandas, numpy, qlib."

SOURCE_DATA_DESC = (
    "Daily OHLCV data is available for ~800 liquid equities. Variables that can "
    "be referenced inside factor expressions: $open, $close, $high, $low, "
    "$volume, $return."
)


def _render_yaml(key: str, **vars) -> str:
    return _Env(undefined=_SU).from_string(_RDAGENT_YAML[key]).render(**vars)


class RealScenario:
    """Mirror of QlibAlphaAgentScenario but built from YAML only."""

    background: str = _render_yaml("qlib_factor_background", runtime_environment=RUNTIME_ENV_DESC)
    interface: str = _render_yaml("qlib_factor_interface")
    strategy: str = _render_yaml("qlib_factor_strategy")
    output_format: str = _render_yaml("qlib_factor_output_format")
    simulator: str = _render_yaml("qlib_factor_simulator")
    experiment_setting: str = _render_yaml("qlib_factor_experiment_setting")

    def get_source_data_desc(self, task=None) -> str:
        return SOURCE_DATA_DESC

    def get_scenario_all_desc(
        self,
        task=None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        """Matches backend/factors/experiment.py:get_scenario_all_desc."""
        if simple_background:
            return f"Background of the scenario:\n{self.background}"

        if filtered_tag == "hypothesis_and_experiment":
            return "\n\n".join([
                f"Background of the scenario:\n{self.background}",
                f"Strategy context:\n{self.strategy}",
                f"Experiment setting:\n{self.experiment_setting}",
            ])

        if filtered_tag == "feature":
            return (
                f"Background of the scenario:\n{self.background}\n\n"
                f"The source data you can use:\n{self.get_source_data_desc(task)}\n\n"
                f"The interface you should follow to write the runnable code:\n{self.interface}\n\n"
                f"The output of your code should be in the format:\n{self.output_format}"
            )

        return "\n\n".join([
            f"Background of the scenario:\n{self.background}",
            f"The source data you can use:\n{self.get_source_data_desc(task)}",
            f"The interface you should follow to write the runnable code:\n{self.interface}",
            f"The output of your code should be in the format:\n{self.output_format}",
            f"The simulator user can use to test your factor:\n{self.simulator}",
            f"Strategy context:\n{self.strategy}",
            f"Experiment setting:\n{self.experiment_setting}",
        ])


SCENARIO = RealScenario()


# ═════════════════════════════════════════════════════════════════════════
#  ── EDIT ──  HYPOTHESIS  (fed into `construct` and `feedback`)
#  Matches the JSON schema emitted by propose (hypothesis_output_format).
# ═════════════════════════════════════════════════════════════════════════

HYPOTHESIS_DICT: dict[str, str] = {
    "hypothesis": (
        "Stocks with positive short-term momentum AND increasing trading volume "
        "tend to outperform over the next 5 days."
    ),
    "concise_observation": (
        "Historically, pairing short-horizon price momentum with volume "
        "confirmation yields predictive cross-sectional signals."
    ),
    "concise_justification": (
        "Price-volume confirmation is a classical indicator of trend strength "
        "in behavioral finance; rising volume increases conviction in the move."
    ),
    "concise_knowledge": (
        "If short-term return > 0 AND volume growth > 0, THEN expected forward "
        "return is positive."
    ),
    "concise_specification": (
        "Use a 5-20 day window, cross-sectional RANK, and require volume "
        "pct-change > 0."
    ),
}

# Text form fed to construct via {{ target_hypothesis }}
HYPOTHESIS_STR = "\n".join([
    f"Hypothesis: {HYPOTHESIS_DICT['hypothesis']}",
    f"Concise Observation: {HYPOTHESIS_DICT['concise_observation']}",
    f"Concise Justification: {HYPOTHESIS_DICT['concise_justification']}",
    f"Concise Knowledge: {HYPOTHESIS_DICT['concise_knowledge']}",
    f"Concise Specification: {HYPOTHESIS_DICT['concise_specification']}",
])


# ═════════════════════════════════════════════════════════════════════════
#  ── EDIT ──  TRACE  (round history fed into propose & construct)
#  Template `hypothesis_and_feedback` iterates:
#      for hypothesis, experiment, feedback in trace.hist[-10:]
#  with:
#      experiment.sub_workspace_list[0].code_dict.get("model.py")
#      feedback.observations / .hypothesis_evaluation / .new_hypothesis /
#      .reason / .decision
# ═════════════════════════════════════════════════════════════════════════

def _mk_workspace(code: str) -> SimpleNamespace:
    return SimpleNamespace(
        sub_workspace_list=[SimpleNamespace(code_dict={"model.py": code})],
    )


def _mk_feedback(obs: str, eva: str, new_h: str, reason: str, decision: bool) -> SimpleNamespace:
    return SimpleNamespace(
        observations=obs,
        hypothesis_evaluation=eva,
        new_hypothesis=new_h,
        reason=reason,
        decision=decision,
    )


TRACE_HIST = [
    (
        "Hypothesis: Simple 10-day price momentum predicts forward return.",
        _mk_workspace('expr = "TS_MEAN($return, 10)"'),
        _mk_feedback(
            obs="IC = 0.012, very low. Max drawdown 15%.",
            eva="Momentum alone is insufficient; volume confirmation is needed.",
            new_h="Combine momentum with volume growth.",
            reason="Volume adds conviction context to the price move.",
            decision=False,
        ),
    ),
    (
        "Hypothesis: Price momentum × volume confirmation predicts 5-day forward return.",
        _mk_workspace(
            'expr = "RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))"'
        ),
        _mk_feedback(
            obs="IC = 0.035, improved over baseline. IR is still 0.4.",
            eva="Direction is correct, but the factor structure can be refined.",
            new_h="Add ZSCORE normalization on volume and extend the window.",
            reason="Normalization reduces noise; longer windows improve stability.",
            decision=True,
        ),
    ),
]


class _MockTrace:
    """Mock Trace — provides .scen and .hist like core.Trace."""

    def __init__(self, scen, hist):
        self.scen = scen
        self.hist = hist


TRACE = _MockTrace(SCENARIO, TRACE_HIST)
EMPTY_TRACE = _MockTrace(SCENARIO, [])


# ═════════════════════════════════════════════════════════════════════════
#  ── EDIT ──  FACTOR TASK  (coder_retry, code_feedback, final_decision)
#  Built via the real FactorTask — `.get_task_information()` /
#  `.get_task_description()` therefore produce the exact strings the
#  production pipeline emits.
# ═════════════════════════════════════════════════════════════════════════

FACTOR_TASK = FactorTask(
    factor_name="Momentum_Volume_Confirm_5D",
    factor_description=(
        "Cross-sectional ranking of 5-day momentum multiplied by the sign of "
        "5-day volume growth."
    ),
    factor_formulation=(
        r"\text{RANK}(\text{TS\_MEAN}(\$return, 5)) \cdot "
        r"\text{SIGN}(\text{TS\_PCTCHANGE}(\$volume, 5))"
    ),
    variables={
        "$return": "daily return",
        "$volume": "daily volume",
    },
    factor_expression="RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))",
    factor_implementation=True,
)


# Failing attempt (for coder_retry former_expression)
FORMER_EXPRESSION = "TS_MEAN($return, 5)"

FORMER_FEEDBACK = (
    "Execution OK but the factor is not cross-sectionally normalized.\n"
    "Value feedback: IC = 0.008, too low; value distribution is skewed because "
    "no ranking is applied. Consider RANK() and combining with volume."
)

# Code emitted by a retry attempt (used by code_feedback + error_summary)
FACTOR_CODE = '''\
import pandas as pd
from qlib.contrib.alpha_expr_engine.expr_engine import ExprEngine

expr = "RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))"

def factor_func(df: pd.DataFrame) -> pd.DataFrame:
    engine = ExprEngine()
    return engine.evaluate(expr, df)
'''

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
    "Factor values are broadly consistent with the ground truth, "
    "but small-cap names exhibit a bias."
)

# Literal string pushed into output_format evaluator (backend builds this
# at eva_utils.py:253 from gen_df.info()).
OUTPUT_DF_INFO_STR = (
    "The user is currently working on a feature related task.\n"
    "The output dataframe info is:\n"
    "<class 'pandas.core.frame.DataFrame'>\n"
    "MultiIndex: 201600 entries, ('2020-01-02', 'BBCA') to ('2024-12-30', 'TLKM')\n"
    "Data columns (total 1 columns):\n"
    " #   Column                      Non-Null Count   Dtype\n"
    "---  -----------------------     --------------   -----\n"
    " 0   Momentum_Volume_Confirm_5D  195432 non-null  float64\n"
    "dtypes: float64(1)\n"
    "memory usage: 2.2+ MB\n"
)


# ═════════════════════════════════════════════════════════════════════════
#  ── EDIT ──  BACKTEST RESULT  (feedback `combined_result`)
#  `process_results()` in the backend emits a similar table.
# ═════════════════════════════════════════════════════════════════════════

COMBINED_RESULT_STR = """\
metric                                                       Current Result  SOTA Result  Bigger columns name
1day.excess_return_without_cost.max_drawdown                        0.0842       0.1120  Current Result
1day.excess_return_without_cost.information_ratio                   0.6310       0.4820  Current Result
1day.excess_return_without_cost.annualized_return                   0.1150       0.0890  Current Result
IC                                                                  0.0351       0.0280  Current Result
"""

# task_details list — backend builds this via
#   [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
TASK_DETAILS = [FACTOR_TASK.get_task_information_and_implementation_result()]


# ═════════════════════════════════════════════════════════════════════════
#  ── EDIT ──  PLANNING  (initial direction for generate_parallel_directions)
# ═════════════════════════════════════════════════════════════════════════

PLANNING_INITIAL_DIRECTION = (
    "Explore momentum-volume factors on daily equities with short windows "
    "(≤5 days) to capture short-term reversal in high-beta names."
)

PLANNING_NUM_DIRECTIONS = 4


# ═════════════════════════════════════════════════════════════════════════
#  ── EDIT ──  EVOLUTION  (mutation & crossover parents)
# ═════════════════════════════════════════════════════════════════════════

PARENT_HYPOTHESIS = HYPOTHESIS_DICT["hypothesis"]

PARENT_FACTORS_STR = (
    "- Momentum_Volume_Confirm_5D: "
    "RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))\n"
    "  Description: 5-day momentum rank multiplied by sign of 5-day volume growth.\n"
)

PARENT_METRICS_STR = (
    "- IC: 0.0351\n"
    "- annualized_return: 0.1150\n"
    "- information_ratio: 0.6310\n"
    "- max_drawdown: 0.0842\n"
)

PARENT_FEEDBACK_STR = (
    "Factor works but leans too heavily on short-term momentum. "
    "Consider exploring liquidity or volatility-regime dimensions for diversification."
)

# Crossover needs multiple parents — backend formats them with
# `evolution_prompts.yaml:crossover.parent_template`.
PARENT_SUMMARIES_STR = """\
### Parent 1: Original Round
**Direction ID**: dir_0
**Hypothesis**: Momentum × volume confirmation for 5-day forward return prediction.
**Factors**:
- RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))
**Metrics**:
- IC: 0.035, IR: 0.63
**Feedback**:
Works but narrow; needs diversification across other dimensions.
---
### Parent 2: Mutation Round
**Direction ID**: dir_1
**Hypothesis**: Volatility-regime switching using rolling variance as a mean-reversion predictor.
**Factors**:
- ZSCORE(TS_STD($return, 20)) * (-1) * TS_RANK($close, 20)
**Metrics**:
- IC: 0.028, IR: 0.55
**Feedback**:
Unstable in high-volatility regimes; needs an additional filter.
---
"""


# ═════════════════════════════════════════════════════════════════════════
#  ── EDIT ──  EVOLUTION REKAYASA  (multiple varied parent sets)
#  3 parents for mutation validation — each with a distinct failure mode.
#  3 crossover groups (2 parents each): 2 complementary + 1 anti-complementary.
# ═════════════════════════════════════════════════════════════════════════

MUTATION_PARENTS: list[dict] = [
    # Parent A: too noisy — no normalization, no volume filter
    {
        "label": "weak_momentum",
        "parent_hypothesis": (
            "Short-term price momentum (3-day) predicts next-day cross-sectional returns."
        ),
        "parent_factors_str": (
            "- Momentum_Raw_3D: TS_MEAN($return, 3)\n"
            "  Description: 3-day rolling mean of daily return, no cross-sectional normalization.\n"
        ),
        "parent_metrics_str": (
            "- IC: 0.005\n"
            "- annualized_return: 0.045\n"
            "- information_ratio: 0.18\n"
            "- max_drawdown: 0.185\n"
        ),
        "parent_feedback_str": (
            "IC is extremely low (0.005). No cross-sectional normalization applied. "
            "Pure 3-day price momentum is too noisy without ranking or volume confirmation. "
            "Annualized return (4.5%) is insufficient relative to max drawdown (18.5%)."
        ),
    },
    # Parent B: strong in-sample metrics but catastrophic MDD in trending regimes
    {
        "label": "overfit_meanrev",
        "parent_hypothesis": (
            "Short-term mean-reversion: stocks with strongly negative 5-day cumulative return "
            "will recover within 2 days — fade the short-term losers cross-sectionally."
        ),
        "parent_factors_str": (
            "- MeanReversion_5D: RANK(TS_MEAN($return, 5)) * (-1)\n"
            "  Description: Cross-sectional rank of 5-day return, inverted to bet on reversal.\n"
        ),
        "parent_metrics_str": (
            "- IC: 0.041\n"
            "- annualized_return: 0.182\n"
            "- information_ratio: 0.71\n"
            "- max_drawdown: 0.312\n"
        ),
        "parent_feedback_str": (
            "Strong in-sample IC (0.041) and IR (0.71) but catastrophic max drawdown (31.2%). "
            "Strategy collapses during trending regimes — pure contrarian fails when momentum is persistent. "
            "No regime filter present. Likely overfit to mean-reverting sub-periods in the test window."
        ),
    },
    # Parent C: high IC but poor risk — vol-scaling amplifies liquidity risk in small-caps
    {
        "label": "high_ic_bad_risk",
        "parent_hypothesis": (
            "Volatility-adjusted momentum: normalize price momentum by realized volatility "
            "to improve cross-sectional Sharpe — dampen signals in high-vol, risky stocks."
        ),
        "parent_factors_str": (
            "- VolAdjMomentum_20D: RANK(TS_MEAN($return, 20)) / (TS_STD($return, 20) + 0.0001)\n"
            "  Description: 20-day momentum rank scaled inversely by realized volatility.\n"
        ),
        "parent_metrics_str": (
            "- IC: 0.058\n"
            "- annualized_return: 0.093\n"
            "- information_ratio: 0.52\n"
            "- max_drawdown: 0.241\n"
        ),
        "parent_feedback_str": (
            "IC is strong (0.058) but IR is only 0.52 — high dispersion in returns across stocks. "
            "Max drawdown (24.1%) is too high. Vol-scaling amplifies signals in illiquid, low-vol micro-caps. "
            "Missing liquidity filter (e.g., volume-based) and a cap on leverage from vol inversion. "
            "No volume dimension incorporated — should add volume confirmation or turnover-based filter."
        ),
    },
]


def _fmt_crossover_parent(
    idx: int, label: str,
    hypothesis: str, factors_str: str,
    metrics_str: str, feedback_str: str,
) -> str:
    return (
        f"### Parent {idx}: {label}\n"
        f"**Hypothesis**: {hypothesis}\n"
        f"**Factors**:\n{factors_str}"
        f"**Metrics**:\n{metrics_str}"
        f"**Feedback**:\n{feedback_str}\n"
        "---"
    )


CROSSOVER_GROUPS: list[dict] = [
    # Group 1: Complementary — momentum (trending) × mean-reversion (ranging)
    {
        "label": "momentum_x_meanrev",
        "description": "Complementary: momentum (trend) × mean-reversion (range) — regime-adaptive fusion expected",
        "parent_summaries_str": "\n".join([
            _fmt_crossover_parent(
                1, "Momentum × Volume Confirmation",
                "Stocks with positive short-term momentum AND increasing trading volume "
                "tend to outperform over the next 5 days.",
                "- Momentum_Volume_Confirm_5D: RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))\n"
                "  Description: 5-day momentum rank multiplied by sign of 5-day volume growth.\n",
                "- IC: 0.035\n- annualized_return: 0.115\n- information_ratio: 0.63\n- max_drawdown: 0.084\n",
                "Good risk control (MDD=8.4%) but narrow — only works in trending, high-volume regimes.",
            ),
            _fmt_crossover_parent(
                2, "Mean-Reversion Contrarian",
                "Short-term mean-reversion: stocks with strongly negative 5-day return will recover.",
                "- MeanReversion_5D: RANK(TS_MEAN($return, 5)) * (-1)\n"
                "  Description: Contrarian fade of 5-day losers.\n",
                "- IC: 0.041\n- annualized_return: 0.182\n- information_ratio: 0.71\n- max_drawdown: 0.312\n",
                "Strong in ranging regime but catastrophic MDD (31.2%). Needs regime detection to suppress in trends.",
            ),
        ]),
    },
    # Group 2: Complementary — vol-adjusted (high IC) × volume-confirmed (low MDD)
    {
        "label": "voladj_x_volconfirm",
        "description": "Complementary: vol-adjusted (high IC) × volume-confirmed (low MDD) — fuse for both",
        "parent_summaries_str": "\n".join([
            _fmt_crossover_parent(
                1, "Volatility-Adjusted Momentum",
                "Normalize price momentum by realized volatility to improve cross-sectional Sharpe.",
                "- VolAdjMomentum_20D: RANK(TS_MEAN($return, 20)) / (TS_STD($return, 20) + 0.0001)\n"
                "  Description: 20-day momentum dampened by realized volatility.\n",
                "- IC: 0.058\n- annualized_return: 0.093\n- information_ratio: 0.52\n- max_drawdown: 0.241\n",
                "High IC (0.058) but MDD too high (24.1%). Vol-scaling amplifies illiquid small-cap signals.",
            ),
            _fmt_crossover_parent(
                2, "Momentum × Volume Confirmation",
                "Stocks with positive short-term momentum AND increasing volume tend to outperform.",
                "- Momentum_Volume_Confirm_5D: RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))\n"
                "  Description: 5-day momentum rank multiplied by sign of 5-day volume growth.\n",
                "- IC: 0.035\n- annualized_return: 0.115\n- information_ratio: 0.63\n- max_drawdown: 0.084\n",
                "Moderate IC but excellent risk control. Volume filter naturally screens illiquid names.",
            ),
        ]),
    },
    # Group 3: Anti-complementary — nearly identical 3D and 5D raw momentum (low orthogonality)
    {
        "label": "anti_complement_momentum",
        "description": "Anti-complementary: near-identical 3D vs 5D momentum — expect LLM to flag overlap",
        "parent_summaries_str": "\n".join([
            _fmt_crossover_parent(
                1, "Raw Short Momentum (3-day)",
                "3-day price momentum predicts next-day cross-sectional returns.",
                "- Momentum_Raw_3D: TS_MEAN($return, 3)\n"
                "  Description: 3-day rolling mean return, no normalization.\n",
                "- IC: 0.005\n- annualized_return: 0.045\n- information_ratio: 0.18\n- max_drawdown: 0.185\n",
                "Too noisy, no normalization. High correlation with 5D version — no new information.",
            ),
            _fmt_crossover_parent(
                2, "Raw Short Momentum (5-day)",
                "5-day price momentum predicts 2-day forward cross-sectional returns.",
                "- Momentum_Raw_5D: TS_MEAN($return, 5)\n"
                "  Description: 5-day rolling mean return, no normalization.\n",
                "- IC: 0.008\n- annualized_return: 0.060\n- information_ratio: 0.22\n- max_drawdown: 0.162\n",
                "Marginally better than 3D but same fundamental flaws. High correlation with 3D factor.",
            ),
        ]),
    },
]
