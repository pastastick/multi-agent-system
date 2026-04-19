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

import yaml as _yaml
from jinja2 import Environment as _Env, StrictUndefined as _SU

_RDAGENT_PROMPTS = Path(
    "/root/projects/first-experiment/ai-agent/.venv/lib/python3.10/"
    "site-packages/rdagent/scenarios/qlib/experiment/prompts.yaml"
)

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
