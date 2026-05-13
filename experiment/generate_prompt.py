"""
Dry-run prompt generator — QuantaAlpha + LatentMAS pipeline.

Renders every (system, user) prompt pair as it would look during a real production
run, WITHOUT calling any LLM.  Useful for prompt engineering, onboarding, and
debugging output collapse.

Usage
-----
    # from quantalatent/ folder:
    python experiment/generate_prompt.py
    python experiment/generate_prompt.py --step planning,propose_r1,construct_a1
    python experiment/generate_prompt.py --output all_prompts.txt

Available steps:
    planning
    propose_r1        Propose — Round 1 (direction supplied, no history)
    propose_rn        Propose — Round N (with trace history)
    construct_a1      Construct — Attempt 1 (full prompt)
    construct_retry   Construct — Attempt 2+ (minimal retry prompt)
    construct_dup     Construct — duplication / complexity feedback fragment
    consistency       Consistency checker (hypothesis → description → expression)
    feedback          Feedback agent (evaluate backtest, seed next hypothesis)
    coder_impl        Coder — implement_one_task (generate Python code)
    coder_error_sum   Coder — error_summary (similar-error knowledge summary)
    coder_evaluator   Coder — evaluator_code_feedback (code-review critique)
    mutation          Evolution: Mutation operator (orthogonal direction)
    crossover         Evolution: Crossover operator (hybrid direction)
    orthogonality     Evolution: Orthogonality check between two strategies

Pipeline order (full run):
    PLANNING → PROPOSE(×N) → CONSTRUCT → CONSISTENCY → CODER → FEEDBACK
             ↘ MUTATION / CROSSOVER (evolution) ↗
"""

from __future__ import annotations

import argparse
import datetime
import textwrap
from pathlib import Path

import yaml
from jinja2 import Environment, Undefined

# ─── path resolution ─────────────────────────────────────────────────────────
# Script lives in  quantalatent/experiment/
# Backend lives in quantalatent/backend/
SCRIPT_DIR  = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent / "backend"


# ─── YAML loader ─────────────────────────────────────────────────────────────
def _load_yaml(rel_path: str) -> dict:
    path = BACKEND_DIR / rel_path
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ─── render helpers ───────────────────────────────────────────────────────────
def _j2(template_str: str | None, **ctx) -> str:
    """Render a Jinja2 template.  Missing variables → empty string (permissive)."""
    if not template_str:
        return "(template not found in yaml)"
    return Environment(undefined=Undefined).from_string(template_str).render(**ctx)


def _fmt(template_str: str | None, **ctx) -> str:
    """Render a Python str.format() template (used by planning & evolution prompts)."""
    if not template_str:
        return "(template not found in yaml)"
    return template_str.format(**ctx)


# ─── output helpers ──────────────────────────────────────────────────────────
_SEP  = "=" * 80
_SSEP = "-" * 60


def _header(title: str) -> str:
    return f"\n{_SEP}\n  {title}\n{_SEP}\n"


def _block(label: str, content: str) -> str:
    return f"\n{_SSEP}\n[{label}]\n{_SSEP}\n{content.strip()}\n"


# ─── mock data ────────────────────────────────────────────────────────────────
# Synthetic but realistic values — gives a faithful picture of what real prompts
# look like without requiring a live data/model environment.

_SCENARIO_FULL = """\
[Background]
You are mining alpha factors for the Chinese A-share equity market.
Data universe: CSI 300 + CSI 500 constituents, daily frequency, 2015-2024.
Available fields: $open, $close, $high, $low, $volume, $return.
Factor signals are evaluated cross-sectionally (ranked per day across universe).

[Interface]
Factors are evaluated via Qlib. Each factor is a symbolic expression that maps
daily OHLCV data to a scalar per-stock signal.  Signals are ranked cross-sectionally;
portfolios are formed on rank deciles.  IC and ICIR are the primary metrics.

[Factor Output Format]
Factor code must produce a pd.Series indexed by (datetime, instrument)."""

_SCENARIO_BG = """\
Qlib factor mining on CSI 300/500 A-share daily data.
Allowed features: $open, $close, $high, $low, $volume, $return.
Factors are cross-sectional daily signals ranked per datetime."""

_INITIAL_DIR  = "Explore short-term return reversal patterns in high-turnover A-share stocks"
_DIRECTION    = "Mean-reversion after abnormal volume spikes signals short-term reversal opportunities"
_HYPO_TEXT    = (
    "Stocks with abnormally high trading volume relative to their 20-day moving average "
    "tend to show short-term price reversal within 3-5 days."
)
_HYPO_FULL    = f"""\
hypothesis: {_HYPO_TEXT}
concise_observation: IC of TS_ZSCORE($volume,20)>2 group shows -0.03 next-5d return (round 1)
concise_justification: High volume signals informed selling/buying exhaustion; reversal follows as liquidity normalises
concise_knowledge: When TS_ZSCORE($volume,20)>2 and SIGN($return)<0, 5d mean-reversion amplitude ≈ -0.04
concise_specification: Signal window 1-5 days, universe CSI-500, normalization RANK cross-sectional"""

_EXPR         = "RANK(TS_ZSCORE($volume, 20) * SIGN(TS_MEAN(-$return, 3)))"
_FACTOR_TASKS = [
    {
        "factor_name": "VolRevFactor",
        "factor_description": "Short-term price reversal based on abnormal volume z-score × return sign",
        "factor_formulation": r"RANK\left(TS\_ZSCORE(\$volume,20)\cdot SIGN\!\left(TS\_MEAN(-\$return,3)\right)\right)",
        "factor_expression": _EXPR,
        "factor_implementation": True,
        "complexity_feedback": None,
        "variables": {"$volume": "daily trading volume", "$return": "daily return"},
    },
    {
        "factor_name": "HighLowRange",
        "factor_description": "Normalised intraday price range as a proxy for volatility compression",
        "factor_formulation": r"TS\_MEAN\!\left(\frac{\$high-\$low}{\$close+10^{-8}},5\right)",
        "factor_expression": "TS_MEAN(($high - $low) / ($close + 1e-8), 5)",
        "factor_implementation": True,
        "complexity_feedback": None,
        "variables": {"$high": "daily high", "$low": "daily low", "$close": "closing price"},
    },
]
_BACKTEST_RESULT = """\
              IC    ICIR  Annualized_Return  Sharpe  Max_Drawdown  Win_Rate
VolRevFactor   0.028  0.41          6.2%       0.83     -8.1%        51.3%
HighLowRange   0.014  0.21          3.1%       0.42    -12.4%        49.8%
---
SOTA Benchmark (previous best): IC=0.035, AR=8.5%"""

_PARENT = {
    "trajectory_id": "traj_001",
    "hypothesis": _HYPO_TEXT,
    "factors": [{"name": "VolRevFactor", "expression": _EXPR, "description": "Volume reversal signal"}],
    "metrics": {"IC": 0.028, "ICIR": 0.41, "annualized_return": 0.062, "max_drawdown": -0.081},
    "feedback": "Signal partially supported but decays after day 3. Needs momentum gating.",
}
_PARENT_SUMMARIES = """\
<parent index="1" phase="Original Round" direction_id="1">
  <hypothesis>{hypo}</hypothesis>
  <factors>VolRevFactor: {expr}</factors>
  <metrics>IC=0.028, AR=6.2%, DD=-8.1%</metrics>
  <feedback>Volume reversal weak, decays after day 3.</feedback>
</parent>
<parent index="2" phase="Original Round" direction_id="2">
  <hypothesis>Momentum persistence at weekly scale driven by institutional herding</hypothesis>
  <factors>RANK(TS_MEAN($return, 20)) + 0.3 * RANK(TS_STD($return, 5))</factors>
  <metrics>IC=0.041, AR=9.1%, DD=-6.8%</metrics>
  <feedback>Strong mid-term signal; misses short-term noise. Combine with volume filter.</feedback>
</parent>""".format(hypo=_HYPO_TEXT, expr=_EXPR)

_MOCK_CODE = """\
import pandas as pd

def get_factor(data):
    volume = data['$volume']
    ret    = data['$return']
    vol_z  = volume.groupby(level='datetime').transform(
        lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-8)
    )
    ret_mean = ret.groupby(level='instrument').transform(lambda x: -x.rolling(3).mean())
    sign_ret = ret_mean.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    factor = vol_z * sign_ret
    return factor.groupby(level='datetime').rank(pct=True)"""

_EXEC_FB   = "KeyError: '$volume' — check field names; Qlib uses lowercase keys from data config."
_TASK_INFO = (
    "VolRevFactor: compute RANK(TS_ZSCORE($volume,20) * SIGN(TS_MEAN(-$return,3))). "
    "Description: Short-term reversal signal after volume spikes."
)


# ─── mock objects for Jinja2 rendering ────────────────────────────────────────

class _MockFeedback:
    observations       = "IC=0.028 (weak positive). Top-decile AR=6.2% annualised. MaxDD=-8.1%."
    hypothesis_evaluation = "Partial support: volume reversal captured but signal decays after day 3."
    new_hypothesis     = "Combine volume z-score with price momentum sign to sharpen reversal entry."
    reason             = "Signal decays fast; adding momentum sign as gate improves selectivity."
    decision           = False


class _MockHypothesis:
    hypothesis            = _HYPO_TEXT
    concise_observation   = "IC of TS_ZSCORE($volume,20)>2 group shows -0.03 next-5d return (round 1)"
    concise_justification = "High volume → exhaustion → reversal as liquidity normalises"
    concise_knowledge     = "When TS_ZSCORE($volume,20)>2 and SIGN($return)<0, 5d reversal ≈ -0.04"
    concise_specification = "Window 1-5d, CSI-500, RANK cross-sectional"

    def __str__(self):
        return _HYPO_FULL


class _MockExp:
    sub_tasks = []


class _MockTrace:
    """Minimal Trace mock that the Jinja2 `hypothesis_and_feedback` template can iterate over."""
    def __init__(self, with_history: bool = True):
        class _Scen:
            background = _SCENARIO_BG
            def get_scenario_all_desc(self, *a, **kw): return _SCENARIO_FULL
            def get_compact_desc(self, *a, **kw):      return _SCENARIO_BG
        self.scen = _Scen()
        if with_history:
            self.hist = [(_MockHypothesis(), _MockExp(), _MockFeedback())]
        else:
            self.hist = []


# ─── individual stage renderers ───────────────────────────────────────────────

def _stage_planning(p: dict) -> tuple[str, str]:
    n = 3
    sys_p  = _fmt(p.get("system", ""), initial_direction=_INITIAL_DIR, n=n)
    user_p = _fmt(p.get("user", ""),   initial_direction=_INITIAL_DIR, n=n)
    out_f  = p.get("output_format", "")
    if out_f:
        user_p += "\n\n" + out_f.replace("{n}", str(n))
    return sys_p, user_p


def _stage_propose_r1(p: dict) -> tuple[str, str]:
    """Round 1 — direction provided, no trace history."""
    haf = _j2(p.get("potential_direction_transformation", ""),
               potential_direction=_DIRECTION)
    sys_p = _j2(p.get("hypothesis_gen", {}).get("system_prompt", ""),
                targets="factors",
                scenario=_SCENARIO_FULL,
                hypothesis_output_format=p.get("hypothesis_output_format", ""),
                hypothesis_specification=p.get("factor_hypothesis_specification", ""))
    user_p = _j2(p.get("hypothesis_gen", {}).get("user_prompt", ""),
                 targets="factors",
                 hypothesis_and_feedback=haf,
                 RAG=None,
                 round=0)
    return sys_p, user_p


def _stage_propose_rn(p: dict) -> tuple[str, str]:
    """Round N — trace history present."""
    trace = _MockTrace(with_history=True)
    haf   = _j2(p.get("hypothesis_and_feedback", ""), trace=trace)
    sys_p = _j2(p.get("hypothesis_gen", {}).get("system_prompt", ""),
                targets="factors",
                scenario=_SCENARIO_FULL,
                hypothesis_output_format=p.get("hypothesis_output_format", ""),
                hypothesis_specification=p.get("factor_hypothesis_specification", ""))
    user_p = _j2(p.get("hypothesis_gen", {}).get("user_prompt", ""),
                 targets="factors",
                 hypothesis_and_feedback=haf,
                 RAG=None,
                 round=1)
    return sys_p, user_p


def _stage_construct_a1(p: dict) -> tuple[str, str]:
    """Construct attempt 1 — full prompt (no duplication feedback yet)."""
    trace = _MockTrace(with_history=True)
    haf   = _j2(p.get("hypothesis_and_feedback", ""), trace=trace)
    sys_p = _j2(p.get("hypothesis2experiment", {}).get("system_prompt", ""),
                targets="factors",
                scenario=_SCENARIO_BG,
                experiment_output_format=p.get("experiment_output_format", ""))
    user_p = _j2(p.get("hypothesis2experiment", {}).get("user_prompt", ""),
                 targets="factors",
                 target_hypothesis=str(_MockHypothesis()),
                 hypothesis_and_feedback=haf,
                 function_lib_description=p.get("function_lib_description", ""),
                 target_list=[],
                 RAG=None,
                 expression_duplication=None)
    return sys_p, user_p


def _stage_construct_retry(p: dict) -> tuple[str, str]:
    """Construct attempt 2+ — minimal retry prompt to avoid prompt overload on small models."""
    retry_sys  = p.get("hypothesis2experiment_retry_system_prompt",
                       "(key 'hypothesis2experiment_retry_system_prompt' not found — add to factors/prompts/prompts.yaml)")
    retry_user = p.get("hypothesis2experiment_retry_user_prompt",
                       "(key 'hypothesis2experiment_retry_user_prompt' not found — add to factors/prompts/prompts.yaml)")
    compact    = p.get("factor_experiment_compact_schema", "")
    sys_p  = retry_sys
    user_p = _j2(retry_user,
                 attempt_n=2,
                 error_log="JSON parse failed: output must be raw JSON object starting with {\"FactorName\"",
                 hypothesis_title=_HYPO_TEXT,
                 function_lib_description=p.get("function_lib_description", ""),
                 compact_schema=compact)
    return sys_p, user_p


def _stage_construct_dup(p: dict) -> str:
    """Duplication/complexity fragment — injected into construct user_prompt when expression fails regulator."""
    return _j2(p.get("expression_duplication", ""),
               prev_expression=_EXPR,
               duplicated_subtree_size=6,
               duplication_threshold=5,
               duplicated_subtree="RANK($volume)",
               matched_alpha="Factor_042_RANK_vol",
               free_args_ratio=0.3,
               num_free_args=3,
               unique_vars_ratio=0.4,
               num_unique_vars=4,
               num_all_nodes=10,
               symbol_length=180,
               symbol_length_threshold=250,
               num_base_features=3,
               base_features_threshold=6)


def _stage_consistency(p: dict) -> tuple[str, str]:
    t = _FACTOR_TASKS[0]
    sys_p  = _j2(p.get("consistency_check_system", ""))
    user_p = _j2(p.get("consistency_check_user", ""),
                 hypothesis=_HYPO_TEXT,
                 factor_name=t["factor_name"],
                 factor_description=t["factor_description"],
                 factor_formulation=t["factor_formulation"],
                 factor_expression=t["factor_expression"],
                 variables=t["variables"])
    return sys_p, user_p


def _stage_feedback(p: dict) -> tuple[str, str]:
    fb = p.get("factor_feedback_generation", {})
    sys_p  = _j2(fb.get("system", ""), scenario=_SCENARIO_FULL)
    user_p = _j2(fb.get("user", ""),
                 hypothesis_text=_HYPO_TEXT,
                 task_details=_FACTOR_TASKS,
                 combined_result=_BACKTEST_RESULT)
    return sys_p, user_p


def _stage_coder_impl(p: dict) -> tuple[str, str]:
    """Coder implement_one_task — first attempt, no prior failure knowledge."""
    sys_p  = _j2(p.get("evolving_strategy_factor_implementation_v1_system", ""),
                 scenario=_SCENARIO_FULL,
                 queried_former_failed_knowledge=[])
    user_p = _j2(p.get("evolving_strategy_factor_implementation_v2_user", ""),
                 factor_information_str=_TASK_INFO,
                 queried_similar_error_knowledge=[],
                 error_summary_critics=None,
                 similar_successful_factor_description="MomentumFactor: TS_MEAN($return,20)",
                 similar_successful_expression="RANK(TS_MEAN($return, 20))",
                 latest_attempt_to_latest_successful_execution=None)
    return sys_p, user_p


def _stage_coder_error_sum(p: dict) -> tuple[str, str]:
    """Coder error_summary — called when CoSTEER has similar-error knowledge to summarise."""
    from types import SimpleNamespace

    _fail_impl  = SimpleNamespace(code=_MOCK_CODE)
    _fixed_code = _MOCK_CODE.replace("data['$volume']", "data['volume']")
    _fixed_impl = SimpleNamespace(code=_fixed_code)
    _task_obj   = SimpleNamespace(get_task_information=lambda: _TASK_INFO)

    class _Fail:
        implementation = _fail_impl
        target_task    = _task_obj
        feedback       = _EXEC_FB

    class _Fixed:
        implementation = _fixed_impl

    similar_err = [("KeyError: '$volume'", [_Fail(), _Fixed()])]

    sys_p  = _j2(p.get("evolving_strategy_error_summary_v2_system", ""),
                 scenario=_SCENARIO_FULL,
                 factor_information_str=_TASK_INFO,
                 code_and_feedback=f"Code:\n{_MOCK_CODE}\n\nFeedback:\n{_EXEC_FB}")
    user_p = _j2(p.get("evolving_strategy_error_summary_v2_user", ""),
                 queried_similar_error_knowledge=similar_err)
    return sys_p, user_p


def _stage_coder_evaluator(p: dict) -> tuple[str, str]:
    """Coder evaluator_code_feedback — code-review critique sent to coding agent."""
    sys_p  = _j2(p.get("evaluator_code_feedback_v1_system", ""), scenario=_SCENARIO_FULL)
    user_p = _j2(p.get("evaluator_code_feedback_v1_user", ""),
                 factor_information=_TASK_INFO,
                 code=_MOCK_CODE,
                 execution_feedback=_EXEC_FB,
                 value_feedback=None,
                 gt_code=None)
    return sys_p, user_p


def _stage_mutation(p: dict) -> tuple[str, str, str]:
    """Mutation operator — system + user + suffix (appended to next Propose user_prompt)."""
    m = p.get("mutation", {})
    metrics_str = "\n".join(
        f"- {k}: {v:.4f}" for k, v in _PARENT["metrics"].items() if v is not None
    )
    factors_str = "\n".join(
        f"- {f['name']}: {f['expression']}\n  Description: {f['description']}"
        for f in _PARENT["factors"]
    )
    sys_p  = m.get("system", "")
    user_p = _fmt(m.get("user", ""),
                  parent_hypothesis=_PARENT["hypothesis"],
                  parent_factors=factors_str,
                  parent_metrics=metrics_str,
                  parent_feedback=_PARENT["feedback"])
    suffix = _fmt(m.get("suffix_template", ""),
                  parent_summary=f"Trajectory {_PARENT['trajectory_id']}: {_PARENT['hypothesis'][:60]}...",
                  new_hypothesis="Overnight return gap × early-session volume surge as a momentum gate",
                  exploration_direction="Cross-session return asymmetry; overnight gap vs intraday continuation",
                  orthogonality_reason="Parent explored intraday volume reversal; this explores overnight drift momentum")
    return sys_p, user_p, suffix


def _stage_crossover(p: dict) -> tuple[str, str, str]:
    """Crossover operator — system + user + suffix."""
    c = p.get("crossover", {})
    sys_p  = c.get("system", "")
    user_p = _fmt(c.get("user", ""), parent_summaries=_PARENT_SUMMARIES)
    suffix = _fmt(c.get("suffix_template", ""),
                  parent_summaries=_PARENT_SUMMARIES,
                  hybrid_hypothesis="Regime-aware alpha: fuse volume-reversal with momentum persistence",
                  fusion_logic="VolRev as entry gate; MomentumPersist as direction filter above 1.5σ volume",
                  innovation_points="Conditional activation: reversal fires only when momentum confirms direction shift")
    return sys_p, user_p, suffix


def _stage_orthogonality(p: dict) -> tuple[str, str]:
    """Orthogonality check between two candidate strategies."""
    o = p.get("orthogonality_check", {})
    sys_p  = o.get("system", "")
    user_p = _fmt(o.get("user", ""),
                  strategy_a=f"Volume reversal: {_EXPR}",
                  strategy_b="Momentum persistence: RANK(TS_MEAN($return, 20))")
    return sys_p, user_p


# ─── step registry ────────────────────────────────────────────────────────────

STEP_ALL = [
    "planning",
    "propose_r1", "propose_rn",
    "construct_a1", "construct_retry", "construct_dup",
    "consistency",
    "feedback",
    "coder_impl", "coder_error_sum", "coder_evaluator",
    "mutation", "crossover", "orthogonality",
]


# ─── main renderer ────────────────────────────────────────────────────────────

def render_all(steps: list[str]) -> str:
    props  = _load_yaml("factors/prompts/prompts.yaml")
    plan   = _load_yaml("pipeline/prompts/planning_prompts.yaml")
    evol   = _load_yaml("pipeline/prompts/evolution_prompts.yaml")
    cons   = _load_yaml("factors/regulator/consistency_prompts.yaml")
    coder  = _load_yaml("factors/coder/prompts.yaml")

    parts: list[str] = []

    def _emit(title: str, sys_p: str, user_p: str, extras: dict | None = None) -> None:
        s = _header(title)
        s += _block("SYSTEM PROMPT", sys_p)
        s += _block("USER PROMPT", user_p)
        if extras:
            for lbl, content in extras.items():
                s += _block(lbl, content)
        parts.append(s)

    # ── 1. Planning ──────────────────────────────────────────────────────────
    if "planning" in steps:
        s, u = _stage_planning(plan)
        _emit(
            "STAGE 1 — PLANNING  (Planner → N parallel Propose directions)",
            s, u,
            {"NOTE": (
                "Template uses Python .format() not Jinja2.\n"
                "Source: pipeline/prompts/planning_prompts.yaml\n"
                "Called by: pipeline/planning.py::generate_parallel_directions()"
            )},
        )

    # ── 2a. Propose Round 1 ──────────────────────────────────────────────────
    if "propose_r1" in steps:
        s, u = _stage_propose_r1(props)
        _emit(
            "STAGE 2a — PROPOSE  Round 1  (no history, initial direction → hypothesis)",
            s, u,
            {"NOTE": (
                "Source: factors/prompts/prompts.yaml  keys: hypothesis_gen.*\n"
                "Called by: factors/proposal.py::AlphaAgentHypothesisGen.gen()\n"
                "hypothesis_and_feedback rendered from potential_direction_transformation template"
            )},
        )

    # ── 2b. Propose Round N ──────────────────────────────────────────────────
    if "propose_rn" in steps:
        s, u = _stage_propose_rn(props)
        _emit(
            "STAGE 2b — PROPOSE  Round N  (with trace history, prev feedback in KV-cache)",
            s, u,
            {"NOTE": (
                "history_limit defaults to last 2 rounds (older rounds in latent KV-cache).\n"
                "hypothesis_and_feedback rendered from hypothesis_and_feedback template (trace.hist[-2:])"
            )},
        )

    # ── 3a. Construct Attempt 1 ──────────────────────────────────────────────
    if "construct_a1" in steps:
        s, u = _stage_construct_a1(props)
        _emit(
            "STAGE 3a — CONSTRUCT  Attempt 1  (hypothesis → factor expressions, full prompt)",
            s, u,
            {"NOTE": (
                "Source: factors/prompts/prompts.yaml  keys: hypothesis2experiment.*\n"
                "Called by: factors/proposal.py::AlphaAgentHypothesis2FactorExpression._convert_with_history_limit()\n"
                "If expression fails regulator → inject duplication feedback and retry (see STAGE 3c)"
            )},
        )

    # ── 3b. Construct Retry ──────────────────────────────────────────────────
    if "construct_retry" in steps:
        s, u = _stage_construct_retry(props)
        _emit(
            "STAGE 3b — CONSTRUCT  Attempt 2+  (minimal retry prompt to avoid overload on Qwen3-4B)",
            s, u,
            {"NOTE": (
                "Keys hypothesis2experiment_retry_system/user_prompt must exist in prompts.yaml.\n"
                "Attempt 3+: also resets _past_kv to None (KV clean slate)."
            )},
        )

    # ── 3c. Construct Duplication Fragment ───────────────────────────────────
    if "construct_dup" in steps:
        dup = _stage_construct_dup(props)
        _emit(
            "STAGE 3c — CONSTRUCT  expression_duplication fragment  (injected into user_prompt on regulator fail)",
            "(injected into construct user_prompt — no separate system prompt)",
            dup,
            {"NOTE": (
                "This fragment is rendered and appended to the construct user_prompt when:\n"
                "  - expression is not parsable by FactorRegulator\n"
                "  - expression fails is_expression_acceptable() (dup/complexity/base-features checks)\n"
                "  - intra-response duplicate detected\n"
                "Source: expression_duplication template in factors/prompts/prompts.yaml"
            )},
        )

    # ── 4. Consistency Check ─────────────────────────────────────────────────
    if "consistency" in steps:
        s, u = _stage_consistency(cons)
        _emit(
            "STAGE 4 — CONSISTENCY CHECK  (hypothesis → description → formulation → expression)",
            s, u,
            {"NOTE": (
                "Source: factors/regulator/consistency_prompts.yaml\n"
                "Called by: factors/regulator/consistency_checker.py::FactorConsistencyChecker.check_consistency()\n"
                "Triggered inside AlphaAgentHypothesis2FactorExpression.convert() when consistency_enabled=True\n"
                "On failure: LLM may suggest corrected_expression or corrected_description"
            )},
        )

    # ── 5. Feedback ──────────────────────────────────────────────────────────
    if "feedback" in steps:
        s, u = _stage_feedback(props)
        _emit(
            "STAGE 5 — FEEDBACK  (evaluate backtest results, generate next hypothesis seed)",
            s, u,
            {"NOTE": (
                "Source: factors/prompts/prompts.yaml  key: factor_feedback_generation\n"
                "Called by: factors/feedback.py::AlphaAgentQlibFactorHypothesisExperiment2Feedback.generate_feedback()\n"
                "Input KV: construct_kv (propose→construct context already in latent KV)\n"
                "Output KV: chained to _pipeline_kv → used in next iteration's Propose"
            )},
        )

    # ── 6a. Coder Implement ──────────────────────────────────────────────────
    if "coder_impl" in steps:
        s, u = _stage_coder_impl(coder)
        _emit(
            "STAGE 6a — CODER  implement_one_task  (generate Python code for one FactorTask)",
            s, u,
            {"NOTE": (
                "Source: factors/coder/prompts.yaml  keys: evolving_strategy_factor_implementation_v1_system + v2_user\n"
                "Called by: factors/coder/evolving_strategy.py::FactorMultiProcessEvolvingStrategy.implement_one_task()\n"
                "Shown here: first attempt (no prior failures, one similar-success example injected)"
            )},
        )

    # ── 6b. Coder Error Summary ──────────────────────────────────────────────
    if "coder_error_sum" in steps:
        s, u = _stage_coder_error_sum(coder)
        _emit(
            "STAGE 6b — CODER  error_summary  (summarise similar-error knowledge before retry)",
            s, u,
            {"NOTE": (
                "Source: factors/coder/prompts.yaml  keys: evolving_strategy_error_summary_v2_system + user\n"
                "Called when: CoSTEERQueriedKnowledgeV2 + v2_error_summary=True + prior failures exist\n"
                "Output critics are injected into implement_one_task user_prompt as error_summary_critics"
            )},
        )

    # ── 6c. Coder Evaluator ──────────────────────────────────────────────────
    if "coder_evaluator" in steps:
        s, u = _stage_coder_evaluator(coder)
        _emit(
            "STAGE 6c — CODER  evaluator_code_feedback  (code-review critique for coding agent)",
            s, u,
            {"NOTE": (
                "Source: factors/coder/prompts.yaml  keys: evaluator_code_feedback_v1_system + user\n"
                "Called by: factors/coder/eva_utils.py::FactorEvaluator\n"
                "Critics are NOT shown to user — sent to coding agent to correct code"
            )},
        )

    # ── 7. Mutation ──────────────────────────────────────────────────────────
    if "mutation" in steps:
        s, u, sfx = _stage_mutation(evol)
        _emit(
            "STAGE 7 — MUTATION  (generate ONE orthogonal direction from one parent trajectory)",
            s, u,
            {
                "SUFFIX TEMPLATE  (appended to next Propose user_prompt as strategy_suffix)": sfx,
                "NOTE": (
                    "Template uses Python .format() not Jinja2.\n"
                    "Source: pipeline/prompts/evolution_prompts.yaml  key: mutation\n"
                    "Called by: pipeline/evolution/mutation.py::MutationOperator.generate_mutation()\n"
                    "Two output paths:\n"
                    "  A) detailed: parse JSON → extract new_hypothesis → build suffix_template\n"
                    "  B) simple_user: output hypothesis text directly (fallback)\n"
                    "Suffix is passed as strategy_suffix to the next AlphaAgentLoop.__init__()"
                ),
            },
        )

    # ── 8. Crossover ─────────────────────────────────────────────────────────
    if "crossover" in steps:
        s, u, sfx = _stage_crossover(evol)
        _emit(
            "STAGE 8 — CROSSOVER  (fuse best elements from N parent trajectories into ONE hybrid direction)",
            s, u,
            {
                "SUFFIX TEMPLATE  (appended to next Propose user_prompt)": sfx,
                "NOTE": (
                    "Template uses Python .format() not Jinja2.\n"
                    "Source: pipeline/prompts/evolution_prompts.yaml  key: crossover\n"
                    "Called by: pipeline/evolution/crossover.py::CrossoverOperator.generate_crossover()\n"
                    "parent_summaries built by rendering crossover.parent_template for each parent."
                ),
            },
        )

    # ── 9. Orthogonality Check ───────────────────────────────────────────────
    if "orthogonality" in steps:
        s, u = _stage_orthogonality(evol)
        _emit(
            "STAGE 9 — ORTHOGONALITY CHECK  (score independence of two strategies, 1-10)",
            s, u,
            {"NOTE": (
                "Source: pipeline/prompts/evolution_prompts.yaml  key: orthogonality_check\n"
                "Template uses Python .format().\n"
                "Used by evolution controller to decide whether to keep a mutation/crossover candidate."
            )},
        )

    return "\n".join(parts)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
        Available steps:
          {', '.join(STEP_ALL)}

        Examples:
          python experiment/generate_prompt.py
          python experiment/generate_prompt.py --step propose_r1,propose_rn
          python experiment/generate_prompt.py --step construct_a1 --output prompt_construct.txt
        """),
    )
    ap.add_argument(
        "--step", default="all",
        help="Comma-separated step names, or 'all' (default: all)",
    )
    ap.add_argument(
        "--output", default=None,
        help="Write output to file instead of stdout",
    )
    args = ap.parse_args()

    if args.step == "all":
        steps = STEP_ALL
    else:
        steps = [s.strip() for s in args.step.split(",") if s.strip()]
        unknown = [s for s in steps if s not in STEP_ALL]
        if unknown:
            ap.error(f"Unknown step(s): {unknown}\nValid steps: {STEP_ALL}")

    body = render_all(steps)

    header = (
        f"\n{'#' * 80}\n"
        f"  QuantaAlpha + LatentMAS — Pipeline Prompt Dry Run\n"
        f"  Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Steps     : {', '.join(steps)}\n"
        f"  Backend   : {BACKEND_DIR}\n"
        f"{'#' * 80}\n"
        f"  NOTE: All data is SYNTHETIC mock data — realistic placeholders only.\n"
        f"        Prompts reflect the exact template structure used in production.\n"
        f"{'#' * 80}\n"
    )
    output = header + body

    if args.output:
        out = Path(args.output)
        out.write_text(output, encoding="utf-8")
        print(f"Written to: {out.resolve()}")
    else:
        print(output)


if __name__ == "__main__":
    main()
