"""
Run the ORIGINAL propose / construct / feedback prompts against the LLM.

Prompts are loaded verbatim from backend/factors/prompts/prompts.yaml and
rendered exactly the way the backend does (see proposal.py + feedback.py):

  propose       → hypothesis_gen.system_prompt + hypothesis_gen.user_prompt
  construct     → hypothesis2experiment.system_prompt + user_prompt
  feedback      → factor_feedback_generation.system + user

All template variables come from `fixtures.py`. To change what the LLM sees,
edit the "── EDIT ──" sections in that file.
"""

from __future__ import annotations

from types import SimpleNamespace

from jinja2 import Environment, StrictUndefined

from .common import load_yaml, run_case, PROMPT_PATHS
from . import fixtures as fx


_FACTORS_YAML: dict | None = None


def _factors() -> dict:
    global _FACTORS_YAML
    if _FACTORS_YAML is None:
        _FACTORS_YAML = load_yaml(PROMPT_PATHS["factors_prompts"])
    return _FACTORS_YAML


def _jinja(template: str, **vars) -> str:
    return Environment(undefined=StrictUndefined).from_string(template).render(**vars)


def _render_hypothesis_and_feedback(trace, limit: int = 6) -> str:
    """Mirror of `factors.proposal.render_hypothesis_and_feedback`."""
    y = _factors()
    if len(trace.hist) == 0:
        return "No previous hypothesis and feedback available since it's the first round."

    lt = SimpleNamespace(scen=trace.scen, hist=trace.hist[-limit:])
    return _jinja(y["hypothesis_and_feedback"], trace=lt)


# ═════════════════════════════════════════════════════════════════════════
# Propose — AlphaAgentHypothesisGen.gen (backend/factors/proposal.py:351)
# Required Jinja vars:
#   system: targets, scenario, hypothesis_output_format, hypothesis_specification
#   user:   targets, hypothesis_and_feedback, RAG, round
# ═════════════════════════════════════════════════════════════════════════

def test_propose():
    y = _factors()
    trace = fx.TRACE
    scenario_desc = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
    hypothesis_and_feedback = _render_hypothesis_and_feedback(trace, limit=6)

    system_prompt = _jinja(
        y["hypothesis_gen"]["system_prompt"],
        targets="factor",
        scenario=scenario_desc,
        hypothesis_output_format=y["hypothesis_output_format"],
        hypothesis_specification=y["factor_hypothesis_specification"],
    )
    user_prompt = _jinja(
        y["hypothesis_gen"]["user_prompt"],
        targets="factor",
        hypothesis_and_feedback=hypothesis_and_feedback,
        RAG=None,
        round=len(trace.hist),
    )

    return run_case(
        group="proposal_feedback", case="propose",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        expected_keys=[
            "hypothesis", "concise_knowledge", "concise_observation",
            "concise_justification", "concise_specification",
        ],
    )


# ═════════════════════════════════════════════════════════════════════════
# Construct — AlphaAgentHypothesis2FactorExpression._convert_with_history_limit
#   (backend/factors/proposal.py:593)
# Required Jinja vars:
#   system: targets, scenario, experiment_output_format
#   user:   targets, target_hypothesis, hypothesis_and_feedback,
#           function_lib_description, target_list, RAG, expression_duplication
# ═════════════════════════════════════════════════════════════════════════

def test_construct():
    y = _factors()
    trace = fx.TRACE
    scenario_desc = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
    hypothesis_and_feedback = _render_hypothesis_and_feedback(trace, limit=6)

    system_prompt = _jinja(
        y["hypothesis2experiment"]["system_prompt"],
        targets="factor",
        scenario=scenario_desc,
        experiment_output_format=y["factor_experiment_output_format"],
    )
    user_prompt = _jinja(
        y["hypothesis2experiment"]["user_prompt"],
        targets="factor",
        target_hypothesis=fx.HYPOTHESIS_STR,
        hypothesis_and_feedback=hypothesis_and_feedback,
        function_lib_description=y["function_lib_description"],
        target_list=None,
        RAG=None,
        expression_duplication=None,
    )

    return run_case(
        group="proposal_feedback", case="construct",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
    )


# ═════════════════════════════════════════════════════════════════════════
# Feedback — AlphaAgentQlibFactorHypothesisExperiment2Feedback.generate_feedback
#   (backend/factors/feedback.py:278)
# Required Jinja vars:
#   system: scenario
#   user:   hypothesis_text, task_details, combined_result
# ═════════════════════════════════════════════════════════════════════════

def test_feedback():
    y = _factors()
    scenario_desc = fx.SCENARIO.get_scenario_all_desc()

    system_prompt = _jinja(
        y["factor_feedback_generation"]["system"],
        scenario=scenario_desc,
    )
    user_prompt = _jinja(
        y["factor_feedback_generation"]["user"],
        hypothesis_text=fx.HYPOTHESIS_DICT["hypothesis"],
        task_details=fx.TASK_DETAILS,
        combined_result=fx.COMBINED_RESULT_STR,
    )

    return run_case(
        group="proposal_feedback", case="feedback",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        expected_keys=[
            "Observations", "Feedback for Hypothesis",
            "New Hypothesis", "Reasoning", "Replace Best Result",
        ],
    )


CASES = {
    "propose": test_propose,
    "construct": test_construct,
    "feedback": test_feedback,
}
