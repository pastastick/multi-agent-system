"""
Test LLM calls untuk proposal (propose + construct) dan feedback.

Semua prompt dari factors/prompts/prompts.yaml dan di-render dengan Jinja2
(StrictUndefined). Tiga kasus:

  - propose    hypothesis_gen.system_prompt + hypothesis_gen.user_prompt
  - construct  hypothesis2experiment.system_prompt + user_prompt
  - feedback   factor_feedback_generation.system + user

Catatan penting:
  * hypothesis_gen.system_prompt menyematkan hypothesis_output_format mentah
    dari YAML (lewat variabel {{ hypothesis_output_format }}) — kita ikuti persis.
  * hypothesis_and_feedback sendiri adalah SUB-template Jinja2; harus dirender
    dulu pakai fx.TRACE sebelum dimasukkan ke user_prompt propose/construct.
  * construct user_prompt juga butuh {{ function_lib_description }} dari YAML.
"""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined

from .common import load_yaml, run_case, PROMPT_PATHS
from . import fixtures as fx


# ═════════════════════════════════════════════════════════════════════════
# Load YAML sekali (backend pakai singleton Prompts — ini mirror-nya)
# ═════════════════════════════════════════════════════════════════════════

_FACTORS_YAML: dict | None = None


def _factors() -> dict:
    global _FACTORS_YAML
    if _FACTORS_YAML is None:
        _FACTORS_YAML = load_yaml(PROMPT_PATHS["factors_prompts"])
    return _FACTORS_YAML


def _jinja(template: str, **vars) -> str:
    """Jinja2 render dengan StrictUndefined (match backend semantics)."""
    return Environment(undefined=StrictUndefined).from_string(template).render(**vars)


def _render_hypothesis_and_feedback(trace, limit: int = 6) -> str:
    """Mirror `factors.proposal.render_hypothesis_and_feedback`."""
    y = _factors()
    if len(trace.hist) == 0:
        return "No previous hypothesis and feedback available since it's the first round."
    # Batasi N entry terakhir
    class _LimitedTrace:
        pass
    lt = _LimitedTrace()
    lt.scen = trace.scen
    lt.hist = trace.hist[-limit:]
    return _jinja(y["hypothesis_and_feedback"], trace=lt)


# Konstanta default seperti di backend (proposal.py)
_TARGETS = "factor"


# ═════════════════════════════════════════════════════════════════════════
# Test cases
# ═════════════════════════════════════════════════════════════════════════

def test_propose():
    """
    Propose: generate hipotesis baru berdasarkan trace riwayat.
    Input:
      - trace.hist dari fx.TRACE (2 round: 1 gagal, 1 parsial sukses)
    Output expected: JSON dengan keys hypothesis / concise_knowledge /
    concise_observation / concise_justification / concise_specification.
    """
    y = _factors()
    trace = fx.TRACE
    scenario_desc = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")

    hypothesis_and_feedback = _render_hypothesis_and_feedback(trace, limit=6)

    system_prompt = _jinja(
        y["hypothesis_gen"]["system_prompt"],
        targets=_TARGETS,
        scenario=scenario_desc,
        hypothesis_output_format=y["hypothesis_output_format"],
        hypothesis_specification=y["factor_hypothesis_specification"],
    )
    user_prompt = _jinja(
        y["hypothesis_gen"]["user_prompt"],
        targets=_TARGETS,
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
            "hypothesis", "concise_knowledge",
            "concise_observation", "concise_justification", "concise_specification",
        ],
    )


def test_construct():
    """
    Construct: dari hipotesis → 2-3 faktor (expression + deskripsi + formulation).
    Input:
      - target_hypothesis = fx.HYPOTHESIS_STR (hasil mock dari propose)
      - hypothesis_and_feedback = riwayat terakhir dari fx.TRACE
    Output expected: JSON dict di mana setiap key = nama faktor, value =
    { description, variables, formulation, expression }.
    """
    y = _factors()
    trace = fx.TRACE
    scenario_desc = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")

    hypothesis_and_feedback = _render_hypothesis_and_feedback(trace, limit=6)

    system_prompt = _jinja(
        y["hypothesis2experiment"]["system_prompt"],
        targets=_TARGETS,
        scenario=scenario_desc,
        experiment_output_format=y["factor_experiment_output_format"],
    )
    user_prompt = _jinja(
        y["hypothesis2experiment"]["user_prompt"],
        targets=_TARGETS,
        target_hypothesis=fx.HYPOTHESIS_STR,
        hypothesis_and_feedback=hypothesis_and_feedback,
        function_lib_description=y["function_lib_description"],
        target_list=None,
        RAG=None,
        expression_duplication=None,
    )

    # NOTE: output di sini bukan skema tetap (key = nama faktor), jadi kita
    # hanya cek "JSON parseable + non-empty", tidak expected_keys.
    return run_case(
        group="proposal_feedback", case="construct",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
    )


def test_feedback():
    """
    Feedback: nilai hasil backtest faktor vs SOTA, beri saran iterasi berikut.
    Input:
      - hypothesis_text
      - task_details (list of dict dengan factor_name/description/...)
      - combined_result (tabel metrik Current vs SOTA)
    Output expected: JSON dengan keys Observations / Feedback for Hypothesis /
    New Hypothesis / Reasoning / Replace Best Result.
    """
    y = _factors()
    scenario_desc = fx.TRACE.scen.get_scenario_all_desc()

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
