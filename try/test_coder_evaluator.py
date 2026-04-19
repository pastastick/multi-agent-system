"""
Test LLM calls untuk coder (retry path) + evaluator bertahap.

Mapping YAML → case:
  - coder_retry          qa_prompts.yaml : evolving_strategy_factor_implementation_v1_system
                                           + evolving_strategy_factor_implementation_v2_user
                         + inline suffix (json_only_suffix) — output ekspresi faktor JSON
  - coder_error_summary  prompts.yaml    : evolving_strategy_error_summary_v2_system / _user
                         (tidak JSON; output kritik bullet)
  - code_feedback        qa_prompts.yaml : evaluator_code_feedback_v1_system / _user
                         (tidak JSON; output komentar 'comment N:')
  - output_format        prompts.yaml    : evaluator_output_format_system (+ user = df.info())
                         JSON {output_format_decision, output_format_feedback}
  - final_decision       prompts.yaml    : evaluator_final_decision_v1_system / _user
                         JSON {final_decision, final_feedback}

Inline suffix `json_only_suffix` untuk coder_retry di-copy dari
backend/factors/coder/evolving_strategy.py:502 agar tes merefleksikan
prompt "seperti di produksi".
"""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined

from .common import load_yaml, run_case, PROMPT_PATHS
from . import fixtures as fx


# ═════════════════════════════════════════════════════════════════════════
# Load YAML sekali
# ═════════════════════════════════════════════════════════════════════════

_QA_YAML: dict | None = None
_BASE_YAML: dict | None = None


def _qa() -> dict:
    global _QA_YAML
    if _QA_YAML is None:
        _QA_YAML = load_yaml(PROMPT_PATHS["factors_coder_qa"])
    return _QA_YAML


def _base() -> dict:
    global _BASE_YAML
    if _BASE_YAML is None:
        _BASE_YAML = load_yaml(PROMPT_PATHS["factors_coder_base"])
    return _BASE_YAML


def _jinja(template: str, **vars) -> str:
    return Environment(undefined=StrictUndefined).from_string(template).render(**vars)


# Inline suffix dari evolving_strategy.py:502 (di-copy verbatim)
JSON_ONLY_SUFFIX = (
    "\n\nOUTPUT INSTRUCTION: Respond with ONLY the raw JSON object "
    "on a single line. No explanation, no preamble, no analysis. "
    'Example: {"expr": "TS_STD($close, 20)"}'
)


# ═════════════════════════════════════════════════════════════════════════
# Test cases
# ═════════════════════════════════════════════════════════════════════════

def test_coder_retry():
    """
    Coder retry path: diberi former_expression + feedback → minta ekspresi baru.
    Output expected: JSON {"expr": "<factor_expression>"}.
    """
    y = _qa()
    system_prompt = _jinja(
        y["evolving_strategy_factor_implementation_v1_system"],
        scenario=fx.SCENARIO_DESC,
    )
    # user_prompt Jinja berisi conditional: queried_similar_error_knowledge,
    # similar_successful_*, latest_attempt_to_latest_successful_execution.
    # Kita set minimal agar template valid (kosong / None).
    user_prompt = _jinja(
        y["evolving_strategy_factor_implementation_v2_user"],
        factor_information_str=fx.FACTOR_TASK_DESCRIPTION,
        former_expression=fx.FORMER_EXPRESSION,
        former_feedback=fx.FORMER_FEEDBACK,
        queried_similar_error_knowledge=[],         # list kosong → loop skip
        error_summary_critics=None,
        similar_successful_factor_description=None,
        similar_successful_expression=None,
        latest_attempt_to_latest_successful_execution=None,
    )
    # Tambahkan suffix yang memaksa JSON-only (mirror produksi)
    user_prompt = user_prompt + JSON_ONLY_SUFFIX

    return run_case(
        group="coder_evaluator", case="coder_retry",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        expected_keys=["expr"],
    )


def test_coder_error_summary():
    """
    Error summary: ringkasan critics dari error sebelumnya.
    Output expected: teks ber-format 'critic 1: ... / critic 2: ...' (bukan JSON).
    """
    # NOTE: source backend pakai `implement_prompts` (coder/prompts.yaml),
    # namespace yang sama dengan key v2_system/_user-nya (lihat prompts.yaml).
    y = _base()
    system_prompt = _jinja(
        y["evolving_strategy_error_summary_v2_system"],
        scenario=fx.SCENARIO_DESC,
        factor_information_str=fx.FACTOR_TASK_DESCRIPTION,
        code_and_feedback=(
            "=====Code=====\n"
            f"{fx.FACTOR_CODE}\n"
            "=====Feedback=====\n"
            f"{fx.EXECUTION_FEEDBACK_FAIL}"
        ),
    )
    # user template mensyaratkan queried_similar_error_knowledge (bisa [])
    user_prompt = _jinja(
        y["evolving_strategy_error_summary_v2_user"],
        queried_similar_error_knowledge=[],
    )
    # Kalau rendering menghasilkan string kosong, isi fallback pendek agar
    # LLM tetap punya konteks.
    if not user_prompt.strip():
        user_prompt = (
            "Refer to the system prompt above. Provide critics addressing the "
            "root cause of the failure. No code, just short bullets."
        )

    return run_case(
        group="coder_evaluator", case="coder_error_summary",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=False,
    )


def test_code_feedback():
    """
    Code feedback (FactorCodeEvaluator): LLM review terhadap kode & value feedback.
    Output expected: teks dengan format 'comment 1: ... / comment 2: ...' atau
    'No comment found'. Tidak JSON.
    """
    y = _qa()
    system_prompt = _jinja(
        y["evaluator_code_feedback_v1_system"],
        scenario=fx.SCENARIO_DESC,
    )
    user_prompt = _jinja(
        y["evaluator_code_feedback_v1_user"],
        factor_information=fx.FACTOR_TASK_INFO,
        code=fx.FACTOR_CODE,
        execution_feedback=fx.EXECUTION_FEEDBACK_OK,
        value_feedback=fx.VALUE_FEEDBACK,
        gt_code=None,
    )

    return run_case(
        group="coder_evaluator", case="code_feedback",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=False,
    )


def test_output_format():
    """
    Output format evaluator: cek apakah dataframe info sesuai format faktor.
    Output expected: JSON {output_format_decision: bool, output_format_feedback: str}.
    """
    y = _base()
    system_prompt = _jinja(
        y["evaluator_output_format_system"],
        scenario=fx.SCENARIO_DESC,
    )
    # User prompt = string info df, di-mock (mirip keluaran gen_df.info()).
    user_prompt = (
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

    return run_case(
        group="coder_evaluator", case="output_format",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        expected_keys=["output_format_decision", "output_format_feedback"],
    )


def test_final_decision():
    """
    Final decision: menggabungkan execution/code/value feedback → keputusan akhir.
    Output expected: JSON {final_decision: bool, final_feedback: str}.
    """
    y = _base()
    system_prompt = _jinja(
        y["evaluator_final_decision_v1_system"],
        scenario=fx.SCENARIO_DESC,
    )
    user_prompt = _jinja(
        y["evaluator_final_decision_v1_user"],
        factor_information=fx.FACTOR_TASK_INFO,
        execution_feedback=fx.EXECUTION_FEEDBACK_OK,
        code_feedback=(
            "comment 1: Expression matches description.\n"
            "comment 2: Consider longer window for stability."
        ),
        value_feedback=fx.VALUE_FEEDBACK,
    )

    return run_case(
        group="coder_evaluator", case="final_decision",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        expected_keys=["final_decision", "final_feedback"],
    )


CASES = {
    "coder_retry": test_coder_retry,
    "coder_error_summary": test_coder_error_summary,
    "code_feedback": test_code_feedback,
    "output_format": test_output_format,
    "final_decision": test_final_decision,
}
