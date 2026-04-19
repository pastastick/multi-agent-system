"""
Test LLM calls untuk planning + evolution (mutation & crossover).

Semua prompt di sini dibaca dari YAML dan di-render dengan .format() style
(KUNCI: ini bukan Jinja2; placeholder-nya `{name}` biasa).

  - planning             pipeline/prompts/planning_prompts.yaml → system / user
  - mutation_detailed    pipeline/prompts/evolution_prompts.yaml → mutation.system / mutation.user
  - mutation_simple      pipeline/prompts/evolution_prompts.yaml → mutation.simple_user
  - crossover_detailed   pipeline/prompts/evolution_prompts.yaml → crossover.system / crossover.user
  - crossover_simple     pipeline/prompts/evolution_prompts.yaml → crossover.simple_user

Catatan: mutation.user dan crossover.user contain `{{...}}` escape-braces di
YAML sehingga JSON example-nya keluar dengan `{...}` tunggal di string akhir.
Kita perlakukan sebagai string Python .format() biasa.
"""

from __future__ import annotations

from .common import load_yaml, render_format, run_case, PROMPT_PATHS
from . import fixtures as fx


# ═════════════════════════════════════════════════════════════════════════
# Lazy-load YAML (dipakai lintas test)
# ═════════════════════════════════════════════════════════════════════════

_PLANNING_YAML: dict | None = None
_EVOLUTION_YAML: dict | None = None


def _planning() -> dict:
    global _PLANNING_YAML
    if _PLANNING_YAML is None:
        _PLANNING_YAML = load_yaml(PROMPT_PATHS["planning"])
    return _PLANNING_YAML


def _evolution() -> dict:
    global _EVOLUTION_YAML
    if _EVOLUTION_YAML is None:
        _EVOLUTION_YAML = load_yaml(PROMPT_PATHS["evolution"])
    return _EVOLUTION_YAML


# ═════════════════════════════════════════════════════════════════════════
# Test cases
# ═════════════════════════════════════════════════════════════════════════

def test_planning():
    """
    Planning: generate N exploration directions dari 1 initial direction.
    Output expected: JSON dengan key "directions" (list of N strings).
    """
    y = _planning()
    system_prompt = y["system"]
    user_prompt = render_format(
        y["user"],
        initial_direction=(
            "Eksplorasi faktor momentum-volume pada saham IDX dengan window pendek "
            "(≤5 hari) untuk menangkap short-term reversal pada high-beta names."
        ),
        n=4,
    )
    return run_case(
        group="planning_evolution", case="planning",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        expected_keys=["directions"],
    )


def test_mutation_detailed():
    """
    Mutation (detailed user): generate orthogonal strategy dari SATU parent.
    Output expected: JSON dengan new_hypothesis / exploration_direction /
    orthogonality_reason / expected_characteristics.
    """
    y = _evolution()["mutation"]
    system_prompt = y["system"]
    user_prompt = render_format(
        y["user"],
        parent_hypothesis=fx.PARENT_HYPOTHESIS,
        parent_factors=fx.PARENT_FACTORS_STR,
        parent_metrics=fx.PARENT_METRICS_STR,
        parent_feedback=fx.PARENT_FEEDBACK_STR,
    )
    return run_case(
        group="planning_evolution", case="mutation_detailed",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        expected_keys=[
            "new_hypothesis", "exploration_direction",
            "orthogonality_reason", "expected_characteristics",
        ],
    )


def test_mutation_simple():
    """
    Mutation (simple_user): output berupa TEKS bebas — satu hipotesis baru.
    Tidak JSON. Kita validasi sekadar non-empty.
    """
    y = _evolution()["mutation"]
    system_prompt = y["system"]
    user_prompt = render_format(
        y["simple_user"],
        parent_hypothesis=fx.PARENT_HYPOTHESIS,
        parent_factors=fx.PARENT_FACTORS_STR,
    )
    return run_case(
        group="planning_evolution", case="mutation_simple",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=False,
    )


def test_crossover_detailed():
    """
    Crossover (detailed user): combine MULTIPLE parents jadi hybrid strategy.
    Output expected: JSON dengan hybrid_hypothesis / fusion_logic /
    innovation_points / expected_benefits.
    """
    y = _evolution()["crossover"]
    system_prompt = y["system"]
    user_prompt = render_format(
        y["user"],
        parent_summaries=fx.PARENT_SUMMARIES_STR,
    )
    return run_case(
        group="planning_evolution", case="crossover_detailed",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        expected_keys=[
            "hybrid_hypothesis", "fusion_logic",
            "innovation_points", "expected_benefits",
        ],
    )


def test_crossover_simple():
    """
    Crossover (simple_user): output berupa TEKS bebas — satu hipotesis fusi.
    """
    y = _evolution()["crossover"]
    system_prompt = y["system"]
    user_prompt = render_format(
        y["simple_user"],
        parent_summaries=fx.PARENT_SUMMARIES_STR,
    )
    return run_case(
        group="planning_evolution", case="crossover_simple",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=False,
    )


CASES = {
    "planning": test_planning,
    "mutation_detailed": test_mutation_detailed,
    "mutation_simple": test_mutation_simple,
    "crossover_detailed": test_crossover_detailed,
    "crossover_simple": test_crossover_simple,
}
