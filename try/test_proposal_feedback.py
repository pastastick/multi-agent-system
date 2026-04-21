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

from .common import (
    load_yaml, run_case, PROMPT_PATHS,
    get_latent_backend, _extract_json, _save_log,
)
from .config import CONFIG
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


# ═════════════════════════════════════════════════════════════════════════
# Compare propose → construct: Text-only (TextMAS) vs Latent (LatentMAS)
#
# Mode A (text_only):
#   propose → JSON text → extract "hypothesis" → inject into construct prompt
#   construct → JSON text
#
# Mode B (latent, mirip backend/core/latent/latent_method.py):
#   propose: run(mode="kv_only", latent_steps=N)
#     → KV-cache (berisi prompt tokens + N latent virtual tokens)
#     → tidak ada text output
#   construct: run(mode="kv_and_text", past_key_values=propose_kv,
#                  latent_steps=N)
#     → latent_pass(N steps) dulu, lalu generate text dari KV gabungan
#     → seperti "judger" pada latent_method.py
#
# Kedua agent pakai SAMA system/user prompt; perbedaan hanya di mode.
# Mode B set `target_hypothesis=""` di construct user prompt supaya info
# hypothesis HANYA mengalir via KV-cache (paper-faithful Latent-MAS).
#
# Untuk fairness: temperature=0.0 (deterministik), same prompts, same seed.
# ═════════════════════════════════════════════════════════════════════════

def _build_propose_messages():
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
    return system_prompt, user_prompt


def _build_construct_messages(target_hypothesis: str):
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
        target_hypothesis=target_hypothesis,
        hypothesis_and_feedback=hypothesis_and_feedback,
        function_lib_description=y["function_lib_description"],
        target_list=None,
        RAG=None,
        expression_duplication=None,
    )
    return system_prompt, user_prompt


def test_propose_construct_compare(latent_steps: int = 2) -> dict:
    """
    Side-by-side comparison: construct output in text-only vs latent mode.

    Only construct's output matters (propose is just a context-builder).
    Returns a dict with both outputs + validity flags; also writes a
    side-by-side log file under outputs/proposal_feedback/.
    """
    import time
    import json as _json

    group = "proposal_feedback"
    case = f"compare_latent{latent_steps}_vs_text"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    expected_keys_construct = []   # construct output is nested JSON; don't enforce

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}")
    print(f"  latent_steps={latent_steps}  temp=0.0  log={log_path}")
    print("═" * 78)

    if CONFIG.dry_run:
        print("── SKIPPED (dry_run=True; this test needs real LLM) ──")
        return {
            "group": group, "case": case, "ok_format": True,
            "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0,
            "log_path": str(log_path),
        }

    backend = get_latent_backend()

    # Build prompts once.
    propose_sys, propose_usr = _build_propose_messages()
    ctor_sys_textA, ctor_usr_textA = None, None  # filled after propose_A
    ctor_sys_latentB, ctor_usr_latentB = _build_construct_messages(target_hypothesis="")

    # Shared deterministic sampling for fairness.
    temp = 0.0
    top_p = 1.0

    # ── MODE A: TEXT-ONLY PROPOSE → TEXT-ONLY CONSTRUCT ──────────────────
    print("\n── MODE A (text_only) : propose ──")
    t0 = time.time()
    propose_result_A = backend.build_messages_and_run(
        user_prompt=propose_usr,
        system_prompt=propose_sys,
        json_mode=True,
        mode="text_only",
        temperature=temp, top_p=top_p,
        role="propose",
    )
    propose_text_A = propose_result_A.text or ""
    propose_elapsed_A = round(time.time() - t0, 2)
    print(f"  elapsed={propose_elapsed_A}s  len={len(propose_text_A)}")

    # Extract hypothesis string from propose's JSON.
    propose_json_A = _extract_json(propose_text_A)
    if propose_json_A and "hypothesis" in propose_json_A:
        extracted_hypothesis = str(propose_json_A["hypothesis"])
    else:
        extracted_hypothesis = propose_text_A.strip()[:800]   # fallback

    ctor_sys_textA, ctor_usr_textA = _build_construct_messages(
        target_hypothesis=extracted_hypothesis
    )

    print("── MODE A (text_only) : construct ──")
    t0 = time.time()
    construct_result_A = backend.build_messages_and_run(
        user_prompt=ctor_usr_textA,
        system_prompt=ctor_sys_textA,
        json_mode=True,
        mode="text_only",
        temperature=temp, top_p=top_p,
        role="construct",
    )
    construct_text_A = construct_result_A.text or ""
    construct_elapsed_A = round(time.time() - t0, 2)
    construct_json_A = _extract_json(construct_text_A)
    print(f"  elapsed={construct_elapsed_A}s  len={len(construct_text_A)}  valid_json={construct_json_A is not None}")

    # ── MODE B: LATENT PROPOSE → LATENT+TEXT CONSTRUCT ───────────────────
    print(f"\n── MODE B (latent, m={latent_steps}) : propose (kv_only) ──")
    t0 = time.time()
    propose_result_B = backend.build_messages_and_run(
        user_prompt=propose_usr,
        system_prompt=propose_sys,
        mode="kv_only",
        latent_steps=latent_steps,
        temperature=temp, top_p=top_p,
        role="propose_latent",
    )
    kv_from_propose = propose_result_B.kv_cache
    propose_elapsed_B = round(time.time() - t0, 2)
    kv_len = 0
    try:
        from backend.llm.models import _past_length   # type: ignore[import-not-found]
        kv_len = _past_length(kv_from_propose)
    except Exception:
        pass
    print(f"  elapsed={propose_elapsed_B}s  kv_len={kv_len} tokens")

    print(f"── MODE B (latent, m={latent_steps}) : construct (kv_and_text, target_hypothesis='') ──")
    t0 = time.time()
    construct_result_B = backend.build_messages_and_run(
        user_prompt=ctor_usr_latentB,
        system_prompt=ctor_sys_latentB,
        json_mode=True,
        mode="kv_and_text",
        past_key_values=kv_from_propose,
        latent_steps=latent_steps,
        temperature=temp, top_p=top_p,
        role="construct_latent",
    )
    construct_text_B = construct_result_B.text or ""
    construct_elapsed_B = round(time.time() - t0, 2)
    construct_json_B = _extract_json(construct_text_B)
    print(f"  elapsed={construct_elapsed_B}s  len={len(construct_text_B)}  valid_json={construct_json_B is not None}")

    # ── SAVE SIDE-BY-SIDE LOG ──────────────────────────────────────────────
    lines = [
        f"# Compare propose→construct  (latent_steps={latent_steps})",
        f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# temperature=0.0, top_p=1.0 (deterministic)",
        "",
        "=" * 78,
        "MODE A — TEXT-ONLY",
        "=" * 78,
        f"[propose]  elapsed={propose_elapsed_A}s",
        "-" * 78,
        propose_text_A,
        "",
        f"[extracted hypothesis injected to construct]",
        "-" * 78,
        extracted_hypothesis,
        "",
        f"[construct]  elapsed={construct_elapsed_A}s  valid_json={construct_json_A is not None}",
        "-" * 78,
        construct_text_A,
        "",
        "=" * 78,
        f"MODE B — LATENT (m={latent_steps})",
        "=" * 78,
        f"[propose: kv_only]  elapsed={propose_elapsed_B}s  kv_len={kv_len}",
        "-" * 78,
        "(no text output — only KV-cache was produced)",
        "",
        f"[construct: kv_and_text, target_hypothesis=\"\"]  "
        f"elapsed={construct_elapsed_B}s  valid_json={construct_json_B is not None}",
        "-" * 78,
        construct_text_B,
        "",
        "=" * 78,
        "PARSED JSON — MODE A",
        "=" * 78,
        _json.dumps(construct_json_A, indent=2, ensure_ascii=False) if construct_json_A else "(parse failed)",
        "",
        "=" * 78,
        "PARSED JSON — MODE B",
        "=" * 78,
        _json.dumps(construct_json_B, indent=2, ensure_ascii=False) if construct_json_B else "(parse failed)",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n── LOG SAVED ── {log_path}")

    both_ok = (construct_json_A is not None) and (construct_json_B is not None)
    return {
        "group": group,
        "case": case,
        "latent_steps": latent_steps,
        "ok_format": both_ok,
        "mode_a": {
            "propose_elapsed_s": propose_elapsed_A,
            "construct_elapsed_s": construct_elapsed_A,
            "construct_valid_json": construct_json_A is not None,
            "construct_parsed": construct_json_A,
        },
        "mode_b": {
            "propose_elapsed_s": propose_elapsed_B,
            "construct_elapsed_s": construct_elapsed_B,
            "construct_valid_json": construct_json_B is not None,
            "construct_parsed": construct_json_B,
            "kv_len_tokens": kv_len,
        },
        "elapsed_s": round(
            propose_elapsed_A + construct_elapsed_A
            + propose_elapsed_B + construct_elapsed_B, 2,
        ),
        "log_path": str(log_path),
    }


CASES = {
    "propose": test_propose,
    "construct": test_construct,
    "feedback": test_feedback,
    "compare_latent_vs_text": test_propose_construct_compare,
}
