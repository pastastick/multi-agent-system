"""
kv_probe_v2: coder prompt minimization grid experiment (Qwen3-4B).

QUESTION
  Given that the construct KV-cache already carries scenario context (factor
  description, hypothesis, allowed operations), how minimal can the coder's own
  system+user prompt be while still producing a valid {"expr": "..."} output?

GRID (3 × 2 × 2 × 2 = 24 cells)
  sys_level   : "full"     — evolving_strategy_factor_implementation_v1_system
                             (~580 tok, full function library + scenario text)
              | "compact"  — evolving_strategy_coder_system_kv
                             (~150 tok, KV-aware, no function list re-stated)
              | "minimal"  — 3-line task instruction only
  user_style  : "detailed" — factor info + former_expression + execution_log
              | "brief"    — factor name + former_expression only
  shots       :  0         — no few-shot examples
              |  2         — 2 worked (wrong→fixed) expression examples prepended
  error_type  : "clean"    — former_expression syntactically valid but semantically wrong
              | "wrong_fn" — former_expression uses undefined function TS_SLOPE

SETUP
  1. Run construct once (kv_and_text mode) → kv_construct (cached for all cells).
  2. For each grid cell: run coder with past_key_values=kv_construct + cell's prompts.
  3. Measure: JSON valid? valid expr (len >= 10)? kv size before vs after.

KV INSPECTION NOTE
  Directly reading KV content (what information a token position encodes) is not
  feasible — KV stores rotated, linearly projected key/value tensors, not raw tokens.
  What IS measurable: KV shape (tokens accumulated per layer) via _kv_shape_report().
  Practical corruption tracing: run ablation (remove KV from specific agents) and
  measure output degradation — the step where removal causes biggest quality drop
  is where the critical context lives. See chain_ablation in test_multi_agent_kv.py.
  _kv_shape_report() is available in common.py for use in any test.

Haiku baseline: run the same prompts manually (without KV chaining) on Claude Haiku
  to compare reference performance. Grid design matches so results are directly
  comparable, though KV chaining is only available for local Qwen3-4B.

Pemakaian:
    python -m try.run --group kv_probe_v2 --case kv_probe_v2_compact_brief_s0_clean
    python -m try.run --group kv_probe_v2                     # semua 24 cells
    python -m try.run --group kv_probe_v2 --dry-run

Override env:
    TEST_MODEL=Qwen/Qwen3-4B TEST_LATENT_STEPS=10 python -m try.run --group kv_probe_v2
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, StrictUndefined

from .common import (
    load_yaml, PROMPT_PATHS, get_latent_backend, _extract_json, _save_log,
)
from .config import CONFIG
from . import fixtures as fx
from .test_coder_evaluator import JSON_ONLY_SUFFIX


# ─── Grid definition ─────────────────────────────────────────────────────────

_SYS_LEVELS = ["full", "compact", "minimal"]
_USER_STYLES = ["detailed", "brief"]
_SHOTS_LIST  = [0, 2]
_ERROR_TYPES = ["clean", "wrong_fn"]


def _cell_key(sl: str, us: str, sh: int, et: str) -> str:
    return f"kv_probe_v2_{sl}_{us}_s{sh}_{et}"


# ─── Shared prompts / config ──────────────────────────────────────────────────

_LATENT_STEPS = int(os.environ.get("TEST_LATENT_STEPS", "10"))

# Former expression variants for error_type
_CLEAN_EXPR = "RANK(TS_MEAN($return, 5))"
_CLEAN_EXEC_LOG = (
    "Execution OK. IC=0.005 — extremely low. "
    "Volume component missing: expression uses only price momentum without volume confirmation."
)

_WRONG_FN_EXPR = "TS_SLOPE($return, 20) * RANK(TS_PCTCHANGE($volume, 5))"
_WRONG_FN_EXEC_LOG = (
    "ExpressionError: Unknown function 'TS_SLOPE' at position 0. "
    "Available alternatives: TS_MEAN, EMA, DECAYLINEAR, REGBETA."
)

# Few-shot examples (2 shots)
_SHOT_EXAMPLES = """\
<examples>
Example 1 — fix undefined variable:
  BAD:  TS_MEAN($price, 10)      [error: $price is not a valid column]
  GOOD: TS_MEAN($close, 10)

Example 2 — fix unknown function:
  BAD:  ROLLING_STD($return, 20)  [error: unknown function ROLLING_STD]
  GOOD: TS_STD($return, 20)
</examples>

"""

# ─── YAML loaders ─────────────────────────────────────────────────────────────

_CODER_QA: dict | None = None
_FACTORS: dict | None = None


def _qa() -> dict:
    global _CODER_QA
    if _CODER_QA is None:
        _CODER_QA = load_yaml(PROMPT_PATHS["factors_coder_qa"])
    return _CODER_QA


def _fct() -> dict:
    global _FACTORS
    if _FACTORS is None:
        _FACTORS = load_yaml(PROMPT_PATHS["factors_prompts"])
    return _FACTORS


def _j2(tmpl: str, **kw: Any) -> str:
    return Environment(undefined=StrictUndefined).from_string(tmpl).render(**kw)


# ─── System prompt builders ───────────────────────────────────────────────────

def _build_sys(level: str) -> str:
    if level == "full":
        scen = fx.SCENARIO.get_scenario_all_desc(filtered_tag="feature")
        return _j2(_qa()["evolving_strategy_factor_implementation_v1_system"], scenario=scen)
    if level == "compact":
        return _qa()["evolving_strategy_coder_system_kv"]
    if level == "minimal":
        return (
            "Fix the factor expression. The factor context is already in your memory.\n"
            "Allowed variables: $open, $close, $high, $low, $volume, $return.\n"
            'Output ONLY this JSON on one line: {"expr": "YOUR_EXPRESSION"}'
        )
    raise ValueError(f"Unknown sys_level: {level!r}")


# ─── User prompt builders ─────────────────────────────────────────────────────

def _build_usr(style: str, shots: int, error_type: str) -> str:
    former_expr = _CLEAN_EXPR if error_type == "clean" else _WRONG_FN_EXPR
    exec_log = _CLEAN_EXEC_LOG if error_type == "clean" else _WRONG_FN_EXEC_LOG
    factor_info = fx.FACTOR_TASK.get_task_description()

    shot_block = _SHOT_EXAMPLES if shots >= 2 else ""

    if style == "detailed":
        usr = _j2(
            _qa()["evolving_strategy_factor_implementation_v2_user"],
            factor_information_str=factor_info,
            former_expression=former_expr,
            execution_log=exec_log,
            code_comment=None,
            queried_similar_error_knowledge=[],
            error_summary_critics=None,
            similar_successful_factor_description=None,
            similar_successful_expression=None,
            latest_attempt_to_latest_successful_execution=None,
        )
        return shot_block + usr + JSON_ONLY_SUFFIX
    if style == "brief":
        return (
            shot_block
            + f"Factor: {fx.FACTOR_TASK.factor_name}\n"
            + f"Former expression: {former_expr}\n"
            + f"Execution log: {exec_log}\n\n"
            + 'Output ONLY this JSON on one line: {"expr": "YOUR_EXPRESSION"}'
        )
    raise ValueError(f"Unknown user_style: {style!r}")


# ─── Construct KV setup (cached) ──────────────────────────────────────────────

_CONSTRUCT_KV: Any = None


def _get_construct_kv(backend):
    global _CONSTRUCT_KV
    if _CONSTRUCT_KV is not None:
        return _CONSTRUCT_KV

    print("\n[kv_probe_v2] Building construct KV (first call, will be cached)...")
    fy = _fct()
    scen = fx.SCENARIO.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
    hf_tmpl = fy.get("hypothesis_and_feedback", "")
    from types import SimpleNamespace
    short_trace = SimpleNamespace(scen=fx.SCENARIO, hist=fx.TRACE_HIST[-2:])
    hf = _j2(hf_tmpl, trace=short_trace) if hf_tmpl else ""

    ctor_sys = _j2(
        fy["hypothesis2experiment"]["system_prompt"],
        targets="factor", scenario=scen,
        experiment_output_format=fy["experiment_output_format"],
    )
    ctor_usr = _j2(
        fy["hypothesis2experiment"]["user_prompt"],
        targets="factor",
        target_hypothesis=fx.HYPOTHESIS_STR,
        hypothesis_and_feedback=hf,
        function_lib_description=fy.get("function_lib_description", ""),
        target_list=None, RAG=None, expression_duplication=None,
    )

    if CONFIG.dry_run:
        print("[kv_probe_v2] dry_run — skipping construct call, KV=None")
        return None

    t0 = time.time()
    r = backend.build_messages_and_run(
        user_prompt=ctor_usr, system_prompt=ctor_sys,
        json_mode=True, mode="kv_and_text",
        latent_steps=_LATENT_STEPS,
        temperature=0.7, top_p=0.95, role="kv_probe_construct",
    )
    elapsed = round(time.time() - t0, 2)
    kv_len = _kv_len(r.kv_cache)
    print(f"[kv_probe_v2] Construct done  elapsed={elapsed}s  kv_len={kv_len} tokens")
    _CONSTRUCT_KV = r.kv_cache
    return _CONSTRUCT_KV


def _kv_len(kv) -> int:
    try:
        from backend.llm.models import _past_length
        return _past_length(kv)
    except Exception:
        return -1


# ─── Single grid cell runner ──────────────────────────────────────────────────

def run_grid_cell(sys_level: str, user_style: str, shots: int, error_type: str) -> dict:
    key = _cell_key(sys_level, user_style, shots, error_type)
    group = "kv_probe_v2"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{key}_{ts}.txt"

    sys_p = _build_sys(sys_level)
    usr_p = _build_usr(user_style, shots, error_type)

    print(f"\n── [{group}] {key}")
    print(f"   sys={sys_level}  user={user_style}  shots={shots}  error={error_type}  log={log_path.name}")

    result: dict = {
        "group": group, "case": key,
        "sys_level": sys_level, "user_style": user_style,
        "shots": shots, "error_type": error_type,
        "json_mode": True, "ok_format": None,
        "valid_expr": False,
        "response": None, "parsed": None,
        "elapsed_s": 0.0, "log_path": str(log_path),
        "kv_len_before": -1, "kv_len_after": -1,
    }

    if CONFIG.dry_run:
        print("   (dry_run)")
        result["ok_format"] = True
        _save_log(log_path, sys_p, usr_p, "(dry_run)", result)
        return result

    backend = get_latent_backend(CONFIG, latent_steps_init=_LATENT_STEPS)
    kv = _get_construct_kv(backend)
    result["kv_len_before"] = _kv_len(kv)

    t0 = time.time()
    try:
        r = backend.build_messages_and_run(
            user_prompt=usr_p, system_prompt=sys_p,
            json_mode=True, mode="kv_and_text",
            past_key_values=kv,
            latent_steps=_LATENT_STEPS,
            temperature=0.5, top_p=0.95,
            role=f"coder_{key}",
        )
        response = r.text or ""
        result["kv_len_after"] = _kv_len(r.kv_cache)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        _save_log(log_path, sys_p, usr_p, f"ERROR: {result['error']}", result)
        return result

    result["elapsed_s"] = round(time.time() - t0, 2)
    result["response"] = response
    parsed = _extract_json(response)
    result["parsed"] = parsed

    if parsed and "expr" in parsed:
        expr = parsed["expr"]
        result["valid_expr"] = isinstance(expr, str) and len(expr) >= 10
        result["ok_format"] = result["valid_expr"]
        status = "✔" if result["valid_expr"] else "❌ short expr"
        print(f"   {status}  elapsed={result['elapsed_s']}s  "
              f"kv_before={result['kv_len_before']}  kv_after={result['kv_len_after']}")
        print(f"   expr: {expr[:80]!r}")
    else:
        result["ok_format"] = False
        print(f"   ❌ no valid JSON  elapsed={result['elapsed_s']}s  resp[:60]={response[:60]!r}")

    _save_log(log_path, sys_p, usr_p, response, result)
    return result


# ─── Grid summary printer ─────────────────────────────────────────────────────

def _print_grid_summary(results: list[dict]) -> None:
    print(f"\n{'═' * 78}")
    print("kv_probe_v2 GRID SUMMARY")
    print(f"{'─' * 78}")
    print(f"{'case':<48}  {'ok':>3}  {'expr':>5}  {'elapsed':>7}  kv_before→after")
    print(f"{'─' * 78}")
    for r in results:
        ok = "✔" if r.get("ok_format") else "❌"
        ve = "✔" if r.get("valid_expr") else "❌"
        el = f"{r.get('elapsed_s', 0):.1f}s"
        kb = r.get("kv_len_before", -1)
        ka = r.get("kv_len_after", -1)
        print(f"  {r['case']:<46}  {ok:>3}  {ve:>5}  {el:>7}  {kb}→{ka}")
    n_ok = sum(1 for r in results if r.get("ok_format"))
    print(f"{'─' * 78}")
    print(f"  passed: {n_ok}/{len(results)}")

    # Save structured JSON for external analysis
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / "kv_probe_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / f"_grid_summary_{ts}.json"
    slim = [
        {k: v for k, v in r.items() if k not in ("response", "parsed")}
        for r in results
    ]
    grid_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Grid JSON: {grid_path}")


# ─── Full grid runner (callable as one batch) ─────────────────────────────────

def run_kv_probe_v2_all() -> dict:
    """Run all 24 grid cells, print summary, return aggregate result."""
    combos = list(itertools.product(_SYS_LEVELS, _USER_STYLES, _SHOTS_LIST, _ERROR_TYPES))
    print(f"\n{'═' * 78}")
    print(f"▶ [kv_probe_v2] running {len(combos)} grid cells  latent_steps={_LATENT_STEPS}")
    print(f"{'═' * 78}")

    results = []
    for sl, us, sh, et in combos:
        r = run_grid_cell(sl, us, sh, et)
        results.append(r)

    _print_grid_summary(results)
    n_ok = sum(1 for r in results if r.get("ok_format"))
    total_elapsed = sum(r.get("elapsed_s", 0.0) for r in results)
    return {
        "group": "kv_probe_v2",
        "case": "kv_probe_v2_all",
        "ok_format": n_ok == len(combos),
        "elapsed_s": total_elapsed,
        "n_cells": len(combos), "n_ok": n_ok,
    }


# ─── CASES (used by run.py) ───────────────────────────────────────────────────

CASES: dict = {}

for _sl, _us, _sh, _et in itertools.product(_SYS_LEVELS, _USER_STYLES, _SHOTS_LIST, _ERROR_TYPES):
    _key = _cell_key(_sl, _us, _sh, _et)
    CASES[_key] = (lambda sl=_sl, us=_us, sh=_sh, et=_et:
                   run_grid_cell(sl, us, sh, et))

CASES["kv_probe_v2_all"] = run_kv_probe_v2_all
