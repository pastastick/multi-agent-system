"""
Multi-agent KV-cache experiment: chain 4 agent (Propose → Construct → Coder → Feedback)

Construct dan Coder adalah agent yang paling sering error karena format output
ketat: Construct perlu nested JSON 4-key, Coder perlu {"expr": "..."}.

Fitur:
  1. chain_ablation        — TextMAS vs LatentMAS 4-agent chain side-by-side
  2. kv_probe              — estimasi isi KV-cache via diagnostic injection
  3. knn_sweep             — cari knn_percentage optimal untuk Construct+Coder
  4. latent_steps_sweep    — trade-off depth vs latency
  5. construct_coder_format — fokus format compliance: berapa run yang valid?

Pemakaian:
    python -m try.run --group multi_agent_kv --case chain_ablation
    python -m try.run --group multi_agent_kv --case kv_probe
    python -m try.run --group multi_agent_kv --case knn_sweep
    python -m try.run --group multi_agent_kv --case latent_steps_sweep
    python -m try.run --group multi_agent_kv --case construct_coder_format
    python -m try.run --group multi_agent_kv   # semua

Dry-run (tidak load model, hanya print prompt):
    python -m try.run --group multi_agent_kv --dry-run

Override environment:
    TEST_MODEL=Qwen/Qwen3-4B TEST_DEVICE=cpu TEST_LATENT_STEPS=10 \\
        python -m try.run --group multi_agent_kv --case kv_probe
"""

from __future__ import annotations

import json as _json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, StrictUndefined

from .common import (
    load_yaml, run_case, PROMPT_PATHS,
    get_latent_backend, _extract_json, _save_log,
)
from .config import CONFIG
from . import fixtures as fx
from .test_coder_evaluator import JSON_ONLY_SUFFIX


# ─── helpers ─────────────────────────────────────────────────────────────────

def _jinja(template: str, **kw) -> str:
    return Environment(undefined=StrictUndefined).from_string(template).render(**kw)


# Direction hints untuk Propose — berdasarkan fungsi yang tersedia di function_lib.
# Setiap value menggantikan `factor_hypothesis_specification` di system prompt.
PROPOSE_DIRECTION_HINTS: dict[str, str] = {
    "momentum": (
        "Focus: price/return momentum — exploit trend continuation after sustained directional moves.\n"
        "Prefer: DELTA, EMA, WMA, MACD, DECAYLINEAR, TS_MEAN on $close or $vwap."
    ),
    "mean_reversion": (
        "Focus: mean-reversion — short-term overreaction that snaps back to fair value.\n"
        "Prefer: TS_ZSCORE, ZSCORE, BB_UPPER/BB_LOWER, PERCENTILE, TS_RANK on price deviation."
    ),
    "volatility": (
        "Focus: volatility regime — vol clustering, vol-of-vol, or risk-on/off signals.\n"
        "Prefer: TS_STD, TS_VAR, TS_MAD, BB bands on $close or return series."
    ),
    "microstructure": (
        "Focus: microstructure — intraday price efficiency signals using OHLCV proxies.\n"
        "Prefer: HIGHDAY, LOWDAY, TS_ARGMAX, TS_ARGMIN, PERCENTILE on $high/$low/$open/$close."
    ),
    "distributional": (
        "Focus: distributional shape — skewness/kurtosis as sentiment or crash-risk proxy.\n"
        "Prefer: TS_SKEW, TS_KURT, TS_QUANTILE, TS_MAD on return or volume series."
    ),
    "regression": (
        "Focus: regression-based alpha — residual returns unexplained by a market factor.\n"
        "Prefer: REGBETA, REGRESI, TS_CORR, TS_COVARIANCE with $close or $volume as reference."
    ),
}


def _factors() -> dict:
    return load_yaml(PROMPT_PATHS["factors_prompts"])


def _render_hf(trace, limit: int = 6) -> str:
    """Render hypothesis_and_feedback template (sama dengan test_proposal_feedback)."""
    y = _factors()
    if len(trace.hist) == 0:
        return "No previous hypothesis and feedback available since it's the first round."
    lt = SimpleNamespace(scen=trace.scen, hist=trace.hist[-limit:])
    return _jinja(y["hypothesis_and_feedback"], trace=lt)


def _build_propose_msgs(direction_hint: str | None = None) -> tuple[str, str]:
    y = _factors()
    trace = fx.TRACE
    scen_desc = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
    hf = _render_hf(trace)
    spec = direction_hint if direction_hint is not None else y["factor_hypothesis_specification"]
    sys = _jinja(
        y["hypothesis_gen"]["system_prompt"],
        targets="factor", scenario=scen_desc,
        hypothesis_output_format=y["hypothesis_output_format"],
        hypothesis_specification=spec,
    )
    usr = _jinja(
        y["hypothesis_gen"]["user_prompt"],
        targets="factor", hypothesis_and_feedback=hf,
        RAG=None, round=len(trace.hist),
    )
    return sys, usr


def _build_construct_msgs(target_hypothesis: str) -> tuple[str, str]:
    y = _factors()
    trace = fx.TRACE
    scen_desc = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
    hf = _render_hf(trace)
    sys = _jinja(
        y["hypothesis2experiment"]["system_prompt"],
        targets="factor", scenario=scen_desc,
        experiment_output_format=y["experiment_output_format"],
    )
    usr = _jinja(
        y["hypothesis2experiment"]["user_prompt"],
        targets="factor", target_hypothesis=target_hypothesis,
        hypothesis_and_feedback=hf,
        function_lib_description=y["function_lib_description"],
        target_list=None, RAG=None, expression_duplication=None,
    )
    return sys, usr


def _build_feedback_msgs() -> tuple[str, str]:
    y = _factors()
    scen_desc = fx.SCENARIO.get_scenario_all_desc()
    sys = _jinja(y["factor_feedback_generation"]["system"], scenario=scen_desc)
    usr = _jinja(
        y["factor_feedback_generation"]["user"],
        hypothesis_text=fx.HYPOTHESIS_DICT["hypothesis"],
        task_details=fx.TASK_DETAILS,
        combined_result=fx.COMBINED_RESULT_STR,
    )
    return sys, usr


_CODER_YAML: dict | None = None
_CODER_BASE_YAML: dict | None = None


def _coder_qa() -> dict:
    global _CODER_YAML
    if _CODER_YAML is None:
        _CODER_YAML = load_yaml(PROMPT_PATHS["factors_coder_qa"])
    return _CODER_YAML


def _coder_base() -> dict:
    global _CODER_BASE_YAML
    if _CODER_BASE_YAML is None:
        _CODER_BASE_YAML = load_yaml(PROMPT_PATHS["factors_coder_base"])
    return _CODER_BASE_YAML


def _build_coder_msgs(
    factor_info_str: str = "",
    former_expression: str = "",
    former_feedback: str = "",
) -> tuple[str, str]:
    """
    Bangun system+user prompt untuk Coder (FactorParsingStrategy.implement_one_task).

    Mode A (TextMAS): factor_info_str berisi deskripsi faktor dari construct output.
    Mode B (LatentMAS): semua string bisa kosong — konteks mengalir via KV dari Construct.

    Output coder yang benar: {"expr": "<valid_expression>"}
    """
    y = _coder_qa()
    scen_desc = fx.SCENARIO.get_scenario_all_desc(filtered_tag="feature")
    sys = _jinja(
        y["evolving_strategy_factor_implementation_v1_system"],
        scenario=scen_desc,
    )
    usr = _jinja(
        y["evolving_strategy_factor_implementation_v2_user"],
        factor_information_str=factor_info_str,
        former_expression=former_expression,
        former_feedback=former_feedback,
        execution_log=None,
        code_comment=None,
        queried_similar_error_knowledge=[],
        error_summary_critics=None,
        similar_successful_factor_description=None,
        similar_successful_expression=None,
        latest_attempt_to_latest_successful_execution=None,
    ) + JSON_ONLY_SUFFIX
    return sys, usr


def _extract_factor_info_from_construct(json_construct: dict | None) -> str:
    """
    Ekstrak string deskripsi faktor pertama dari output construct JSON.
    Dipakai Mode A: inject ke coder sebagai factor_information_str.
    Format: "Factor: <name>\nDescription: ...\nFormulation: ...\nExpression: ..."
    """
    if not json_construct:
        return fx.FACTOR_TASK.get_task_description()
    try:
        name, info = next(iter(json_construct.items()))
        parts = [f"Factor: {name}"]
        if "description" in info:
            parts.append(f"Description: {info['description']}")
        if "formulation" in info:
            parts.append(f"Formulation: {info['formulation']}")
        if "expression" in info:
            parts.append(f"Expression: {info['expression']}")
        return "\n".join(parts)
    except (StopIteration, AttributeError, TypeError):
        return fx.FACTOR_TASK.get_task_description()


def _has_correct_schema(parsed: dict | None) -> bool:
    """Cek apakah output construct memiliki nested schema yang benar."""
    if not parsed or not isinstance(parsed, dict):
        return False
    for v in parsed.values():
        if not isinstance(v, dict):
            return False
        if not all(k in v for k in ("description", "variables", "formulation", "expression")):
            return False
    return True


def _has_valid_expr(parsed: dict | None) -> bool:
    """Cek apakah output coder memiliki key 'expr' berisi expression valid."""
    if not parsed or "expr" not in parsed:
        return False
    expr = parsed.get("expr", "")
    return isinstance(expr, str) and len(expr) >= 10


def _kv_len(kv) -> int:
    try:
        from backend.llm.models import _past_length
        return _past_length(kv)
    except Exception:
        return -1


def _log_save(log_path: Path, sections: list[tuple[str, str]]) -> None:
    """Simpan multi-section log ke file. sections = [(title, content), ...]"""
    lines = [f"# Log: {log_path.name}", f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for title, content in sections:
        lines += ["=" * 78, title, "=" * 78, content, ""]
    log_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Test 1: Chain ablation (TextMAS vs LatentMAS) ───────────────────────────

def test_chain_ablation(latent_steps: int | None = None) -> dict:
    """
    Jalankan pipeline 4-agent dalam 2 mode, bandingkan output:

    Mode A — TextMAS (text_only):
        Propose(text) → Construct(text, hyp injected) → Coder(text, factor injected) → Feedback(text)
        Setiap agent hanya melihat konteks via teks yang diinjek secara eksplisit.

    Mode B — LatentMAS (KV chain):
        Propose(kv_only) → Construct(kv+text, hyp="") → Coder(kv+text, factor="") → Feedback(kv+text)
        Konteks mengalir via KV-cache; masing-masing agent menerima KV dari agent sebelumnya.

    Construct dan Coder diuji bersama karena keduanya paling sering error akibat
    format output yang ketat (Construct: nested 4-key JSON; Coder: {"expr": "..."}).
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "40"))

    group, case = "multi_agent_kv", f"chain_ablation_m{latent_steps}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}")
    print(f"  latent_steps={latent_steps}  dry_run={CONFIG.dry_run}  log={log_path}")
    print("═" * 78)

    if CONFIG.dry_run:
        prop_sys, prop_usr = _build_propose_msgs()
        ctor_sys, ctor_usr = _build_construct_msgs(fx.HYPOTHESIS_STR)
        coder_sys, coder_usr = _build_coder_msgs(fx.FACTOR_TASK.get_task_description())
        fb_sys, fb_usr = _build_feedback_msgs()
        print("── PROPOSE system ──"); print(prop_sys[:600])
        print("── PROPOSE user ──"); print(prop_usr[:400])
        print("── CONSTRUCT system ──"); print(ctor_sys[:600])
        print("── CODER system ──"); print(coder_sys[:400])
        print("── CODER user (Mode A, factor injected) ──"); print(coder_usr[:400])
        print("── FEEDBACK system ──"); print(fb_sys[:400])
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()
    temp, top_p = 0.7, 0.95

    prop_sys, prop_usr = _build_propose_msgs()
    ctor_sys_latent, ctor_usr_latent = _build_construct_msgs(target_hypothesis="")
    coder_sys_latent, coder_usr_latent = _build_coder_msgs()  # semua kosong — andalkan KV
    fb_sys, fb_usr = _build_feedback_msgs()

    # ── MODE A: TextMAS ───────────────────────────────────────────────────────
    print("\n── MODE A (TextMAS: text_only × 4) ──")
    t0 = time.time()

    print("  [A1] Propose...")
    r_A1 = backend.build_messages_and_run(
        user_prompt=prop_usr, system_prompt=prop_sys,
        json_mode=True, mode="text_only",
        temperature=temp, top_p=top_p, role="propose_A",
    )
    text_A1 = r_A1.text or ""
    json_A1 = _extract_json(text_A1)
    hyp_text = str(json_A1.get("hypothesis", text_A1[:300])) if json_A1 else text_A1[:300]
    elapsed_A1 = round(time.time() - t0, 2)
    print(f"    elapsed={elapsed_A1}s  valid_json={json_A1 is not None}  hypothesis_len={len(hyp_text)}")

    ctor_sys_A, ctor_usr_A = _build_construct_msgs(target_hypothesis=hyp_text)

    t1 = time.time()
    print("  [A2] Construct...")
    r_A2 = backend.build_messages_and_run(
        user_prompt=ctor_usr_A, system_prompt=ctor_sys_A,
        json_mode=True, mode="text_only",
        temperature=temp, top_p=top_p, role="construct_A",
    )
    text_A2 = r_A2.text or ""
    json_A2 = _extract_json(text_A2)
    elapsed_A2 = round(time.time() - t1, 2)
    schema_A2 = _has_correct_schema(json_A2)
    print(f"    elapsed={elapsed_A2}s  valid_json={json_A2 is not None}  correct_schema={schema_A2}")

    # Inject factor info dari construct output ke coder (TextMAS: eksplisit via teks)
    factor_info_A = _extract_factor_info_from_construct(json_A2)
    coder_sys_A, coder_usr_A = _build_coder_msgs(factor_info_str=factor_info_A)

    t2 = time.time()
    print("  [A3] Coder...")
    r_A3 = backend.build_messages_and_run(
        user_prompt=coder_usr_A, system_prompt=coder_sys_A,
        json_mode=True, mode="text_only",
        temperature=0.5, top_p=top_p, role="coder_A",  # temp lebih rendah untuk presisi ekspresi
    )
    text_A3 = r_A3.text or ""
    json_A3 = _extract_json(text_A3)
    elapsed_A3 = round(time.time() - t2, 2)
    expr_A3 = _has_valid_expr(json_A3)
    print(f"    elapsed={elapsed_A3}s  valid_json={json_A3 is not None}  valid_expr={expr_A3}")
    if json_A3:
        print(f"    expr: {str(json_A3.get('expr', ''))[:100]!r}")

    t3 = time.time()
    print("  [A4] Feedback...")
    r_A4 = backend.build_messages_and_run(
        user_prompt=fb_usr, system_prompt=fb_sys,
        json_mode=True, mode="text_only",
        temperature=temp, top_p=top_p, role="feedback_A",
    )
    text_A4 = r_A4.text or ""
    json_A4 = _extract_json(text_A4)
    elapsed_A4 = round(time.time() - t3, 2)
    print(f"    elapsed={elapsed_A4}s  valid_json={json_A4 is not None}")

    # ── MODE B: LatentMAS ─────────────────────────────────────────────────────
    print(f"\n── MODE B (LatentMAS: kv_only → kv+text × 3, m={latent_steps}) ──")
    t4 = time.time()

    print("  [B1] Propose (kv_only — no text output)...")
    r_B1 = backend.build_messages_and_run(
        user_prompt=prop_usr, system_prompt=prop_sys,
        mode="kv_only", latent_steps=latent_steps,
        temperature=temp, top_p=top_p, role="propose_B",
    )
    kv_from_propose = r_B1.kv_cache
    kv_len_B1 = _kv_len(kv_from_propose)
    elapsed_B1 = round(time.time() - t4, 2)
    print(f"    elapsed={elapsed_B1}s  kv_len={kv_len_B1} tokens  (no text output)")

    t5 = time.time()
    print("  [B2] Construct (kv_and_text, hypothesis=\"\")...")
    r_B2 = backend.build_messages_and_run(
        user_prompt=ctor_usr_latent, system_prompt=ctor_sys_latent,
        json_mode=True, mode="kv_and_text",
        past_key_values=kv_from_propose, latent_steps=latent_steps,
        temperature=temp, top_p=top_p, role="construct_B",
    )
    text_B2 = r_B2.text or ""
    json_B2 = _extract_json(text_B2)
    kv_from_construct = r_B2.kv_cache
    kv_len_B2 = _kv_len(kv_from_construct)
    elapsed_B2 = round(time.time() - t5, 2)
    schema_B2 = _has_correct_schema(json_B2)
    print(f"    elapsed={elapsed_B2}s  valid_json={json_B2 is not None}  correct_schema={schema_B2}  kv_len={kv_len_B2}")

    t6 = time.time()
    # Coder menerima KV dari Construct: sudah berisi hypothesis + factor expressions.
    # factor_info kosong — semua konteks via KV. Coder harus bisa generate expr yang relevan.
    print("  [B3] Coder (kv_and_text, factor_info=\"\")...")
    r_B3 = backend.build_messages_and_run(
        user_prompt=coder_usr_latent, system_prompt=coder_sys_latent,
        json_mode=True, mode="kv_and_text",
        past_key_values=kv_from_construct, latent_steps=latent_steps,
        temperature=0.5, top_p=top_p, role="coder_B",
    )
    text_B3 = r_B3.text or ""
    json_B3 = _extract_json(text_B3)
    kv_from_coder = r_B3.kv_cache
    kv_len_B3 = _kv_len(kv_from_coder)
    elapsed_B3 = round(time.time() - t6, 2)
    expr_B3 = _has_valid_expr(json_B3)
    print(f"    elapsed={elapsed_B3}s  valid_json={json_B3 is not None}  valid_expr={expr_B3}  kv_len={kv_len_B3}")
    if json_B3:
        print(f"    expr: {str(json_B3.get('expr', ''))[:100]!r}")

    t7 = time.time()
    print("  [B4] Feedback (kv_and_text)...")
    r_B4 = backend.build_messages_and_run(
        user_prompt=fb_usr, system_prompt=fb_sys,
        json_mode=True, mode="kv_and_text",
        past_key_values=kv_from_coder, latent_steps=latent_steps,
        temperature=temp, top_p=top_p, role="feedback_B",
    )
    text_B4 = r_B4.text or ""
    json_B4 = _extract_json(text_B4)
    elapsed_B4 = round(time.time() - t7, 2)
    print(f"    elapsed={elapsed_B4}s  valid_json={json_B4 is not None}")

    # ── Ringkasan ──────────────────────────────────────────────────────────────
    total_A = elapsed_A1 + elapsed_A2 + elapsed_A3 + elapsed_A4
    total_B = elapsed_B1 + elapsed_B2 + elapsed_B3 + elapsed_B4
    print("\n── RINGKASAN ──")
    print(f"  TextMAS  : propose={elapsed_A1}s  construct={elapsed_A2}s  coder={elapsed_A3}s  fb={elapsed_A4}s  total={total_A:.2f}s")
    print(f"  LatentMAS: propose={elapsed_B1}s  construct={elapsed_B2}s  coder={elapsed_B3}s  fb={elapsed_B4}s  total={total_B:.2f}s")
    print(f"  Construct schema OK: A={schema_A2}  B={schema_B2}")
    print(f"  Coder expr   OK: A={expr_A3}  B={expr_B3}")
    print(f"  KV chain: propose→{kv_len_B1}  construct→{kv_len_B2}  coder→{kv_len_B3} tokens")

    # ── Simpan log ─────────────────────────────────────────────────────────────
    _log_save(log_path, [
        (f"CONFIG  latent_steps={latent_steps}  temp={temp}  top_p={top_p}", ""),
        ("MODE A — TextMAS — [A1] PROPOSE output", text_A1),
        (f"MODE A — TextMAS — [A2] CONSTRUCT output  schema_ok={schema_A2}", text_A2),
        (f"MODE A — TextMAS — [A3] CODER output  valid_expr={expr_A3}", text_A3),
        (f"MODE A — TextMAS — [A4] FEEDBACK output  json_ok={json_A4 is not None}", text_A4),
        (f"MODE B — LatentMAS — [B1] PROPOSE  kv_len={kv_len_B1} tokens", "(kv_only — no text)"),
        (f"MODE B — LatentMAS — [B2] CONSTRUCT output  schema_ok={schema_B2}  kv_len={kv_len_B2}", text_B2),
        (f"MODE B — LatentMAS — [B3] CODER output  valid_expr={expr_B3}  kv_len={kv_len_B3}", text_B3),
        (f"MODE B — LatentMAS — [B4] FEEDBACK output  json_ok={json_B4 is not None}", text_B4),
        ("PARSED — MODE A construct", _json.dumps(json_A2, indent=2, ensure_ascii=False) if json_A2 else "(parse failed)"),
        ("PARSED — MODE A coder", _json.dumps(json_A3, indent=2, ensure_ascii=False) if json_A3 else "(parse failed)"),
        ("PARSED — MODE B construct", _json.dumps(json_B2, indent=2, ensure_ascii=False) if json_B2 else "(parse failed)"),
        ("PARSED — MODE B coder", _json.dumps(json_B3, indent=2, ensure_ascii=False) if json_B3 else "(parse failed)"),
        ("SUMMARY", (
            f"TextMAS   total={total_A:.2f}s  construct_schema={schema_A2}  coder_expr={expr_A3}\n"
            f"LatentMAS total={total_B:.2f}s  construct_schema={schema_B2}  coder_expr={expr_B3}\n"
            f"KV chain: propose={kv_len_B1}tok → construct={kv_len_B2}tok → coder={kv_len_B3}tok"
        )),
    ])
    print(f"\n── LOG SAVED ── {log_path}")

    both_ok = (schema_A2 and expr_A3) and (schema_B2 and expr_B3)
    return {
        "group": group, "case": case, "latent_steps": latent_steps,
        "ok_format": both_ok, "elapsed_s": round(total_A + total_B, 2),
        "log_path": str(log_path),
        "mode_a": {"propose_valid": json_A1 is not None, "construct_schema": schema_A2,
                   "coder_valid_expr": expr_A3, "feedback_valid": json_A4 is not None,
                   "total_s": total_A},
        "mode_b": {"propose_kv_len": kv_len_B1, "construct_schema": schema_B2,
                   "construct_kv_len": kv_len_B2, "coder_valid_expr": expr_B3,
                   "coder_kv_len": kv_len_B3, "feedback_valid": json_B4 is not None,
                   "total_s": total_B},
        "parsed": {"A_construct": json_A2, "A_coder": json_A3,
                   "B_construct": json_B2, "B_coder": json_B3},
    }


# ─── Test 2: KV-cache content probe ──────────────────────────────────────────

def test_kv_probe(latent_steps: int | None = None) -> dict:
    """
    Estimasi isi KV-cache dengan injeksi pertanyaan diagnostik.

    Mekanisme:
      1. Jalankan Propose dalam kv_only mode → bangun KV
      2. Kirim "probe question" menggunakan KV tersebut (mode kv_and_text):
         "Summarize in ONE sentence the key market signal you are exploring."
      3. Bandingkan probe response dengan output text-mode propose
      4. Overlap antara keduanya → ukuran seberapa banyak KV merepresentasikan
         konteks asli propose

    Ini cara tidak langsung untuk "membaca" isi latent KV tanpa mengekstrak
    raw tensor (yang tidak interpretable). Model sendiri yang mengeksternalisasi
    apa yang ada di KV-nya melalui bahasa.
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    group, case = "multi_agent_kv", f"kv_probe_m{latent_steps}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    # Probe prompt: minimal, tidak membocorkan konteks apapun dari luar
    PROBE_SYS = "You are a market research assistant."
    PROBE_USR_TEMPLATE = (
        "Based on your context, complete these questions in ONE sentence each:\n"
        "1. What market phenomenon or signal am I exploring?\n"
        "2. What data features (e.g. $close, $volume) are most relevant?\n"
        "3. What time horizon is involved?\n"
        "Answer as a JSON: {{\"signal\": \"...\", \"features\": \"...\", \"horizon\": \"...\"}}"
    )

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  latent_steps={latent_steps}")
    print("═" * 78)

    if CONFIG.dry_run:
        prop_sys, prop_usr = _build_propose_msgs()
        print("── PROPOSE system ──"); print(prop_sys[:600])
        print("── PROBE user ──"); print(PROBE_USR_TEMPLATE)
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()
    temp, top_p = 0.7, 0.95

    prop_sys, prop_usr = _build_propose_msgs()

    # ── Step 1: Propose text_only (baseline referensi) ────────────────────────
    print("── [Ref] Propose text_only (untuk referensi) ──")
    t0 = time.time()
    r_ref = backend.build_messages_and_run(
        user_prompt=prop_usr, system_prompt=prop_sys,
        json_mode=True, mode="text_only",
        temperature=temp, top_p=top_p, role="propose_ref",
    )
    text_ref = r_ref.text or ""
    json_ref = _extract_json(text_ref)
    elapsed_ref = round(time.time() - t0, 2)
    hyp_ref = str(json_ref.get("hypothesis", "")) if json_ref else ""
    print(f"  elapsed={elapsed_ref}s  hypothesis_preview: {hyp_ref[:120]!r}")

    # ── Step 2: Propose kv_only → bangun KV ──────────────────────────────────
    print(f"── [KV] Propose kv_only (m={latent_steps}) ──")
    t1 = time.time()
    r_kv = backend.build_messages_and_run(
        user_prompt=prop_usr, system_prompt=prop_sys,
        mode="kv_only", latent_steps=latent_steps,
        temperature=temp, top_p=top_p, role="propose_kv",
    )
    kv = r_kv.kv_cache
    kv_len = _kv_len(kv)
    elapsed_kv = round(time.time() - t1, 2)
    print(f"  elapsed={elapsed_kv}s  kv_len={kv_len} tokens")

    # ── Step 3: Probe menggunakan KV tersebut ─────────────────────────────────
    print("── [Probe] Inject diagnostic question into KV ──")
    t2 = time.time()
    r_probe = backend.build_messages_and_run(
        user_prompt=PROBE_USR_TEMPLATE, system_prompt=PROBE_SYS,
        json_mode=True, mode="kv_and_text",
        past_key_values=kv, latent_steps=latent_steps,
        temperature=0.3, top_p=0.95, role="kv_probe",  # temp rendah untuk deterministic
    )
    text_probe = r_probe.text or ""
    json_probe = _extract_json(text_probe)
    elapsed_probe = round(time.time() - t2, 2)
    print(f"  elapsed={elapsed_probe}s  valid_json={json_probe is not None}")
    if json_probe:
        print(f"  signal  : {json_probe.get('signal', '?')!r}")
        print(f"  features: {json_probe.get('features', '?')!r}")
        print(f"  horizon : {json_probe.get('horizon', '?')!r}")
    print(f"\n  Reference hypothesis: {hyp_ref!r}")

    # ── Simpan log ─────────────────────────────────────────────────────────────
    _log_save(log_path, [
        (f"CONFIG  latent_steps={latent_steps}  kv_len={kv_len}", ""),
        ("REFERENCE — Propose text_only (baseline)", text_ref),
        (f"KV BUILD — Propose kv_only  elapsed={elapsed_kv}s", f"(no text output; kv_len={kv_len} tokens)"),
        ("PROBE QUESTION injected into KV", PROBE_USR_TEMPLATE),
        (f"PROBE RESPONSE  elapsed={elapsed_probe}s  valid_json={json_probe is not None}", text_probe),
        ("ANALYSIS — Did KV encode the right context?", (
            f"Reference hypothesis: {hyp_ref}\n\n"
            f"Probe signal       : {(json_probe or {}).get('signal', '(parse failed)')}\n"
            f"Probe features     : {(json_probe or {}).get('features', '')}\n"
            f"Probe horizon      : {(json_probe or {}).get('horizon', '')}\n\n"
            "Interpretation: jika probe signal mirip dengan reference hypothesis,\n"
            "KV berhasil merepresentasikan konteks propose secara latent."
        )),
    ])
    print(f"\n── LOG SAVED ── {log_path}")

    return {
        "group": group, "case": case, "latent_steps": latent_steps,
        "ok_format": json_probe is not None,
        "elapsed_s": round(elapsed_ref + elapsed_kv + elapsed_probe, 2),
        "log_path": str(log_path),
        "reference_hypothesis": hyp_ref,
        "probe_result": json_probe,
        "kv_len_tokens": kv_len,
        "parsed": json_probe,
    }


# ─── Test 3: KNN percentage sweep ────────────────────────────────────────────

def test_knn_sweep(latent_steps: int | None = None) -> dict:
    """
    Sweep knn_percentage dari 0.2 ke 1.0, ukur pengaruh terhadap Construct DAN Coder.

    Construct dan Coder adalah dua agent paling sensitif terhadap kualitas KV:
    - Construct perlu nested JSON 4-key: gagal → flat dict atau parse error
    - Coder perlu {"expr": "..."}: gagal → missing key atau expression tidak valid

    Untuk setiap nilai knn_percentage:
      - Propose(kv_only) → bangun KV base sekali
      - Construct(kv_and_text, knn=p) → test format nested JSON
      - Coder(kv_and_text dari Construct, knn=p) → test format {"expr": "..."}

    Tujuan: cari knn_percentage minimum yang masih menghasilkan KEDUA output valid.
    Terlalu rendah (0.2) = konteks hilang; terlalu tinggi (1.0) = OOM risk di GPU.
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    KNN_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]
    group, case = "multi_agent_kv", f"knn_sweep_m{latent_steps}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  latent_steps={latent_steps}  knn_values={KNN_VALUES}")
    print("═" * 78)

    if CONFIG.dry_run:
        print("  dry_run=True: hanya menampilkan konfigurasi sweep")
        print(f"  knn_values={KNN_VALUES}  latent_steps={latent_steps}")
        prop_sys, prop_usr = _build_propose_msgs()
        ctor_sys, ctor_usr = _build_construct_msgs(target_hypothesis="")
        print("── PROPOSE system (preview) ──"); print(prop_sys[:400])
        print("── CONSTRUCT system (preview) ──"); print(ctor_sys[:400])
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()
    temp, top_p = 0.7, 0.95
    prop_sys, prop_usr = _build_propose_msgs()
    ctor_sys, ctor_usr = _build_construct_msgs(target_hypothesis="")

    # Bangun satu KV dari propose sekali — reuse untuk semua knn experiments
    print(f"\n── Build propose KV (dipakai ulang untuk semua sweep) ──")
    t0 = time.time()
    r_propose = backend.build_messages_and_run(
        user_prompt=prop_usr, system_prompt=prop_sys,
        mode="kv_only", latent_steps=latent_steps,
        temperature=temp, top_p=top_p, role="propose_knn_base",
    )
    base_kv = r_propose.kv_cache
    kv_len_base = _kv_len(base_kv)
    elapsed_propose = round(time.time() - t0, 2)
    print(f"  elapsed={elapsed_propose}s  kv_len={kv_len_base} tokens")

    coder_sys, coder_usr = _build_coder_msgs()  # kosong — andalkan KV dari construct

    # Sweep knn_percentage
    results_table: list[dict] = []
    log_sections: list[tuple[str, str]] = [
        (f"CONFIG  latent_steps={latent_steps}  kv_len_base={kv_len_base}", ""),
        (f"Propose KV build  elapsed={elapsed_propose}s", f"kv_len={kv_len_base} tokens"),
    ]

    original_knn = backend._engine.knn_percentage
    original_knn_enabled = backend._engine.knn_enabled

    try:
        for knn_p in KNN_VALUES:
            print(f"\n── knn_percentage={knn_p:.1f} ──")
            backend._engine.knn_percentage = knn_p
            backend._engine.knn_enabled = True

            # ─ Construct ─
            t1 = time.time()
            r_ctor = backend.build_messages_and_run(
                user_prompt=ctor_usr, system_prompt=ctor_sys,
                json_mode=True, mode="kv_and_text",
                past_key_values=base_kv, latent_steps=latent_steps,
                temperature=temp, top_p=top_p, role=f"construct_knn{int(knn_p*100)}",
            )
            text_ctor = r_ctor.text or ""
            kv_from_ctor = r_ctor.kv_cache
            elapsed_ctor = round(time.time() - t1, 2)
            json_ctor = _extract_json(text_ctor)
            schema_ok = _has_correct_schema(json_ctor)
            kv_len_after = int(kv_len_base * knn_p)  # estimasi token setelah filter
            print(f"  [construct] tokens≈{kv_len_after}/{kv_len_base}  "
                  f"schema={schema_ok}  elapsed={elapsed_ctor}s")

            # ─ Coder (menerima KV dari construct) ─
            t2 = time.time()
            r_coder = backend.build_messages_and_run(
                user_prompt=coder_usr, system_prompt=coder_sys,
                json_mode=True, mode="kv_and_text",
                past_key_values=kv_from_ctor, latent_steps=latent_steps,
                temperature=0.5, top_p=top_p, role=f"coder_knn{int(knn_p*100)}",
            )
            text_coder = r_coder.text or ""
            elapsed_coder = round(time.time() - t2, 2)
            json_coder = _extract_json(text_coder)
            expr_ok = _has_valid_expr(json_coder)
            print(f"  [coder]     valid_expr={expr_ok}  elapsed={elapsed_coder}s")
            if json_coder:
                print(f"  expr: {str(json_coder.get('expr', ''))[:80]!r}")

            row = {
                "knn_percentage": knn_p,
                "tokens_kept_est": kv_len_after,
                "construct_valid_json": json_ctor is not None,
                "construct_schema": schema_ok,
                "coder_valid_json": json_coder is not None,
                "coder_valid_expr": expr_ok,
                "both_ok": schema_ok and expr_ok,
                "construct_elapsed_s": elapsed_ctor,
                "coder_elapsed_s": elapsed_coder,
                "total_elapsed_s": elapsed_ctor + elapsed_coder,
            }
            results_table.append(row)
            log_sections.append((
                f"knn={knn_p:.1f}  tokens≈{kv_len_after}  "
                f"construct_schema={schema_ok}  coder_expr={expr_ok}  "
                f"construct={elapsed_ctor}s  coder={elapsed_coder}s",
                f"CONSTRUCT OUTPUT:\n{text_ctor}\n\nCODER OUTPUT:\n{text_coder}",
            ))
    finally:
        backend._engine.knn_percentage = original_knn
        backend._engine.knn_enabled = original_knn_enabled

    # ── Tabel ringkasan ───────────────────────────────────────────────────────
    print("\n── TABEL HASIL KNN SWEEP (Construct + Coder) ──")
    header = f"{'knn%':>6} | {'tokens':>7} | {'ctor_schema':>11} | {'coder_expr':>10} | {'both_ok':>7} | {'total_s':>8}"
    print(header)
    print("─" * len(header))
    for r in results_table:
        print(f"{r['knn_percentage']:>6.1f} | {r['tokens_kept_est']:>7} | "
              f"{str(r['construct_schema']):>11} | {str(r['coder_valid_expr']):>10} | "
              f"{str(r['both_ok']):>7} | {r['total_elapsed_s']:>7.2f}s")

    table_str = header + "\n"
    for r in results_table:
        table_str += (f"{r['knn_percentage']:>6.1f} | {r['tokens_kept_est']:>7} | "
                      f"{str(r['construct_schema']):>11} | {str(r['coder_valid_expr']):>10} | "
                      f"{str(r['both_ok']):>7} | {r['total_elapsed_s']:>7.2f}s\n")

    log_sections.append(("SUMMARY TABLE", table_str))
    _log_save(log_path, log_sections)
    print(f"\n── LOG SAVED ── {log_path}")

    both_ok_rows = [r for r in results_table if r["both_ok"]]
    min_working_knn = min((r["knn_percentage"] for r in both_ok_rows), default=None)
    if min_working_knn is not None:
        print(f"\n  Rekomendasi: knn_percentage minimal yang valid KEDUANYA = {min_working_knn:.1f}")
    else:
        print("\n  ⚠ Tidak ada knn_percentage yang menghasilkan construct+coder valid bersamaan.")

    return {
        "group": group, "case": case, "latent_steps": latent_steps,
        "ok_format": bool(both_ok_rows),
        "elapsed_s": round(elapsed_propose + sum(r["total_elapsed_s"] for r in results_table), 2),
        "log_path": str(log_path),
        "table": results_table,
        "min_working_knn_percentage": min_working_knn,
        "parsed": None,
    }


# ─── Test 4: Latent steps sweep ──────────────────────────────────────────────

def test_latent_steps_sweep() -> dict:
    """
    Sweep latent_steps × propose_direction: ukur trade-off kedalaman reasoning,
    latency, dan pengaruh direction hint terhadap kualitas output construct.

    Untuk setiap (direction, latent_steps):
      - Propose(kv_only, direction_hint, m=n) → Construct(kv_and_text, m=n)
      - Catat: JSON validity, schema correctness, elapsed_s, kv_len

    Konfigurasi user:
      STEPS_VALUES    — nilai latent_steps yang di-sweep
      PROPOSE_DIRS    — subset key dari PROPOSE_DIRECTION_HINTS yang ingin diuji;
                        set ke None untuk memakai default (tanpa direction hint)
    """
    STEPS_VALUES: list[int] = [10, 20, 40, 80]
    # ── Konfigurasi direction — ubah sesuai kebutuhan ─────────────────────────
    # Pilihan: "momentum", "mean_reversion", "volatility",
    #          "microstructure", "distributional", "regression", None (default)
    PROPOSE_DIRS: list[str | None] = ["momentum", "mean_reversion", "volatility"]
    # ─────────────────────────────────────────────────────────────────────────

    group, case = "multi_agent_kv", "latent_steps_sweep"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  steps={STEPS_VALUES}  directions={PROPOSE_DIRS}")
    print("  ⚠ PERINGATAN: m≥20 sangat lambat di CPU. Gunakan GPU atau dry_run.")
    print("═" * 78)

    if CONFIG.dry_run:
        print("  dry_run=True: hanya menampilkan konfigurasi sweep")
        print(f"  steps_values={STEPS_VALUES}  directions={PROPOSE_DIRS}")
        for d in PROPOSE_DIRS:
            hint = PROPOSE_DIRECTION_HINTS.get(d) if d else None
            label = d or "default"
            print(f"\n  [{label}] hint:\n    {hint or '(none — pakai factor_hypothesis_specification)'}")
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()
    temp, top_p = 0.7, 0.95
    ctor_sys, ctor_usr = _build_construct_msgs(target_hypothesis="")

    results_table: list[dict] = []
    log_sections: list[tuple[str, str]] = [
        (f"CONFIG  steps={STEPS_VALUES}  directions={PROPOSE_DIRS}  temp={temp}", ""),
    ]

    for direction in PROPOSE_DIRS:
        hint = PROPOSE_DIRECTION_HINTS.get(direction) if direction else None
        dir_label = direction or "default"
        prop_sys, prop_usr = _build_propose_msgs(direction_hint=hint)

        for m in STEPS_VALUES:
            print(f"\n── direction={dir_label}  latent_steps={m} ──")
            t0 = time.time()
            r_prop = backend.build_messages_and_run(
                user_prompt=prop_usr, system_prompt=prop_sys,
                mode="kv_only", latent_steps=m,
                temperature=temp, top_p=top_p, role=f"propose_{dir_label}_m{m}",
            )
            kv = r_prop.kv_cache
            kv_len = _kv_len(kv)
            elapsed_prop = round(time.time() - t0, 2)
            print(f"  [propose] elapsed={elapsed_prop}s  kv_len={kv_len}")

            t1 = time.time()
            r_ctor = backend.build_messages_and_run(
                user_prompt=ctor_usr, system_prompt=ctor_sys,
                json_mode=True, mode="kv_and_text",
                past_key_values=kv, latent_steps=m,
                temperature=temp, top_p=top_p, role=f"construct_{dir_label}_m{m}",
            )
            text_out = r_ctor.text or ""
            elapsed_ctor = round(time.time() - t1, 2)
            json_out = _extract_json(text_out)
            schema_ok = _has_correct_schema(json_out)
            kv_len_out = _kv_len(r_ctor.kv_cache)
            print(f"  [construct] elapsed={elapsed_ctor}s  valid_json={json_out is not None}  schema={schema_ok}")

            row = {
                "direction": dir_label,
                "latent_steps": m,
                "propose_kv_len": kv_len,
                "construct_kv_len": kv_len_out,
                "valid_json": json_out is not None,
                "correct_schema": schema_ok,
                "propose_elapsed_s": elapsed_prop,
                "construct_elapsed_s": elapsed_ctor,
                "total_elapsed_s": elapsed_prop + elapsed_ctor,
            }
            results_table.append(row)
            log_sections.append((
                f"dir={dir_label}  m={m}  propose_kv={kv_len}  construct_kv={kv_len_out}  "
                f"valid={json_out is not None}  schema={schema_ok}  "
                f"propose={elapsed_prop}s  construct={elapsed_ctor}s",
                text_out,
            ))

    # ── Tabel ringkasan ───────────────────────────────────────────────────────
    print("\n── TABEL HASIL SWEEP ──")
    header = (f"{'direction':>15} | {'m':>4} | {'prop_kv':>7} | {'ctor_kv':>7} | "
              f"{'json':>5} | {'schema':>6} | {'prop_s':>7} | {'ctor_s':>7} | {'total_s':>8}")
    print(header)
    print("─" * len(header))
    for r in results_table:
        print(f"{r['direction']:>15} | {r['latent_steps']:>4} | {r['propose_kv_len']:>7} | "
              f"{r['construct_kv_len']:>7} | {str(r['valid_json']):>5} | {str(r['correct_schema']):>6} | "
              f"{r['propose_elapsed_s']:>6.2f}s | {r['construct_elapsed_s']:>6.2f}s | {r['total_elapsed_s']:>7.2f}s")

    table_str = header + "\n"
    for r in results_table:
        table_str += (f"{r['direction']:>15} | {r['latent_steps']:>4} | {r['propose_kv_len']:>7} | "
                      f"{r['construct_kv_len']:>7} | {str(r['valid_json']):>5} | {str(r['correct_schema']):>6} | "
                      f"{r['propose_elapsed_s']:>6.2f}s | {r['construct_elapsed_s']:>6.2f}s | {r['total_elapsed_s']:>7.2f}s\n")

    log_sections.append(("SUMMARY TABLE", table_str))
    _log_save(log_path, log_sections)
    print(f"\n── LOG SAVED ── {log_path}")

    best = min((r for r in results_table if r["correct_schema"]),
               key=lambda r: r["total_elapsed_s"], default=None)
    if best:
        print(f"\n  Efisiensi terbaik (benar + paling cepat): "
              f"direction={best['direction']}  m={best['latent_steps']}  total={best['total_elapsed_s']:.2f}s")

    return {
        "group": group, "case": case,
        "ok_format": any(r["correct_schema"] for r in results_table),
        "elapsed_s": round(sum(r["total_elapsed_s"] for r in results_table), 2),
        "log_path": str(log_path),
        "table": results_table,
        "parsed": None,
    }


# ─── Test 5: Construct + Coder format stress test ────────────────────────────

def test_construct_coder_format(
    latent_steps: int | None = None,
    n_runs: int = 3,
) -> dict:
    """
    Fokus pada format compliance Construct dan Coder: jalankan N kali, hitung
    berapa persen run yang menghasilkan output dengan format benar.

    Ini stress-test khusus untuk dua agent yang paling sering error:
    - Construct: apakah selalu nested JSON dengan 4 key wajib?
    - Coder: apakah selalu {"expr": "<valid_expression>"}?

    Setiap run = Propose(kv_only) → Construct(kv+text) → Coder(kv+text).
    Variasi hasil antar run disebabkan oleh temperature sampling.

    Mode perbandingan:
      - LatentMAS: propose kv_only → construct kv+text → coder kv+text
      - TextMAS:   propose text → construct text (hyp injected) → coder text (factor injected)

    Output: tabel per-run + persentase success rate tiap agent.
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    group, case = "multi_agent_kv", f"construct_coder_format_m{latent_steps}_n{n_runs}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}")
    print(f"  n_runs={n_runs}  latent_steps={latent_steps}  dry_run={CONFIG.dry_run}")
    print("═" * 78)

    if CONFIG.dry_run:
        prop_sys, prop_usr = _build_propose_msgs()
        ctor_sys, ctor_usr = _build_construct_msgs(fx.HYPOTHESIS_STR)
        coder_sys, coder_usr = _build_coder_msgs(fx.FACTOR_TASK.get_task_description())
        print("── CONSTRUCT system (preview) ──"); print(ctor_sys[:600])
        print("── CODER system (preview) ──"); print(coder_sys[:400])
        print("── CODER user format (JSON_ONLY_SUFFIX appended) ──")
        print("... (factor_information_str) ...\n" + JSON_ONLY_SUFFIX)
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()
    temp, top_p = 0.7, 0.95

    prop_sys, prop_usr = _build_propose_msgs()
    ctor_sys_latent, ctor_usr_latent = _build_construct_msgs(target_hypothesis="")
    coder_sys_latent, coder_usr_latent = _build_coder_msgs()

    latent_runs: list[dict] = []
    text_runs: list[dict] = []
    log_sections: list[tuple[str, str]] = [
        (f"CONFIG  n_runs={n_runs}  latent_steps={latent_steps}  temp={temp}", ""),
    ]

    for i in range(n_runs):
        print(f"\n── Run {i+1}/{n_runs} ──")

        # ─ LatentMAS run ─────────────────────────────────────────────────────
        print(f"  [Latent] Propose (kv_only)...")
        t0 = time.time()
        r_prop_L = backend.build_messages_and_run(
            user_prompt=prop_usr, system_prompt=prop_sys,
            mode="kv_only", latent_steps=latent_steps,
            temperature=temp, top_p=top_p, role=f"propose_L{i}",
        )
        kv_prop = r_prop_L.kv_cache

        print(f"  [Latent] Construct (kv+text)...")
        r_ctor_L = backend.build_messages_and_run(
            user_prompt=ctor_usr_latent, system_prompt=ctor_sys_latent,
            json_mode=True, mode="kv_and_text",
            past_key_values=kv_prop, latent_steps=latent_steps,
            temperature=temp, top_p=top_p, role=f"construct_L{i}",
        )
        text_ctor_L = r_ctor_L.text or ""
        json_ctor_L = _extract_json(text_ctor_L)
        schema_L = _has_correct_schema(json_ctor_L)
        kv_ctor = r_ctor_L.kv_cache

        print(f"  [Latent] Coder (kv+text)...")
        r_coder_L = backend.build_messages_and_run(
            user_prompt=coder_usr_latent, system_prompt=coder_sys_latent,
            json_mode=True, mode="kv_and_text",
            past_key_values=kv_ctor, latent_steps=latent_steps,
            temperature=0.5, top_p=top_p, role=f"coder_L{i}",
        )
        text_coder_L = r_coder_L.text or ""
        json_coder_L = _extract_json(text_coder_L)
        expr_L = _has_valid_expr(json_coder_L)
        elapsed_L = round(time.time() - t0, 2)
        print(f"    construct_schema={schema_L}  coder_expr={expr_L}  elapsed={elapsed_L}s")

        latent_runs.append({
            "run": i + 1, "construct_schema": schema_L, "coder_valid_expr": expr_L,
            "both_ok": schema_L and expr_L, "elapsed_s": elapsed_L,
            "construct_text": text_ctor_L, "coder_text": text_coder_L,
        })

        # ─ TextMAS run ───────────────────────────────────────────────────────
        print(f"  [Text ] Propose (text_only)...")
        t1 = time.time()
        r_prop_T = backend.build_messages_and_run(
            user_prompt=prop_usr, system_prompt=prop_sys,
            json_mode=True, mode="text_only",
            temperature=temp, top_p=top_p, role=f"propose_T{i}",
        )
        text_prop_T = r_prop_T.text or ""
        json_prop_T = _extract_json(text_prop_T)
        hyp_T = str(json_prop_T.get("hypothesis", text_prop_T[:300])) if json_prop_T else text_prop_T[:300]

        ctor_sys_T, ctor_usr_T = _build_construct_msgs(target_hypothesis=hyp_T)
        print(f"  [Text ] Construct (text_only)...")
        r_ctor_T = backend.build_messages_and_run(
            user_prompt=ctor_usr_T, system_prompt=ctor_sys_T,
            json_mode=True, mode="text_only",
            temperature=temp, top_p=top_p, role=f"construct_T{i}",
        )
        text_ctor_T = r_ctor_T.text or ""
        json_ctor_T = _extract_json(text_ctor_T)
        schema_T = _has_correct_schema(json_ctor_T)
        factor_info_T = _extract_factor_info_from_construct(json_ctor_T)

        coder_sys_T, coder_usr_T = _build_coder_msgs(factor_info_str=factor_info_T)
        print(f"  [Text ] Coder (text_only)...")
        r_coder_T = backend.build_messages_and_run(
            user_prompt=coder_usr_T, system_prompt=coder_sys_T,
            json_mode=True, mode="text_only",
            temperature=0.5, top_p=top_p, role=f"coder_T{i}",
        )
        text_coder_T = r_coder_T.text or ""
        json_coder_T = _extract_json(text_coder_T)
        expr_T = _has_valid_expr(json_coder_T)
        elapsed_T = round(time.time() - t1, 2)
        print(f"    construct_schema={schema_T}  coder_expr={expr_T}  elapsed={elapsed_T}s")

        text_runs.append({
            "run": i + 1, "construct_schema": schema_T, "coder_valid_expr": expr_T,
            "both_ok": schema_T and expr_T, "elapsed_s": elapsed_T,
            "construct_text": text_ctor_T, "coder_text": text_coder_T,
        })

        log_sections.append((
            f"Run {i+1} — LatentMAS  construct={schema_L}  coder={expr_L}  both={schema_L and expr_L}",
            f"CONSTRUCT:\n{text_ctor_L}\n\nCODER:\n{text_coder_L}",
        ))
        log_sections.append((
            f"Run {i+1} — TextMAS    construct={schema_T}  coder={expr_T}  both={schema_T and expr_T}",
            f"CONSTRUCT:\n{text_ctor_T}\n\nCODER:\n{text_coder_T}",
        ))

    # ── Ringkasan success rate ─────────────────────────────────────────────────
    def _pct(runs, key):
        return f"{sum(r[key] for r in runs)}/{len(runs)} ({100*sum(r[key] for r in runs)//len(runs)}%)"

    print("\n── SUCCESS RATE ──")
    print(f"                   construct_schema    coder_valid_expr    both_ok")
    print(f"  LatentMAS  :     {_pct(latent_runs,'construct_schema'):>20}  {_pct(latent_runs,'coder_valid_expr'):>18}  {_pct(latent_runs,'both_ok')}")
    print(f"  TextMAS    :     {_pct(text_runs,'construct_schema'):>20}  {_pct(text_runs,'coder_valid_expr'):>18}  {_pct(text_runs,'both_ok')}")

    summary_str = (
        f"n_runs={n_runs}  latent_steps={latent_steps}\n\n"
        f"{'':20} construct_schema    coder_valid_expr    both_ok\n"
        f"{'LatentMAS':20} {_pct(latent_runs,'construct_schema'):>20}  "
        f"{_pct(latent_runs,'coder_valid_expr'):>18}  {_pct(latent_runs,'both_ok')}\n"
        f"{'TextMAS':20} {_pct(text_runs,'construct_schema'):>20}  "
        f"{_pct(text_runs,'coder_valid_expr'):>18}  {_pct(text_runs,'both_ok')}\n"
    )
    log_sections.append(("SUMMARY — FORMAT SUCCESS RATE", summary_str))
    _log_save(log_path, log_sections)
    print(f"\n── LOG SAVED ── {log_path}")

    latent_ok = sum(r["both_ok"] for r in latent_runs)
    text_ok = sum(r["both_ok"] for r in text_runs)
    return {
        "group": group, "case": case, "latent_steps": latent_steps, "n_runs": n_runs,
        "ok_format": latent_ok > 0 or text_ok > 0,
        "elapsed_s": round(sum(r["elapsed_s"] for r in latent_runs + text_runs), 2),
        "log_path": str(log_path),
        "latent_success_rate": f"{latent_ok}/{n_runs}",
        "text_success_rate": f"{text_ok}/{n_runs}",
        "parsed": None,
    }


# ─── Registry ─────────────────────────────────────────────────────────────────

CASES = {
    "chain_ablation":         test_chain_ablation,
    "kv_probe":               test_kv_probe,
    "knn_sweep":              test_knn_sweep,
    "latent_steps_sweep":     test_latent_steps_sweep,
    "construct_coder_format": test_construct_coder_format,
}
