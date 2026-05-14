"""
Pair test: Construct ↔ Coder retry loop dengan eksekusi expression beneran.

Tujuan: investigasi bug "Coder retry collapse" (lihat debug session 075518,
panggilan 7-12 di mana coder retry menghasilkan text_len=7 atau 0 berkali-kali).

Pola yang diobservasi di production:
  - Setelah construct → eksekusi expression gagal → coder retry
  - Attempt 1 mungkin valid (text_len=84 di session 075518)
  - Attempt 2-5 collapse: text_len=7 (output cuma <think>\n) atau 0 (kosong)
  - Walaupun ada kv_baseline.crop() di evolving_strategy.py, KV tetap tercemari
    oleh prompt + latent steps yang menumpuk → infinite loop tanpa progress

Skenario test:
  A. first_run    → Construct → ekstrak expr → parse_expression(expr).
                    Kalau VALID: success, no LLM call (mirror first-run path).
                    Kalau INVALID: tampilkan error real dari parser, no retry.
  B. retry_loop   → Construct → ekstrak expr → parse_expression(expr).
                    Force-trigger retry path (5 attempt) dengan production
                    crop behavior. Mirror evolving_strategy.py:520+.

Setiap retry attempt diukur:
  - text_len (deteksi collapse: <20 char)
  - mirror detection: apakah expr sama (normalized) dengan attempt sebelumnya?
  - parse_expression(new_expr): apakah valid?
  - KV length sebelum/sesudah crop
  - Elapsed time

Pemakaian:
    python -m try.run --group pair_construct_coder --case first_run
    python -m try.run --group pair_construct_coder --case retry_loop
    python -m try.run --group pair_construct_coder  # semua

Dry-run:
    python -m try.run --group pair_construct_coder --dry-run
"""

from __future__ import annotations

import json as _json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, StrictUndefined

from .common import (
    load_yaml, PROMPT_PATHS, get_latent_backend, _extract_json,
)
from .config import CONFIG
from . import fixtures as fx
from .test_coder_evaluator import JSON_ONLY_SUFFIX
from .probe import (
    enabled_modes_from_env, run_probes_at,
    format_probes_for_log, print_probe_summary,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _jinja(template: str, **kw) -> str:
    return Environment(undefined=StrictUndefined).from_string(template).render(**kw)


def _factors_yaml() -> dict:
    return load_yaml(PROMPT_PATHS["factors_prompts"])


def _coder_qa_yaml() -> dict:
    return load_yaml(PROMPT_PATHS["factors_coder_qa"])


def _kv_len(kv) -> int:
    try:
        from backend.llm.models import _past_length
        return _past_length(kv)
    except Exception:
        return -1


def _is_collapse(text: str) -> bool:
    """Coder collapse: sangat pendek atau hanya tag <think>."""
    if not text or not text.strip():
        return True
    s = text.strip()
    if len(s) < 20:
        return True
    # Cek apakah hanya think-tag
    if s.startswith("<think>") and len(s) < 50:
        return True
    return False


def _parse_expression_safe(expr: str) -> tuple[bool, str]:
    """
    Jalankan parse_expression() beneran. Return (valid, error_msg).
    Pakai backend parser asli — kalau gagal, return error string yang akan
    dijadikan former_feedback untuk retry path.
    """
    if not expr or not isinstance(expr, str):
        return False, "Expression is empty or not a string."
    try:
        # Add backend to path
        backend_path = Path(__file__).resolve().parent.parent / "backend"
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))
        from backend.factors.coder.expr_parser import parse_expression, parse_symbol
        # Standard columns
        STANDARD_COLS = ["$open", "$close", "$high", "$low", "$volume", "$return"]
        e = parse_symbol(expr, STANDARD_COLS)
        _ = parse_expression(e)
        return True, ""
    except Exception as exc:
        tb_short = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return False, f"parse_expression failed: {tb_short}\n  on expression: {expr[:200]}"


def _normalize_expr(expr: str) -> str:
    """Untuk mirror detection: normalize whitespace + lowercase."""
    return "".join(expr.split()).lower()


def _extract_expr_from_construct(construct_json: dict | None) -> str:
    """Ambil expression dari faktor pertama di construct output."""
    if not construct_json:
        return ""
    try:
        _, info = next(iter(construct_json.items()))
        return str(info.get("expression", ""))
    except (StopIteration, AttributeError):
        return ""


def _extract_expr_from_coder(coder_json: dict | None) -> str:
    """Ambil expr dari coder output {"expr": "..."}."""
    if not coder_json:
        return ""
    return str(coder_json.get("expr", ""))


def _log_save(log_path: Path, sections: list[tuple[str, str]]) -> None:
    lines = [f"# Log: {log_path.name}", f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for title, content in sections:
        lines += ["=" * 78, title, "=" * 78, content, ""]
    log_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Message builders ────────────────────────────────────────────────────────

def _render_hf(trace, limit: int = 6) -> str:
    y = _factors_yaml()
    if len(trace.hist) == 0:
        return "No previous hypothesis and feedback available since it's the first round."
    lt = SimpleNamespace(scen=trace.scen, hist=trace.hist[-limit:])
    return _jinja(y["hypothesis_and_feedback"], trace=lt)


def _build_construct_msgs(target_hypothesis: str = "") -> tuple[str, str]:
    y = _factors_yaml()
    trace = fx.TRACE
    scen_desc = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
    hf = _render_hf(trace)
    sys_c = _jinja(
        y["hypothesis2experiment"]["system_prompt"],
        targets="factor", scenario=scen_desc,
        experiment_output_format=y["experiment_output_format"],
    )
    usr_c = _jinja(
        y["hypothesis2experiment"]["user_prompt"],
        targets="factor", target_hypothesis=target_hypothesis,
        hypothesis_and_feedback=hf,
        function_lib_description=y["function_lib_description"],
        target_list=None, RAG=None, expression_duplication=None,
    )
    return sys_c, usr_c


def _build_coder_retry_msgs(
    factor_info_str: str,
    former_expression: str,
    former_feedback: str,
) -> tuple[str, str]:
    """
    Build coder retry prompt (FactorParsingStrategy.implement_one_task retry path).
    Mirror dari production: system + user + JSON_ONLY_SUFFIX.
    """
    y = _coder_qa_yaml()
    scen_desc = fx.SCENARIO.get_scenario_all_desc(filtered_tag="feature")
    sys_c = _jinja(
        y["evolving_strategy_factor_implementation_v1_system"],
        scenario=scen_desc,
    )
    usr_c = _jinja(
        y["evolving_strategy_factor_implementation_v2_user"],
        factor_information_str=factor_info_str,
        former_expression=former_expression,
        execution_log=former_feedback,
        code_comment=None,
        queried_similar_error_knowledge=[],
        error_summary_critics=None,
        similar_successful_factor_description=None,
        similar_successful_expression=None,
        latest_attempt_to_latest_successful_execution=None,
    )
    return sys_c, usr_c + JSON_ONLY_SUFFIX


def _build_factor_info_from_construct(construct_json: dict | None) -> str:
    """Inject construct output sebagai factor_information_str ke coder."""
    if not construct_json:
        return fx.FACTOR_TASK.get_task_description()
    try:
        name, info = next(iter(construct_json.items()))
        return (
            f"Factor: {name}\n"
            f"Description: {info.get('description', '')}\n"
            f"Formulation: {info.get('formulation', '')}\n"
            f"Expression: {info.get('expression', '')}"
        )
    except (StopIteration, AttributeError):
        return fx.FACTOR_TASK.get_task_description()


# ─── Test A: first_run path (no LLM call for coder if expr valid) ────────────

def test_first_run(latent_steps: int | None = None) -> dict:
    """
    Mirror first-run path production: Construct → ekstrak expr → parse_expression.
    Kalau VALID: tidak ada panggilan LLM ke coder (cuma template render).
    Kalau INVALID: laporkan error real dari parser.
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    group, case = "pair_construct_coder", f"first_run_m{latent_steps}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  (success path: parse_expression validates construct output)")
    print("═" * 78)

    if CONFIG.dry_run:
        cs, cu = _build_construct_msgs()
        print("── CONSTRUCT system ──"); print(cs[:400])
        print("── PARSER ──: parse_expression() dari backend.factors.coder.expr_parser")
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()
    construct_sys, construct_usr = _build_construct_msgs(target_hypothesis="")

    # Step 1: Construct (text_only — kita tidak butuh KV chain di test ini)
    print("── Step 1: Construct (text_only) untuk hasilkan factor expression ──")
    t0 = time.time()
    r_ctor = backend.build_messages_and_run(
        user_prompt=construct_usr, system_prompt=construct_sys,
        json_mode=True, mode="text_only",
        temperature=0.7, top_p=0.95, role="construct_first_run",
    )
    construct_text = r_ctor.text or ""
    construct_json = _extract_json(construct_text)
    elapsed_ctor = round(time.time() - t0, 2)
    print(f"  text_len={len(construct_text)}  json_ok={construct_json is not None}  elapsed={elapsed_ctor}s")

    # Step 2: Ekstrak expression + parse
    expr = _extract_expr_from_construct(construct_json)
    print(f"\n── Step 2: Ekstrak & parse expression: {expr!r}")
    valid, err_msg = _parse_expression_safe(expr)
    print(f"  parse_expression: {'VALID — first-run success, NO LLM CALL needed' if valid else 'INVALID'}")
    if not valid:
        print(f"  error: {err_msg}")

    _log_save(log_path, [
        (f"FIRST-RUN PATH  construct_elapsed={elapsed_ctor}s  expr_valid={valid}", ""),
        ("CONSTRUCT OUTPUT", construct_text),
        (f"EXTRACTED EXPRESSION", expr),
        (f"PARSE RESULT", "VALID — production would render template directly (no coder LLM call)"
         if valid else f"INVALID:\n{err_msg}"),
    ])
    print(f"\n── LOG SAVED ── {log_path}")

    return {
        "group": group, "case": case, "latent_steps": latent_steps,
        "ok_format": construct_json is not None and valid,
        "elapsed_s": elapsed_ctor,
        "log_path": str(log_path),
        "construct_json": construct_json,
        "expression": expr,
        "expr_valid": valid,
        "parse_error": err_msg if not valid else None,
        "parsed": construct_json,
    }


# ─── Test B: retry_loop (REPRODUKSI BUG 2) ───────────────────────────────────

def test_retry_loop(
    latent_steps: int | None = None,
    n_attempts: int = 5,
) -> dict:
    """
    Reproduksi bug coder retry collapse: Construct → expr → force retry 5x dengan
    production crop behavior.

    Production behavior (evolving_strategy.py):
      - kv_baseline = self._past_kv (KV dari construct step)
      - kv_baseline_len = panjang KV sebelum retry loop
      - Tiap attempt: kv_baseline.crop(kv_baseline_len) untuk reset
      - Lalu inject prompt retry (former_expr + former_feedback) + latent steps

    Yang diukur per attempt:
      - text_len (collapse jika <20)
      - mirror detection: expr_norm == former_expr_norm?
      - parse_expression(new_expr) hasil
      - KV length sebelum & sesudah crop
      - elapsed_s

    Kalau eksekusi expression construct LOLOS parser, kita inject synthetic
    error message (mirip EXECUTION_FEEDBACK_FAIL) untuk simulate runtime error.
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    group, case = "pair_construct_coder", f"retry_loop_m{latent_steps}_n{n_attempts}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  (REPRODUKSI BUG 2: coder retry collapse)")
    print(f"  n_attempts={n_attempts}  KV strategy: production crop")
    print("═" * 78)

    if CONFIG.dry_run:
        cs, cu = _build_construct_msgs()
        coder_s, coder_u = _build_coder_retry_msgs(
            factor_info_str=fx.FACTOR_TASK.get_task_description(),
            former_expression="RANK(TS_MEAN($return, 5))",
            former_feedback="ValueError: NaN encountered in output",
        )
        print("── CONSTRUCT system ──"); print(cs[:300])
        print("── CODER RETRY system ──"); print(coder_s[:300])
        print("── CODER RETRY user (former_expression + former_feedback injected) ──")
        print(coder_u[:600])
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()

    # Step 1: Construct (kv_and_text untuk dapat construct_kv yang akan jadi
    # kv_baseline di retry loop, mirror production behavior)
    print("── Step 1: Construct (kv_and_text) untuk hasilkan kv_baseline + initial expr ──")
    construct_sys, construct_usr = _build_construct_msgs(target_hypothesis="")
    t0 = time.time()
    r_ctor = backend.build_messages_and_run(
        user_prompt=construct_usr, system_prompt=construct_sys,
        json_mode=True, mode="kv_and_text",
        latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role="construct_for_retry",
    )
    construct_text = r_ctor.text or ""
    construct_json = _extract_json(construct_text)
    construct_kv = r_ctor.kv_cache
    construct_kv_len = _kv_len(construct_kv)
    elapsed_ctor = round(time.time() - t0, 2)
    initial_expr = _extract_expr_from_construct(construct_json)
    print(f"  construct: text_len={len(construct_text)}  kv_len={construct_kv_len}  "
          f"elapsed={elapsed_ctor}s")
    print(f"  initial_expr: {initial_expr!r}")

    # Probe construct_kv SEBELUM retry loop — snapshot baseline.
    # Setiap probe modes yang aktif via TEST_PROBE akan di-jalankan; results
    # disimpan untuk dibandingkan dengan probe POST-retry agar terlihat apakah
    # retry loop mengubah isi KV (Bug 2: crop seharusnya kembali ke baseline).
    probe_modes = enabled_modes_from_env()
    pre_retry_probes = run_probes_at(
        backend, construct_kv, kv_label="construct_kv_pre_retry", modes=probe_modes,
    )
    if pre_retry_probes:
        print_probe_summary(pre_retry_probes)

    # Step 2: Validasi initial expression. Tentukan former_feedback untuk retry.
    initial_valid, initial_err = _parse_expression_safe(initial_expr)
    if initial_valid:
        # Force-trigger retry path dengan synthetic execution error
        former_feedback = (
            "ValueError: NaN encountered in output at >12% of rows.\n"
            "Traceback (most recent call last):\n"
            '  File "factor.py", line 23, in calculate_factor\n'
            "    df[name] = eval(expr)\n"
            "RuntimeWarning: divide by zero encountered in true_divide.\n"
            "Output stability check failed: factor values exhibit extreme outliers."
        )
        print(f"  initial expr is VALID syntactically — inject synthetic execution error to trigger retry")
    else:
        former_feedback = initial_err
        print(f"  initial expr is INVALID — using real parser error as former_feedback")
    print(f"  former_feedback (truncated): {former_feedback[:150]!r}")

    # Step 3: Build coder retry prompt template (sys + usr_template tanpa eksekusi)
    factor_info = _build_factor_info_from_construct(construct_json)
    coder_sys, coder_usr_initial = _build_coder_retry_msgs(
        factor_info_str=factor_info,
        former_expression=initial_expr,
        former_feedback=former_feedback,
    )

    # Step 4: Retry loop dengan production crop behavior
    print(f"\n── Step 3: Retry loop {n_attempts}x (kv_baseline.crop tiap attempt) ──")

    # Cek apakah construct_kv punya method crop (DynamicCache)
    has_crop = hasattr(construct_kv, "crop") if construct_kv is not None else False
    if not has_crop:
        print(f"  ⚠ construct_kv tidak punya .crop() — fallback: tidak reset KV per attempt "
              f"(behavior tetap mirror production yang juga akan tanpa crop)")

    attempts: list[dict] = []
    former_expr_for_attempt = initial_expr
    former_feedback_for_attempt = former_feedback
    temp_schedule = [None, 0.7, 0.9, 1.0, 1.0]

    for i in range(n_attempts):
        print(f"\n  ─ Attempt {i+1}/{n_attempts} (temp={temp_schedule[min(i, 4)]}) ─")

        # Production: reset KV ke baseline length sebelum tiap attempt
        if has_crop:
            try:
                construct_kv.crop(construct_kv_len)
                kv_pre = _kv_len(construct_kv)
                print(f"    [crop] kv_pre={kv_pre} (reset to baseline {construct_kv_len})")
            except Exception as crop_err:
                print(f"    [crop] FAILED: {crop_err} — proceed with current KV")
                kv_pre = _kv_len(construct_kv)
        else:
            kv_pre = _kv_len(construct_kv)

        # Build prompt untuk attempt ini (former_expr + former_feedback dari prev)
        _, coder_usr = _build_coder_retry_msgs(
            factor_info_str=factor_info,
            former_expression=former_expr_for_attempt,
            former_feedback=former_feedback_for_attempt,
        )

        # Mirror hint kalau attempt sebelumnya mirror (production behavior)
        if i > 0 and attempts[-1]["mirror_detected"]:
            mirror_hint = (
                f"\n\n**PREVIOUS ATTEMPT RETURNED THE SAME EXPRESSION "
                f"({former_expr_for_attempt}) — THIS IS A FAILURE. You MUST change operator, "
                f"window size, or base variable. Try a structurally different expression now.**"
            )
            coder_usr = coder_usr + mirror_hint

        t1 = time.time()
        r_coder = backend.build_messages_and_run(
            user_prompt=coder_usr, system_prompt=coder_sys,
            json_mode=True, mode="kv_and_text",
            past_key_values=construct_kv, latent_steps=latent_steps,
            temperature=temp_schedule[min(i, 4)], top_p=0.95,
            role=f"coder_retry_a{i+1}",
        )
        primary_text = r_coder.text or ""
        primary_text_len = len(primary_text)
        kv_post = _kv_len(r_coder.kv_cache)
        elapsed_primary = round(time.time() - t1, 2)

        # Mirror produksi: kalau kv_and_text collapse (text kosong), fallback
        # ke text_only. Bypass latent_pass → model tidak ter-bias EOS oleh
        # 10 latent virtual tokens di atas construct_kv besar.
        # Lihat evolving_strategy.py:307-322.
        fallback_used = False
        fallback_text = ""
        fallback_elapsed_s = None
        if not primary_text.strip():
            fallback_used = True
            t_fb = time.time()
            r_fb = backend.build_messages_and_run(
                user_prompt=coder_usr, system_prompt=coder_sys,
                json_mode=True, mode="text_only",
                past_key_values=construct_kv,
                temperature=temp_schedule[min(i, 4)], top_p=0.95,
                role=f"coder_retry_a{i+1}_textonly",
            )
            fallback_text = r_fb.text or ""
            fallback_elapsed_s = round(time.time() - t_fb, 2)
            print(f"    [fallback text_only] text_len={len(fallback_text)}  "
                  f"elapsed={fallback_elapsed_s}s")

        coder_text = fallback_text if fallback_used else primary_text
        coder_json = _extract_json(coder_text)
        elapsed_a = round(elapsed_primary + (fallback_elapsed_s or 0), 2)

        new_expr = _extract_expr_from_coder(coder_json)
        collapse = _is_collapse(coder_text)
        mirror = (_normalize_expr(new_expr) == _normalize_expr(former_expr_for_attempt)) and bool(new_expr)
        new_valid, new_err = _parse_expression_safe(new_expr) if new_expr else (False, "no expression extracted")

        attempts.append({
            "attempt": i + 1,
            "kv_pre": kv_pre,
            "kv_post": kv_post,
            "kv_growth": kv_post - kv_pre,
            "text_len": len(coder_text),
            "primary_text_len": primary_text_len,
            "fallback_used": fallback_used,
            "fallback_text_len": len(fallback_text) if fallback_used else None,
            "fallback_elapsed_s": fallback_elapsed_s,
            "elapsed_s": elapsed_a,
            "collapse_detected": collapse,
            "mirror_detected": mirror,
            "new_expr": new_expr,
            "new_expr_valid": new_valid,
            "new_expr_parse_error": new_err if not new_valid else None,
            "coder_text": coder_text,
            "primary_text": primary_text,
        })

        print(f"    kv_pre={kv_pre} → kv_post={kv_post} (Δ{kv_post-kv_pre})")
        print(f"    primary_text_len={primary_text_len}  fb={'Y' if fallback_used else 'N'}  "
              f"text_len={len(coder_text)}  collapse={collapse}  mirror={mirror}  "
              f"new_valid={new_valid}  elapsed={elapsed_a}s")
        if new_expr:
            print(f"    new_expr: {new_expr!r}")
        if not new_valid and new_expr:
            print(f"    parse error: {new_err[:120]}")

        # Stop kalau dapat expression valid yang berbeda
        if new_valid and not mirror:
            print(f"    ✓ SUCCESS — valid new expression different from former. Stop retry.")
            break

        # Update former_* untuk attempt berikutnya
        if new_expr and not mirror:
            former_expr_for_attempt = new_expr
            former_feedback_for_attempt = new_err if not new_valid else "Output validation failed."

    # ── Tabel ringkasan ──────────────────────────────────────────────────────
    print("\n── TABEL HASIL RETRY LOOP ──")
    header = (f"{'#':>2} | {'kv_pre':>7} | {'kv_post':>7} | {'prim_len':>8} | {'fb':>3} | "
              f"{'text_len':>8} | {'collapse':>8} | {'mirror':>6} | {'valid':>5} | {'elapsed':>7}")
    print(header)
    print("─" * len(header))
    for a in attempts:
        print(f"{a['attempt']:>2} | {a['kv_pre']:>7} | {a['kv_post']:>7} | "
              f"{a['primary_text_len']:>8} | {('Y' if a['fallback_used'] else 'N'):>3} | "
              f"{a['text_len']:>8} | {str(a['collapse_detected']):>8} | "
              f"{str(a['mirror_detected']):>6} | {str(a['new_expr_valid']):>5} | {a['elapsed_s']:>6.2f}s")

    # Statistik
    n_collapse = sum(a["collapse_detected"] for a in attempts)
    n_mirror = sum(a["mirror_detected"] for a in attempts)
    n_valid_diff = sum(a["new_expr_valid"] and not a["mirror_detected"] for a in attempts)
    n_primary_collapse = sum(a["primary_text_len"] == 0 for a in attempts)
    n_fb_used = sum(a["fallback_used"] for a in attempts)
    n_fb_recovered = sum(
        a["fallback_used"] and (a["fallback_text_len"] or 0) > 0 for a in attempts
    )
    print(f"\n  collapse: {n_collapse}/{len(attempts)}  mirror: {n_mirror}/{len(attempts)}  "
          f"valid_and_different: {n_valid_diff}/{len(attempts)}")
    print(f"  primary_collapse(kv_and_text): {n_primary_collapse}/{len(attempts)}  "
          f"fallback_used(text_only): {n_fb_used}  fallback_recovered: {n_fb_recovered}")

    if n_valid_diff == 0:
        print("  ⚠ INFINITE LOOP CONFIRMED — tidak ada attempt yang menghasilkan expr valid+berbeda")
    if n_primary_collapse > 0 and n_fb_recovered > 0:
        print(f"  ℹ text_only fallback recovered {n_fb_recovered}/{n_primary_collapse} "
              f"primary collapses — konsisten dengan production workaround")

    # Probe construct_kv SETELAH retry loop — apakah isi KV bergeser?
    # Bandingkan dengan pre_retry_probes; deviasi besar = crop tidak efektif.
    post_retry_probes = run_probes_at(
        backend, construct_kv, kv_label="construct_kv_post_retry", modes=probe_modes,
    )
    if post_retry_probes:
        print_probe_summary(post_retry_probes)

    # ── Save log ─────────────────────────────────────────────────────────────
    table_str = header + "\n"
    for a in attempts:
        table_str += (f"{a['attempt']:>2} | {a['kv_pre']:>7} | {a['kv_post']:>7} | "
                      f"{a['primary_text_len']:>8} | {('Y' if a['fallback_used'] else 'N'):>3} | "
                      f"{a['text_len']:>8} | {str(a['collapse_detected']):>8} | "
                      f"{str(a['mirror_detected']):>6} | {str(a['new_expr_valid']):>5} | "
                      f"{a['elapsed_s']:>6.2f}s\n")

    sections = [
        (f"CONFIG  latent_steps={latent_steps}  n_attempts={n_attempts}  "
         f"kv_baseline={construct_kv_len}  has_crop={has_crop}", ""),
        ("CONSTRUCT OUTPUT (basis untuk retry)", construct_text),
        (f"INITIAL EXPRESSION  valid={initial_valid}", initial_expr),
        ("FORMER FEEDBACK injected ke retry attempt 1",
         former_feedback if not initial_valid else f"(synthetic, initial valid)\n{former_feedback}"),
        (f"RETRY ATTEMPTS TABLE", table_str),
        (f"SUMMARY", (
            f"collapse: {n_collapse}/{len(attempts)}  "
            f"mirror: {n_mirror}/{len(attempts)}  "
            f"valid_and_different: {n_valid_diff}/{len(attempts)}\n"
            f"primary_collapse(kv_and_text): {n_primary_collapse}/{len(attempts)}  "
            f"fallback_used(text_only): {n_fb_used}  fallback_recovered: {n_fb_recovered}\n\n"
            + ("INFINITE LOOP CONFIRMED" if n_valid_diff == 0 else "Retry succeeded")
        )),
    ]
    for a in attempts:
        body = a["coder_text"]
        if a["fallback_used"]:
            body = (
                f"[primary kv_and_text text_len={a['primary_text_len']}] {a['primary_text']!r}\n\n"
                f"[fallback text_only text_len={a['fallback_text_len']} "
                f"elapsed={a['fallback_elapsed_s']}s]\n{a['coder_text']}"
            )
        sections.append((
            f"ATTEMPT {a['attempt']}  primary_len={a['primary_text_len']}  "
            f"fb={'Y' if a['fallback_used'] else 'N'}  text_len={a['text_len']}  "
            f"collapse={a['collapse_detected']}  mirror={a['mirror_detected']}  "
            f"valid={a['new_expr_valid']}",
            body,
        ))

    # Append KV-introspection probes (pre vs post retry) ke akhir log untuk
    # perbandingan langsung di file yang sama
    if pre_retry_probes:
        sections.extend(format_probes_for_log(pre_retry_probes))
    if post_retry_probes:
        sections.extend(format_probes_for_log(post_retry_probes))

    _log_save(log_path, sections)
    print(f"\n── LOG SAVED ── {log_path}")

    return {
        "group": group, "case": case, "latent_steps": latent_steps, "n_attempts": n_attempts,
        "ok_format": n_valid_diff > 0,
        "elapsed_s": round(elapsed_ctor + sum(a["elapsed_s"] for a in attempts), 2),
        "log_path": str(log_path),
        "construct_kv_baseline": construct_kv_len,
        "initial_expr": initial_expr,
        "initial_valid": initial_valid,
        "n_attempts_run": len(attempts),
        "n_collapse": n_collapse,
        "n_mirror": n_mirror,
        "n_valid_different": n_valid_diff,
        "attempts_summary": [
            {k: v for k, v in a.items() if k != "coder_text"} for a in attempts
        ],
        "parsed": None,
    }


# ─── Registry ────────────────────────────────────────────────────────────────

CASES = {
    "first_run":  test_first_run,
    "retry_loop": test_retry_loop,
}
