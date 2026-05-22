"""
A/B/C prompt comparison infrastructure untuk investigasi:

  1. Format/struktur prompt (XML vs flat, verbose vs compact)
  2. Variabel agent sebelumnya ditulis EKSPLISIT (text) vs IMPLISIT ("ada di KV")
  3. Apa yang sebenarnya ter-transfer via KV-cache

Tiga variant prompt yang dibandingkan:

  - "old"      : snapshot prompt SEBELUM commit d626266 (verbose, XML markup,
                 explicit target_hypothesis + hypothesis_and_feedback di construct)
  - "new"      : prompt saat ini (kompak, dengan {{ target_hypothesis_oneline }}
                 eksplisit, hasil perbaikan setelah evaluasi bug regression)
  - "implicit" : prompt saat ini tapi hypothesis text di-STRIP — model harus
                 mengandalkan KV-cache propose untuk mengingat hipotesis
                 (mereplikasi state broken antara commit d626266 dan fix)

Juga menyediakan eksperimen kombinasi KV-cache:

  - concat_kv_raw       : append kv_b ke kv_a tanpa fix-up posisi RoPE
                          (positions B di-encode di 0..len(B) padahal di
                          combined cache mereka di posisi len(A)..len(A+B))
  - run_corrected_chain : "true combining" — A sebagai seed, B di-process di atas
                          (chain normal — RoPE positions korek)

Penggunaan dari test files:
    from .prompt_ab import (
        build_propose_msgs, build_construct_msgs, build_coder_msgs,
        build_feedback_msgs, diff_prompt_structure, run_full_chain,
        concat_kv_raw, PROMPT_VARIANTS,
    )
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from . import fixtures as fx
from .common import PROMPT_PATHS, load_yaml


# ─── Constants ──────────────────────────────────────────────────────────────

# Commit dengan simplifikasi prompt yang menyebabkan output bias.
# Kita ambil PARENT (`^`) untuk dapat prompt SEBELUM simplifikasi.
OLD_PROMPTS_SHA = "d626266^"

PROMPT_VARIANTS = ("old", "new", "implicit")


# Copied verbatim dari backend/factors/coder/evolving_strategy.py:502
JSON_ONLY_SUFFIX = (
    "\n\nOUTPUT INSTRUCTION: Respond with ONLY the raw JSON object "
    "on a single line. No explanation, no preamble, no analysis. "
    'Example: {"expr": "TS_STD($close, 20)"}'
)


# ─── Git-based OLD prompt loader ────────────────────────────────────────────

_OLD_YAML_CACHE: dict[str, dict] = {}


def load_prompts_at_sha(sha: str, repo_relpath: str) -> dict:
    """`git show <sha>:<path>` → parsed YAML. In-memory cache per (sha, path)."""
    key = f"{sha}::{repo_relpath}"
    if key in _OLD_YAML_CACHE:
        return _OLD_YAML_CACHE[key]
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        ["git", "show", f"{sha}:{repo_relpath}"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git show failed for {sha}:{repo_relpath}\n"
            f"stderr={proc.stderr[:300]}"
        )
    data = yaml.safe_load(proc.stdout)
    if data is None:
        raise ValueError(f"Empty YAML from {sha}:{repo_relpath}")
    _OLD_YAML_CACHE[key] = data
    return data


def _yaml_factors_new() -> dict:
    return load_yaml(PROMPT_PATHS["factors_prompts"])


def _yaml_factors_old() -> dict:
    return load_prompts_at_sha(OLD_PROMPTS_SHA, "backend/factors/prompts/prompts.yaml")


def _yaml_coder_qa() -> dict:
    return load_yaml(PROMPT_PATHS["factors_coder_qa"])


def _jinja(template: str, **kw) -> str:
    return Environment(undefined=StrictUndefined).from_string(template).render(**kw)


# ─── History rendering (per variant) ─────────────────────────────────────────

def _render_hf(yaml_dict: dict, trace, limit: int = 6) -> str:
    if len(trace.hist) == 0:
        return "No previous hypothesis and feedback available since it's the first round."
    lt = SimpleNamespace(scen=trace.scen, hist=trace.hist[-limit:])
    return _jinja(yaml_dict["hypothesis_and_feedback"], trace=lt)


# ─── Prompt builders: propose / construct / coder / feedback × variant ──────

def build_propose_msgs(variant: str) -> tuple[str, str]:
    """Render propose system+user untuk variant. 'implicit' identik 'new'
    karena propose tidak menerima KV-cache upstream."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"variant={variant!r} not in {PROMPT_VARIANTS}")
    y = _yaml_factors_old() if variant == "old" else _yaml_factors_new()
    trace = fx.TRACE
    scen_desc = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
    hf = _render_hf(y, trace)
    sys_p = _jinja(
        y["hypothesis_gen"]["system_prompt"],
        targets="factor", scenario=scen_desc,
        hypothesis_output_format=y["hypothesis_output_format"],
        hypothesis_specification=y["factor_hypothesis_specification"],
    )
    usr_p = _jinja(
        y["hypothesis_gen"]["user_prompt"],
        targets="factor", hypothesis_and_feedback=hf,
        RAG=None, round=len(trace.hist),
    )
    return sys_p, usr_p


def build_construct_msgs(
    variant: str,
    target_hypothesis: str = "",
    target_hypothesis_oneline: str = "",
    expression_duplication: str | None = None,
) -> tuple[str, str]:
    """Render construct prompts. Variant menentukan:
        - 'old'      : XML <target_hypothesis>{{ target_hypothesis }}</target_hypothesis>
                       + scenario di system + hypothesis_and_feedback di user.
        - 'new'      : compact 'Hypothesis to implement:\\n{{ target_hypothesis_oneline }}'
                       — explicit text, tanpa scenario block, tanpa history.
        - 'implicit' : new tapi target_hypothesis_oneline DIGANTI text statis
                       'Target hypothesis is already in the prior context'
                       (replikasi state broken pasca d626266 sebelum fix).
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"variant={variant!r}")

    if variant == "old":
        y = _yaml_factors_old()
        trace = fx.TRACE
        scen_desc = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
        hf = _render_hf(y, trace)
        sys_c = _jinja(
            y["hypothesis2experiment"]["system_prompt"],
            targets="factor", scenario=scen_desc,
            experiment_output_format=y["experiment_output_format"],
        )
        usr_c = _jinja(
            y["hypothesis2experiment"]["user_prompt"],
            targets="factor",
            target_hypothesis=target_hypothesis or fx.HYPOTHESIS_STR,
            hypothesis_and_feedback=hf,
            function_lib_description=y["function_lib_description"],
            target_list=None, RAG=None,
            expression_duplication=expression_duplication,
        )
        return sys_c, usr_c

    # variant in ('new', 'implicit')
    y = _yaml_factors_new()
    sys_c = _jinja(
        y["hypothesis2experiment"]["system_prompt"],
        experiment_output_format=y["experiment_output_format"],
    )

    # For 'implicit', replace the actual hypothesis with the broken static text
    # that existed between d626266 and the fix commit. This isolates the effect
    # of "hypothesis told via KV alone" vs "hypothesis written in prompt text".
    hyp_inline = (
        target_hypothesis_oneline or fx.HYPOTHESIS_DICT["hypothesis"]
        if variant == "new"
        else "(see prior context — hypothesis already loaded in KV)"
    )
    usr_c = _jinja(
        y["hypothesis2experiment"]["user_prompt"],
        targets="factor",
        target_hypothesis_oneline=hyp_inline,
        function_lib_description=y["function_lib_description"],
        expression_duplication=expression_duplication,
    )
    return sys_c, usr_c


def build_coder_msgs(
    factor_info_str: str,
    former_expression: str,
    former_feedback: str,
) -> tuple[str, str]:
    """Coder retry prompt — TIDAK berubah lintas commit. Variant tidak relevan."""
    y = _yaml_coder_qa()
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


def build_feedback_msgs(
    variant: str,
    hypothesis_text: str = "",
    factor_summary: str = "",
    task_details: list | None = None,
    combined_result: str = "",
    complexity_warnings: str = "",
) -> tuple[str, str]:
    """Render feedback prompts. Variant menentukan:
        - 'old'      : task_details (full per-factor dict) + scenario di system
                       + hypothesis_text + combined_result.
        - 'new'      : hypothesis_oneline + factor_summary (compact bullet list)
                       + complexity_warnings + combined_result.
        - 'implicit' : new tapi factor_summary di-blank — model harus mengandalkan
                       KV upstream (construct/coder) untuk mengingat factor details.
    """
    if variant == "old":
        y = _yaml_factors_old()
        scen_desc = fx.SCENARIO.get_scenario_all_desc()
        sys_f = _jinja(y["factor_feedback_generation"]["system"], scenario=scen_desc)
        usr_f = _jinja(
            y["factor_feedback_generation"]["user"],
            hypothesis_text=hypothesis_text or fx.HYPOTHESIS_DICT["hypothesis"],
            task_details=task_details if task_details is not None else fx.TASK_DETAILS,
            combined_result=combined_result or fx.COMBINED_RESULT_STR,
        )
        return sys_f, usr_f

    y = _yaml_factors_new()
    sys_f = y["factor_feedback_generation"]["system"]

    if variant == "implicit":
        factor_summary_render = "(factor details already in prior context — refer to construct/coder KV)"
    else:
        factor_summary_render = factor_summary or "(no factors tested)"

    usr_f = _jinja(
        y["factor_feedback_generation"]["user"],
        hypothesis_oneline=hypothesis_text or fx.HYPOTHESIS_DICT["hypothesis"],
        factor_summary=factor_summary_render,
        complexity_warnings=complexity_warnings,
        combined_result=combined_result or fx.COMBINED_RESULT_STR,
    )
    return sys_f, usr_f


# ─── Prompt structural diff (quantitative comparison) ────────────────────────

_XML_TAG_RE = re.compile(r"<([a-z_][a-z0-9_]*)>", re.IGNORECASE)
_JINJA_VAR_RE = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}", re.IGNORECASE)


def diff_prompt_structure(label_a: str, prompt_a: str, label_b: str, prompt_b: str) -> dict:
    """Compute quantitative structural differences between two rendered prompts.

    Metrik:
      - char_len, line_count
      - xml_tags         : list of <tag> markers (proxy untuk verbosity old prompt)
      - hypothesis_anchor: apakah hipotesis literal muncul di prompt text?
                           (kunci untuk explicit-vs-implicit comparison)
      - explicit_history : apakah ada blok 'Round 1 / Round N' / '<history>'?
      - sentinel_phrases : phrase yang menunjukkan implicit reference
                           ("already in prior context", "in latent", dll.)
    """
    HYP_SAMPLE = fx.HYPOTHESIS_DICT["hypothesis"][:40].lower()
    IMPLICIT_PHRASES = [
        "already in the prior context", "in prior context",
        "already loaded in kv", "in latent", "in latent kv",
        "latent representation", "see prior context",
    ]

    def _analyze(label: str, prompt: str) -> dict:
        prompt_lower = prompt.lower()
        return {
            "label": label,
            "char_len": len(prompt),
            "line_count": prompt.count("\n") + 1 if prompt else 0,
            "xml_tags": sorted(set(_XML_TAG_RE.findall(prompt))),
            "n_xml_tags": len(_XML_TAG_RE.findall(prompt)),
            "hypothesis_anchor_present": HYP_SAMPLE in prompt_lower,
            "explicit_history_block": any(
                m in prompt_lower for m in ("round 1", "<history>", "─ round")
            ),
            "implicit_phrases_present": [
                ph for ph in IMPLICIT_PHRASES if ph in prompt_lower
            ],
            "n_jinja_unrendered": len(_JINJA_VAR_RE.findall(prompt)),  # sanity
        }

    return {
        label_a: _analyze(label_a, prompt_a),
        label_b: _analyze(label_b, prompt_b),
    }


def render_prompt_diff_table(diffs: dict[str, dict]) -> str:
    """Format diff dict jadi tabel teks untuk log."""
    labels = list(diffs.keys())
    metrics = ["char_len", "line_count", "n_xml_tags",
               "hypothesis_anchor_present", "explicit_history_block",
               "implicit_phrases_present", "n_jinja_unrendered"]
    rows = []
    rows.append(f"{'metric':<32} " + " ".join(f"{lbl:>20}" for lbl in labels))
    rows.append("─" * (32 + 21 * len(labels)))
    for m in metrics:
        vals = []
        for lbl in labels:
            v = diffs[lbl].get(m)
            if isinstance(v, list):
                v = ",".join(str(x) for x in v) or "(none)"
                v = v[:20]
            vals.append(f"{str(v):>20}")
        rows.append(f"{m:<32} " + " ".join(vals))
    # Show xml_tags separately (list form)
    rows.append("")
    rows.append("xml_tags found:")
    for lbl in labels:
        tags = diffs[lbl].get("xml_tags") or []
        rows.append(f"  [{lbl}] {tags or '(none)'}")
    return "\n".join(rows)


# ─── KV-cache concat experiments ─────────────────────────────────────────────

def concat_kv_raw(kv_a, kv_b):
    """Concatenate two DynamicCaches along sequence dim, WITHOUT position fix-up.

    Returns a fresh DynamicCache whose per-layer (keys, values) are
    torch.cat([kv_a.layer[i], kv_b.layer[i]], dim=-2).

    EXPECTED BEHAVIOR: model will misbehave when used as past_key_values
    because B's keys carry RoPE positions [0..len(B)] but in the concatenated
    cache they sit at sequence positions [len(A)..len(A)+len(B)]. The query at
    a later position will compute attention scores against B's keys as if B
    started at position 0 — i.e., distance is wrong. This is exactly the
    diagnostic we want to expose.

    Returns None if either KV is None or shapes incompatible.
    """
    if kv_a is None or kv_b is None:
        return None
    import torch
    from transformers import DynamicCache

    # Both must be DynamicCache with .layers
    if not (hasattr(kv_a, "layers") and hasattr(kv_b, "layers")):
        return None

    out = DynamicCache()
    n_layers = min(len(kv_a.layers), len(kv_b.layers))
    for i in range(n_layers):
        la = kv_a.layers[i]
        lb = kv_b.layers[i]
        ka = getattr(la, "keys", None); va = getattr(la, "values", None)
        kb = getattr(lb, "keys", None); vb = getattr(lb, "values", None)
        if ka is None or kb is None:
            continue
        try:
            new_k = torch.cat([ka, kb], dim=-2)
            new_v = torch.cat([va, vb], dim=-2)
        except Exception as e:
            print(f"  [concat_kv_raw] layer {i} cat failed: {e}")
            continue
        # DynamicCache layers populate via .update(); we call it with 1 step worth
        # of data per layer to set the initialized state. Easiest: directly
        # construct via update on a freshly-made layer.
        out.update(new_k, new_v, i)
    return out


def kv_length(kv) -> int:
    """Convenience — depends on backend.llm._shared._past_length."""
    if kv is None:
        return 0
    try:
        from backend.llm._shared import _past_length
        return int(_past_length(kv))
    except Exception:
        try:
            return int(kv.get_seq_length())
        except Exception:
            return -1


# ─── Full-chain runner ───────────────────────────────────────────────────────

def _synth_backtest_after(coder_expr: str) -> str:
    """Synthetic backtest table. Numbers vary slightly with expr length so each
    run looks different. NOT a real backtest — just realistic-looking output
    so the feedback agent receives metric-like content."""
    # Stable pseudo-variation based on len(expr) so the same chain runs reproduce.
    base_ic = 0.022 + (len(coder_expr or "") % 17) / 1000.0
    return (
        "metric                                                       Current Result  SOTA Result  Bigger columns name\n"
        f"1day.excess_return_without_cost.max_drawdown                        {0.10 + base_ic:.4f}       0.1120  Current Result\n"
        f"1day.excess_return_without_cost.information_ratio                   {0.55 + base_ic:.4f}       0.4820  Current Result\n"
        f"1day.excess_return_without_cost.annualized_return                   {0.09 + base_ic:.4f}       0.0890  Current Result\n"
        f"IC                                                                  {base_ic:.4f}       0.0280  Current Result\n"
    )


def _extract_construct_first_expr(json_obj: dict | None) -> tuple[str, str, str]:
    """Return (factor_name, expression, description) for the first factor."""
    if not json_obj or not isinstance(json_obj, dict):
        return "", "", ""
    try:
        name, info = next(iter(json_obj.items()))
        return str(name), str(info.get("expression", "")), str(info.get("description", ""))
    except Exception:
        return "", "", ""


def _extract_coder_expr(json_obj: dict | None) -> str:
    if not json_obj or not isinstance(json_obj, dict):
        return ""
    return str(json_obj.get("expr", ""))


def run_full_chain(
    backend,
    variant: str,
    *,
    latent_steps: int = 10,
    do_live_coder: bool = True,
    n_coder_attempts: int = 2,
    chain_label: str = "",
) -> dict:
    """Jalankan satu full chain: Propose → Construct → Coder(live) → Feedback.

    Args:
        backend         : latent-enabled LocalLLMBackend.
        variant         : 'old', 'new', atau 'implicit' — semua agent yang
                          parameter-promptnya berubah lintas commit akan pakai
                          variant yang sama untuk konsistensi A/B test.
        latent_steps    : latent injection steps untuk kv_only / kv_and_text.
        do_live_coder   : True → coder LLM beneran (mahal, akurat).
                          False → skip coder, pakai construct expr langsung.
        n_coder_attempts: max retry pada coder (per user: live coder).
        chain_label     : label untuk role tagging di debug snapshots.

    Returns:
        dict berisi prompts + outputs + metrics + KV lens per step.
    """
    from .common import _extract_json
    from .probe import enabled_modes_from_env, run_probes_at

    result: dict[str, Any] = {
        "variant": variant,
        "chain_label": chain_label,
        "latent_steps": latent_steps,
        "do_live_coder": do_live_coder,
    }
    role_tag = f"{chain_label}_{variant}" if chain_label else variant

    # ── Step 1: Propose ─────────────────────────────────────────────────────
    sys_p, usr_p = build_propose_msgs(variant)
    t0 = time.time()
    r_prop = backend.build_messages_and_run(
        user_prompt=usr_p, system_prompt=sys_p,
        json_mode=True, mode="kv_and_text",
        latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role=f"propose_{role_tag}",
    )
    propose_text = r_prop.text or ""
    propose_json = _extract_json(propose_text)
    propose_kv = r_prop.kv_cache
    propose_kv_len = kv_length(propose_kv)
    result["propose"] = {
        "sys_prompt": sys_p, "usr_prompt": usr_p,
        "text": propose_text, "json": propose_json,
        "kv_len": propose_kv_len,
        "elapsed_s": round(time.time() - t0, 2),
    }
    hypothesis_oneline = (propose_json or {}).get("hypothesis", "") or fx.HYPOTHESIS_DICT["hypothesis"]
    print(f"  [{variant}] propose : text_len={len(propose_text)}  kv_len={propose_kv_len}  "
          f"elapsed={result['propose']['elapsed_s']}s  hyp={hypothesis_oneline[:60]!r}")

    # ── Step 2: Construct ───────────────────────────────────────────────────
    sys_c, usr_c = build_construct_msgs(
        variant,
        target_hypothesis=propose_text if variant == "old" else "",
        target_hypothesis_oneline=hypothesis_oneline,
    )
    t1 = time.time()
    r_ctor = backend.build_messages_and_run(
        user_prompt=usr_c, system_prompt=sys_c,
        json_mode=True, mode="kv_and_text",
        past_key_values=propose_kv, latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role=f"construct_{role_tag}",
    )
    construct_text = r_ctor.text or ""
    construct_json = _extract_json(construct_text)
    construct_kv = r_ctor.kv_cache
    construct_kv_len = kv_length(construct_kv)
    factor_name, factor_expr, factor_desc = _extract_construct_first_expr(construct_json)
    result["construct"] = {
        "sys_prompt": sys_c, "usr_prompt": usr_c,
        "text": construct_text, "json": construct_json,
        "factor_name": factor_name, "factor_expr": factor_expr,
        "factor_description": factor_desc,
        "kv_len": construct_kv_len,
        "elapsed_s": round(time.time() - t1, 2),
    }
    print(f"  [{variant}] construct: text_len={len(construct_text)}  kv_len={construct_kv_len}  "
          f"elapsed={result['construct']['elapsed_s']}s  expr={factor_expr[:60]!r}")

    # ── Step 3: Coder (live, dengan retry) ──────────────────────────────────
    if do_live_coder and factor_expr:
        factor_info = (
            f"Factor: {factor_name or 'F1'}\n"
            f"Description: {factor_desc}\n"
            f"Expression (initial): {factor_expr}"
        )
        # Inject synthetic execution feedback supaya coder retry path tersentuh
        synthetic_err = (
            "ValueError: NaN > 5% of output rows.\n"
            "Suggestion: ensure RANK/ZSCORE wrapping for cross-sectional output."
        )
        sys_co, usr_co = build_coder_msgs(factor_info, factor_expr, synthetic_err)
        t2 = time.time()
        coder_attempts = []
        former_expr = factor_expr
        former_fb = synthetic_err
        for ai in range(n_coder_attempts):
            r_co = backend.build_messages_and_run(
                user_prompt=usr_co, system_prompt=sys_co,
                json_mode=True, mode="kv_and_text",
                past_key_values=construct_kv, latent_steps=latent_steps,
                temperature=0.7 + 0.15 * ai, top_p=0.95,
                role=f"coder_{role_tag}_a{ai+1}",
            )
            ct = r_co.text or ""
            cj = _extract_json(ct)
            ce = _extract_coder_expr(cj)
            coder_attempts.append({
                "attempt": ai + 1, "text_len": len(ct),
                "expr": ce, "json_ok": cj is not None,
                "kv_len": kv_length(r_co.kv_cache),
            })
            print(f"    [{variant}] coder a{ai+1}: text_len={len(ct)}  expr={ce[:60]!r}")
            if ce and ce != former_expr:
                former_expr = ce
                break
            # Rebuild user prompt with latest expr+feedback for next attempt
            sys_co, usr_co = build_coder_msgs(factor_info, former_expr, former_fb)
        result["coder"] = {
            "sys_prompt": sys_co, "usr_prompt": usr_co,
            "attempts": coder_attempts,
            "final_expr": former_expr,
            "elapsed_s": round(time.time() - t2, 2),
        }
        coder_final_expr = former_expr
    else:
        result["coder"] = {"skipped": True, "final_expr": factor_expr}
        coder_final_expr = factor_expr

    # ── Step 4: Feedback ────────────────────────────────────────────────────
    # Build factor_summary (new variant) atau task_details (old variant).
    factor_summary_str = f"- {factor_name or 'F1'}: `{coder_final_expr}` [implemented=True]"
    synthetic_bt = _synth_backtest_after(coder_final_expr)
    # Build task_details list (untuk old variant). Mimic shape dari
    # FactorTask.get_task_information_and_implementation_result() di backend.
    synth_task_details = [{
        "factor_name": factor_name or "F1",
        "factor_description": factor_desc,
        "factor_formulation": coder_final_expr,
        "variables": {"$close": "daily close", "$volume": "daily volume"},
        "factor_implementation": True,
        "factor_expression": coder_final_expr,
    }]

    sys_fb, usr_fb = build_feedback_msgs(
        variant,
        hypothesis_text=hypothesis_oneline,
        factor_summary=factor_summary_str,
        task_details=synth_task_details,
        combined_result=synthetic_bt,
        complexity_warnings="",
    )
    t3 = time.time()
    r_fb = backend.build_messages_and_run(
        user_prompt=usr_fb, system_prompt=sys_fb,
        json_mode=True, mode="kv_and_text",
        past_key_values=construct_kv, latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role=f"feedback_{role_tag}",
    )
    fb_text = r_fb.text or ""
    fb_json = _extract_json(fb_text)
    result["feedback"] = {
        "sys_prompt": sys_fb, "usr_prompt": usr_fb,
        "text": fb_text, "json": fb_json,
        "kv_len": kv_length(r_fb.kv_cache),
        "elapsed_s": round(time.time() - t3, 2),
        "synthetic_backtest": synthetic_bt,
    }
    print(f"  [{variant}] feedback: text_len={len(fb_text)}  schema_ok={fb_json is not None}  "
          f"elapsed={result['feedback']['elapsed_s']}s")

    # ── Optional KV probing (TEST_PROBE env) ────────────────────────────────
    probe_modes = enabled_modes_from_env()
    if probe_modes:
        probes_construct = run_probes_at(
            backend, construct_kv,
            kv_label=f"construct_kv_{variant}", modes=probe_modes,
        )
        result["probes_construct"] = probes_construct

    return result
