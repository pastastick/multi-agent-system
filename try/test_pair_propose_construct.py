"""
Pair test: Propose ↔ Construct dengan 3 skenario seed KV berbeda.

Tujuan: investigasi bug "Construct collapse post-MUTATION" (lihat
debug session 20260501_135425, panggilan 0015-0017).

Pola yang diobservasi di production:
  - Mutation (call 0013) menghasilkan JSON dengan key BERBEDA dari format propose:
    {"new_hypothesis", "reasoning", "evaluation_metrics", "expected_characteristics"}
  - Propose post-mutation (0014) menerima KV dari mutation → output collapse ke
    format feedback: {"Observations", "New Hypothesis", "Replace Best Result"}
  - Construct (0015-0017) menerima KV dari propose-yang-salah-format → collapse:
    flat dict → echo placeholder <factor_name_A> → repetisi infinite

Skenario test:
  A. fresh_start         → Propose tanpa KV seed (baseline)
  B. mutation_seeded     → Mutation live → KV → Propose seed → Construct
                           (REPRODUKSI BUG 1)
  C. feedback_chained    → Feedback live → KV → Propose seed → Construct
                           (chain normal antar-iterasi)

Untuk setiap skenario diukur:
  - Propose JSON validity & format match (apakah struktur key sesuai propose schema?)
  - Construct schema correctness (nested 4-key vs flat dict vs collapse)
  - Text length & duration → deteksi pelambatan generasi
  - KV length pertumbuhan
  - Collapse detection (text_len < 50 atau repetition ratio < 0.15)

Pemakaian:
    python -m try.run --group pair_propose_construct --case fresh_start
    python -m try.run --group pair_propose_construct --case mutation_seeded
    python -m try.run --group pair_propose_construct --case feedback_chained
    python -m try.run --group pair_propose_construct  # semua

Dry-run:
    python -m try.run --group pair_propose_construct --dry-run
"""

from __future__ import annotations

import json as _json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, StrictUndefined

from .common import (
    load_yaml, PROMPT_PATHS, get_latent_backend, _extract_json,
)
from .config import CONFIG
from . import fixtures as fx
from .probe import (
    enabled_modes_from_env, run_probes_at,
    format_probes_for_log, print_probe_summary,
)
from .prompt_ab import (
    PROMPT_VARIANTS, run_full_chain,
    build_propose_msgs, build_construct_msgs, build_feedback_msgs,
    diff_prompt_structure, render_prompt_diff_table,
    concat_kv_raw, kv_length,
)


# ─── Helpers shared (mirror dari test_multi_agent_kv) ────────────────────────

def _jinja(template: str, **kw) -> str:
    return Environment(undefined=StrictUndefined).from_string(template).render(**kw)


def _factors_yaml() -> dict:
    return load_yaml(PROMPT_PATHS["factors_prompts"])


def _evolution_yaml() -> dict:
    return load_yaml(PROMPT_PATHS["evolution"])


def _kv_len(kv) -> int:
    try:
        from backend.llm.models import _past_length
        return _past_length(kv)
    except Exception:
        return -1


def _is_collapse(text: str) -> bool:
    """Deteksi output degenerate (mirror backend logic)."""
    if not text or not text.strip() or len(text.strip()) < 50:
        return True
    words = text.split()
    if len(words) < 8:
        return True
    return len(set(words)) / len(words) < 0.15


def _has_propose_schema(parsed: dict | None) -> bool:
    """Cek apakah output propose memiliki 5 key wajib."""
    if not parsed or not isinstance(parsed, dict):
        return False
    REQUIRED = {"hypothesis", "concise_knowledge", "concise_observation",
                "concise_justification", "concise_specification"}
    return REQUIRED.issubset(set(parsed.keys()))


def _has_construct_schema(text_or_json) -> bool:
    """Cek apakah output construct well-formed.

    Format baru (post-commit 94e873d): plain text NAME:/DESC:/EXPR: per factor.
    Format lama (fallback): nested JSON dict dengan key description/expression/...

    Menerima raw text string ATAU parsed dict agar caller tidak perlu diubah.
    """
    # ── Plain text format: cukup ada ≥1 baris EXPR: yang non-empty ──────────
    if isinstance(text_or_json, str):
        return any(
            ln.strip().upper().startswith("EXPR:") and ln.strip()[5:].strip()
            for ln in text_or_json.splitlines()
        )
    # ── JSON fallback (format lama) ──────────────────────────────────────────
    if not text_or_json or not isinstance(text_or_json, dict):
        return False
    for v in text_or_json.values():
        if not isinstance(v, dict):
            return False
        if not all(k in v for k in ("description", "expression")):
            return False
    return True


def _log_save(log_path: Path, sections: list[tuple[str, str]]) -> None:
    lines = [f"# Log: {log_path.name}", f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for title, content in sections:
        lines += ["=" * 78, title, "=" * 78, content, ""]
    log_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Message builders (Propose / Construct / Mutation / Feedback) ────────────

def _render_hf(trace, limit: int = 6) -> str:
    y = _factors_yaml()
    if len(trace.hist) == 0:
        return "No previous hypothesis and feedback available since it's the first round."
    lt = SimpleNamespace(scen=trace.scen, hist=trace.hist[-limit:])
    return _jinja(y["hypothesis_and_feedback"], trace=lt)


def _build_propose_msgs() -> tuple[str, str]:
    y = _factors_yaml()
    trace = fx.TRACE
    scen_desc = trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")
    hf = _render_hf(trace)
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


def _build_construct_msgs(target_hypothesis_oneline: str = "") -> tuple[str, str]:
    """Build construct prompts sesuai format baru (post-commit 94e873d).

    Template hypothesis2experiment berubah:
      - system: tidak lagi pakai {{ targets }} / {{ scenario }}, hanya {{ experiment_output_format }}
      - user  : {{ target_hypothesis_oneline }} (bukan target_hypothesis),
                tidak ada hypothesis_and_feedback / target_list / RAG
    Output model: plain text NAME:/DESC:/EXPR: (bukan JSON nested)
    """
    y = _factors_yaml()
    sys_c = _jinja(
        y["hypothesis2experiment"]["system_prompt"],
        experiment_output_format=y["experiment_output_format"],
    )
    usr_c = _jinja(
        y["hypothesis2experiment"]["user_prompt"],
        targets="factor",
        target_hypothesis_oneline=target_hypothesis_oneline or fx.HYPOTHESIS_DICT["hypothesis"],
        function_lib_description=y["function_lib_description"],
        expression_duplication=None,
    )
    return sys_c, usr_c


def _build_mutation_msgs() -> tuple[str, str]:
    """
    Render mutation prompt persis seperti pipeline/evolution/mutation.py.
    Pakai fixtures.PARENT_* untuk simulasi parent trajectory.
    """
    y = _evolution_yaml()
    mp = y["mutation"]
    sys_m = mp["system"]
    usr_m = mp["user"].format(
        parent_hypothesis=fx.PARENT_HYPOTHESIS,
        parent_factors=fx.PARENT_FACTORS_STR,
        parent_metrics=fx.PARENT_METRICS_STR,
        parent_feedback=fx.PARENT_FEEDBACK_STR,
    )
    return sys_m, usr_m


def _build_feedback_msgs() -> tuple[str, str]:
    y = _factors_yaml()
    # System prompt (new format tidak pakai {{ scenario }})
    sys_f = y["factor_feedback_generation"]["system"]
    # Build factor_summary compact string dari fixture (format baru: hypothesis_oneline +
    # factor_summary, bukan lagi hypothesis_text + task_details)
    factor_summary = (
        f"- {fx.FACTOR_TASK.factor_name}: "
        f"`{fx.FACTOR_TASK.factor_expression}` "
        f"[implemented={fx.FACTOR_TASK.factor_implementation}]"
    )
    usr_f = _jinja(
        y["factor_feedback_generation"]["user"],
        hypothesis_oneline=fx.HYPOTHESIS_DICT["hypothesis"],
        factor_summary=factor_summary,
        complexity_warnings="",
        combined_result=fx.COMBINED_RESULT_STR,
    )
    return sys_f, usr_f


# ─── Core measurement ────────────────────────────────────────────────────────

def _measure_propose_construct(
    backend, propose_sys: str, propose_usr: str,
    construct_sys: str, construct_usr: str,
    seed_kv, latent_steps: int, scenario_label: str,
) -> dict:
    """
    Jalankan Propose (kv_only, dengan optional seed) → Construct (kv_and_text).
    Return semua metrik untuk analisis.

    Jika env var TEST_PROBE diset, probe diagnostik dijalankan di 2 titik:
      1. Setelah propose_kv terbentuk (sebelum construct memakai KV-nya) —
         menjawab: "apa yang construct akan 'lihat' dari KV propose?"
      2. Setelah construct selesai — menjawab: "apa yang ter-encode di
         KV setelah propose+construct gabungan?"
      Probe TIDAK mempengaruhi metrik utama (KV di-crop kembali setelahnya).
    """
    temp, top_p = 0.7, 0.95
    probe_modes = enabled_modes_from_env()
    probes: list = []

    # Propose: kv_only, dengan seed (None untuk fresh, KV untuk seeded)
    print(f"  [{scenario_label}] Propose (kv_only, seed={'YES' if seed_kv is not None else 'NO'})...")
    t0 = time.time()
    r_prop = backend.build_messages_and_run(
        user_prompt=propose_usr, system_prompt=propose_sys,
        mode="kv_only", latent_steps=latent_steps,
        past_key_values=seed_kv,
        temperature=temp, top_p=top_p, role=f"propose_{scenario_label}",
    )
    propose_kv = r_prop.kv_cache
    propose_kv_len = _kv_len(propose_kv)
    propose_elapsed = round(time.time() - t0, 2)
    print(f"    elapsed={propose_elapsed}s  kv_len={propose_kv_len}")

    # Probe checkpoint #1: introspeksi propose_kv sebelum construct memakainya
    if probe_modes:
        print(f"  [{scenario_label}] Probing propose_kv ({len(probe_modes)} mode)...")
        kv_label = f"propose_kv_{scenario_label}"
        probe1 = run_probes_at(backend, propose_kv, kv_label=kv_label, modes=probe_modes)
        print_probe_summary(probe1)
        probes.extend(probe1)

    # Diagnostik propose: jalankan sekali lagi text_only untuk lihat apa yang
    # diproduksi (probe). Kita tidak ambil KV dari ini, hanya untuk inspeksi format.
    print(f"  [{scenario_label}] Propose probe (text_only, seed={'YES' if seed_kv is not None else 'NO'})...")
    t1 = time.time()
    r_prop_text = backend.build_messages_and_run(
        user_prompt=propose_usr, system_prompt=propose_sys,
        json_mode=True, mode="text_only" if seed_kv is None else "kv_and_text",
        past_key_values=seed_kv,
        temperature=temp, top_p=top_p, role=f"propose_text_{scenario_label}",
    )
    propose_text = r_prop_text.text or ""
    propose_json = _extract_json(propose_text)
    propose_schema_ok = _has_propose_schema(propose_json)
    propose_text_elapsed = round(time.time() - t1, 2)
    propose_text_len = len(propose_text)
    propose_collapse = _is_collapse(propose_text)
    print(f"    text_len={propose_text_len}  schema_ok={propose_schema_ok}  collapse={propose_collapse}")

    # Construct: kv_and_text dari propose_kv
    print(f"  [{scenario_label}] Construct (kv_and_text, dari propose_kv)...")
    t2 = time.time()
    r_ctor = backend.build_messages_and_run(
        user_prompt=construct_usr, system_prompt=construct_sys,
        json_mode=False, mode="kv_and_text",   # format baru: plain text NAME:/DESC:/EXPR:
        past_key_values=propose_kv, latent_steps=latent_steps,
        temperature=temp, top_p=top_p, role=f"construct_{scenario_label}",
    )
    construct_text = r_ctor.text or ""
    construct_json = _extract_json(construct_text)  # fallback untuk logging saja
    construct_schema_ok = _has_construct_schema(construct_text)
    construct_kv = r_ctor.kv_cache
    construct_kv_len = _kv_len(construct_kv)
    construct_elapsed = round(time.time() - t2, 2)
    construct_text_len = len(construct_text)
    construct_collapse = _is_collapse(construct_text)
    print(f"    text_len={construct_text_len}  schema_ok={construct_schema_ok}  "
          f"collapse={construct_collapse}  elapsed={construct_elapsed}s  kv_len={construct_kv_len}")

    # Probe checkpoint #2: introspeksi construct_kv (state akhir gabungan)
    if probe_modes:
        print(f"  [{scenario_label}] Probing construct_kv ({len(probe_modes)} mode)...")
        kv_label = f"construct_kv_{scenario_label}"
        probe2 = run_probes_at(backend, construct_kv, kv_label=kv_label, modes=probe_modes)
        print_probe_summary(probe2)
        probes.extend(probe2)

    return {
        "scenario": scenario_label,
        "propose_kv_len": propose_kv_len,
        "propose_elapsed_s": propose_elapsed,
        "propose_text": propose_text,
        "propose_json": propose_json,
        "propose_text_len": propose_text_len,
        "propose_text_elapsed_s": propose_text_elapsed,
        "propose_schema_ok": propose_schema_ok,
        "propose_collapse": propose_collapse,
        "construct_text": construct_text,
        "construct_json": construct_json,
        "construct_text_len": construct_text_len,
        "construct_elapsed_s": construct_elapsed,
        "construct_schema_ok": construct_schema_ok,
        "construct_collapse": construct_collapse,
        "construct_kv_len": construct_kv_len,
        "probes": probes,
    }


def _save_scenario_log(log_path: Path, label: str, m: dict, extra_sections: list = None):
    sections = [
        (f"SCENARIO: {label}", (
            f"Propose : kv_len={m['propose_kv_len']}  text_len={m['propose_text_len']}  "
            f"schema_ok={m['propose_schema_ok']}  collapse={m['propose_collapse']}  "
            f"elapsed={m['propose_elapsed_s']}s\n"
            f"Construct: kv_len={m['construct_kv_len']}  text_len={m['construct_text_len']}  "
            f"schema_ok={m['construct_schema_ok']}  collapse={m['construct_collapse']}  "
            f"elapsed={m['construct_elapsed_s']}s"
        )),
    ]
    if extra_sections:
        sections.extend(extra_sections)
    sections += [
        ("PROPOSE PROBE OUTPUT (text_only/kv_and_text)", m["propose_text"]),
        ("CONSTRUCT OUTPUT", m["construct_text"]),
        ("PARSED PROPOSE JSON", _json.dumps(m["propose_json"], indent=2, ensure_ascii=False)
            if m["propose_json"] else "(parse failed)"),
        ("PARSED CONSTRUCT JSON", _json.dumps(m["construct_json"], indent=2, ensure_ascii=False)
            if m["construct_json"] else "(parse failed)"),
    ]
    # KV-introspection probes (only present when TEST_PROBE env var was set)
    probes = m.get("probes") or []
    if probes:
        sections.extend(format_probes_for_log(probes))
    _log_save(log_path, sections)


# ─── Test A: fresh_start (baseline) ──────────────────────────────────────────

def test_fresh_start(latent_steps: int | None = None) -> dict:
    """
    Baseline: Propose tanpa seed KV → Construct.
    Skenario "round 1" yang seharusnya selalu sukses.
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    group, case = "pair_propose_construct", f"fresh_start_m{latent_steps}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  (BASELINE: round 1, no seed KV)")
    print("═" * 78)

    if CONFIG.dry_run:
        s, u = _build_propose_msgs()
        print("── PROPOSE (no seed KV) ──"); print(s[:400])
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()
    propose_sys, propose_usr = _build_propose_msgs()
    construct_sys, construct_usr = _build_construct_msgs(target_hypothesis_oneline="")

    m = _measure_propose_construct(
        backend, propose_sys, propose_usr, construct_sys, construct_usr,
        seed_kv=None, latent_steps=latent_steps, scenario_label="fresh",
    )
    _save_scenario_log(log_path, "FRESH START (no seed KV)", m)
    print(f"\n── LOG SAVED ── {log_path}")

    return {
        "group": group, "case": case, "latent_steps": latent_steps,
        "ok_format": m["propose_schema_ok"] and m["construct_schema_ok"],
        "elapsed_s": round(m["propose_elapsed_s"] + m["propose_text_elapsed_s"]
                           + m["construct_elapsed_s"], 2),
        "log_path": str(log_path),
        "metrics": m,
        "parsed": m["construct_json"],
    }


# ─── Test B: mutation_seeded (REPRODUKSI BUG 1) ──────────────────────────────

def test_mutation_seeded(latent_steps: int | None = None) -> dict:
    """
    Reproduksi bug post-MUTATION: Mutation live → KV → Propose → Construct.

    Mutation prompt menghasilkan JSON dengan format berbeda dari propose schema.
    KV dari mutation ini ketika di-seed ke propose menyebabkan model collapse:
    propose generate format feedback (Observations, ...) bukan hypothesis schema,
    dan construct yang downstream juga collapse ke flat dict atau placeholder echo.
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    group, case = "pair_propose_construct", f"mutation_seeded_m{latent_steps}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  (REPRODUKSI BUG 1: mutation seed → propose → construct)")
    print("═" * 78)

    if CONFIG.dry_run:
        sm, um = _build_mutation_msgs()
        print("── MUTATION (live call to seed KV) ──"); print(sm[:400])
        print("── USER ──"); print(um[:400])
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()

    # Step 1: Mutation live untuk dapat seed KV
    print("── Step 1: Mutation live (kv_only) untuk hasilkan seed KV ──")
    mut_sys, mut_usr = _build_mutation_msgs()
    t0 = time.time()
    r_mut = backend.build_messages_and_run(
        user_prompt=mut_usr, system_prompt=mut_sys,
        mode="kv_only", latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role="mutation_seed",
    )
    mut_kv = r_mut.kv_cache
    mut_kv_len = _kv_len(mut_kv)
    mut_elapsed = round(time.time() - t0, 2)
    print(f"  mutation done: kv_len={mut_kv_len}  elapsed={mut_elapsed}s")

    # Probe seed mutation_kv sebelum propose memakainya — ini titik paling
    # informatif untuk Bug 1, karena di sinilah polusi format bermula
    seed_probes = run_probes_at(backend, mut_kv, kv_label="mutation_seed_kv")
    if seed_probes:
        print_probe_summary(seed_probes)

    # Step 1b: Mutation text probe — lihat output format mutation untuk konfirmasi
    print("── Step 1b: Mutation text probe (untuk inspeksi format JSON) ──")
    t1 = time.time()
    r_mut_text = backend.build_messages_and_run(
        user_prompt=mut_usr, system_prompt=mut_sys,
        json_mode=True, mode="text_only",
        temperature=0.7, top_p=0.95, role="mutation_text_probe",
    )
    mut_text = r_mut_text.text or ""
    mut_text_elapsed = round(time.time() - t1, 2)
    print(f"  mutation_text len={len(mut_text)}  elapsed={mut_text_elapsed}s")
    print(f"  preview: {mut_text[:200]!r}")

    # Step 2: Propose dengan seed KV dari mutation → Construct
    print("\n── Step 2: Propose+Construct dengan mutation_kv sebagai seed ──")
    propose_sys, propose_usr = _build_propose_msgs()
    construct_sys, construct_usr = _build_construct_msgs(target_hypothesis_oneline="")

    m = _measure_propose_construct(
        backend, propose_sys, propose_usr, construct_sys, construct_usr,
        seed_kv=mut_kv, latent_steps=latent_steps, scenario_label="mut_seeded",
    )

    # Bandingkan: apakah propose_text berisi format mutation/feedback bukan hypothesis?
    pollution_indicators = []
    pt_lower = m["propose_text"].lower()
    for bad_key in ["new_hypothesis", "evaluation_metrics", "expected_characteristics",
                    "observations", "feedback for hypothesis", "replace best result",
                    "<factor_name", "orthogonality"]:
        if bad_key in pt_lower:
            pollution_indicators.append(bad_key)

    print(f"\n  ► Propose pollution indicators: {pollution_indicators or '(none — clean)'}")
    print(f"  ► Construct schema OK: {m['construct_schema_ok']}")
    if not m["construct_schema_ok"] and m["construct_json"]:
        keys = list(m["construct_json"].keys())[:5]
        print(f"  ► Construct flat-dict keys (collapse evidence): {keys}")

    extra = [
        (f"MUTATION SEED  kv_len={mut_kv_len}  elapsed={mut_elapsed}s", mut_text),
        ("POLLUTION INDICATORS in propose output",
         ", ".join(pollution_indicators) if pollution_indicators else "(none)"),
    ]
    if seed_probes:
        extra.extend(format_probes_for_log(seed_probes))
    _save_scenario_log(log_path, "MUTATION-SEEDED PROPOSE+CONSTRUCT", m, extra_sections=extra)
    print(f"\n── LOG SAVED ── {log_path}")

    return {
        "group": group, "case": case, "latent_steps": latent_steps,
        "ok_format": m["propose_schema_ok"] and m["construct_schema_ok"],
        "elapsed_s": round(mut_elapsed + mut_text_elapsed + m["propose_elapsed_s"]
                           + m["propose_text_elapsed_s"] + m["construct_elapsed_s"], 2),
        "log_path": str(log_path),
        "mutation_kv_len": mut_kv_len,
        "pollution_indicators": pollution_indicators,
        "metrics": m,
        "parsed": m["construct_json"],
    }


# ─── Test C: feedback_chained (chain antar-iterasi normal) ───────────────────

def test_feedback_chained(latent_steps: int | None = None) -> dict:
    """
    Chain antar-iterasi normal: Feedback live → KV → Propose → Construct.

    Berbeda dari mutation_seeded — feedback prompt menghasilkan JSON dengan
    schema yang lebih dekat ke propose. Test ini cek apakah chain via feedback
    KV juga menyebabkan format collapse, atau lebih stabil.
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    group, case = "pair_propose_construct", f"feedback_chained_m{latent_steps}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  (chain via feedback KV — chain antar-iterasi normal)")
    print("═" * 78)

    if CONFIG.dry_run:
        sf, uf = _build_feedback_msgs()
        print("── FEEDBACK (live call to seed KV) ──"); print(sf[:400])
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()

    # Step 1: Feedback live untuk dapat seed KV
    print("── Step 1: Feedback live (kv_only) untuk hasilkan seed KV ──")
    fb_sys, fb_usr = _build_feedback_msgs()
    t0 = time.time()
    r_fb = backend.build_messages_and_run(
        user_prompt=fb_usr, system_prompt=fb_sys,
        mode="kv_only", latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role="feedback_seed",
    )
    fb_kv = r_fb.kv_cache
    fb_kv_len = _kv_len(fb_kv)
    fb_elapsed = round(time.time() - t0, 2)
    print(f"  feedback done: kv_len={fb_kv_len}  elapsed={fb_elapsed}s")

    # Probe feedback seed KV sebelum propose memakainya — counterfactual ke
    # mutation case: kalau probe ini SAMA "rapi"-nya dengan probe mutation,
    # berarti bug bukan di KV pollution melainkan format-prompt collision
    seed_probes = run_probes_at(backend, fb_kv, kv_label="feedback_seed_kv")
    if seed_probes:
        print_probe_summary(seed_probes)

    # Step 2: Propose dengan seed KV dari feedback → Construct
    print("\n── Step 2: Propose+Construct dengan feedback_kv sebagai seed ──")
    propose_sys, propose_usr = _build_propose_msgs()
    construct_sys, construct_usr = _build_construct_msgs(target_hypothesis_oneline="")

    m = _measure_propose_construct(
        backend, propose_sys, propose_usr, construct_sys, construct_usr,
        seed_kv=fb_kv, latent_steps=latent_steps, scenario_label="fb_chained",
    )

    extra = [(f"FEEDBACK SEED  kv_len={fb_kv_len}  elapsed={fb_elapsed}s",
              "(KV dari feedback step — bentuk chain antar-iterasi normal)")]
    if seed_probes:
        extra.extend(format_probes_for_log(seed_probes))
    _save_scenario_log(log_path, "FEEDBACK-CHAINED PROPOSE+CONSTRUCT", m, extra_sections=extra)
    print(f"\n── LOG SAVED ── {log_path}")

    return {
        "group": group, "case": case, "latent_steps": latent_steps,
        "ok_format": m["propose_schema_ok"] and m["construct_schema_ok"],
        "elapsed_s": round(fb_elapsed + m["propose_elapsed_s"]
                           + m["propose_text_elapsed_s"] + m["construct_elapsed_s"], 2),
        "log_path": str(log_path),
        "feedback_kv_len": fb_kv_len,
        "metrics": m,
        "parsed": m["construct_json"],
    }


# ─── Test D: prompt_ab_chain (A/B/C comparison: full chain hingga feedback) ─

def test_prompt_ab_chain(latent_steps: int | None = None) -> dict:
    """
    Bandingkan 3 variant prompt (old/new/implicit) lewat FULL CHAIN:
    Propose → Construct → Coder (live) → Feedback (synthetic backtest).

    Setiap variant pakai SAMA fixtures + SAMA backend instance, jadi
    perbedaan output murni karena perbedaan prompt + KV-cache transfer.

    Yang diukur per variant:
      - Per-stage text_len & kv_len (apakah chain stabil?)
      - schema_ok di propose/construct/feedback
      - Coder retry behavior
      - Quantitative prompt diff (XML tags, hypothesis anchor, implicit phrases)

    Membantu jawab:
      a) Apakah verbosity OLD prompt (XML markup) bantu/sakit model kecil?
      b) Apakah "hypothesis written explicitly" >> "in latent KV" ketika
         model di-bias dengan kalimat 'see prior context'?
      c) Bagaimana isi KV-cache yang ter-transfer (via probe)?
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    group, case = "pair_propose_construct", f"prompt_ab_chain_m{latent_steps}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  (A/B/C chain — old vs new vs implicit)")
    print("═" * 78)

    # Render & diff prompts BEFORE any LLM call — sehingga walau dry_run
    # bagian comparisonnya tetap informatif.
    diffs_construct: dict[str, dict] = {}
    diffs_feedback: dict[str, dict] = {}
    rendered_construct: dict[str, str] = {}
    rendered_feedback: dict[str, str] = {}
    for v in PROMPT_VARIANTS:
        _sys_c, usr_c = build_construct_msgs(v)
        _sys_f, usr_f = build_feedback_msgs(v, factor_summary="- F1: `RANK($volume)` [implemented=True]")
        rendered_construct[v] = usr_c
        rendered_feedback[v] = usr_f
    # Pairwise diffs vs 'new' baseline
    diffs_construct = diff_prompt_structure(
        "old", rendered_construct["old"], "new", rendered_construct["new"]
    )
    diffs_construct_vs_impl = diff_prompt_structure(
        "new", rendered_construct["new"], "implicit", rendered_construct["implicit"]
    )
    diffs_feedback = diff_prompt_structure(
        "old", rendered_feedback["old"], "new", rendered_feedback["new"]
    )

    print("\n── Prompt diff (CONSTRUCT user) old vs new ──")
    print(render_prompt_diff_table(diffs_construct))
    print("\n── Prompt diff (CONSTRUCT user) new vs implicit ──")
    print(render_prompt_diff_table(diffs_construct_vs_impl))
    print("\n── Prompt diff (FEEDBACK user) old vs new ──")
    print(render_prompt_diff_table(diffs_feedback))

    if CONFIG.dry_run:
        sections = [
            ("PROMPT DIFF — CONSTRUCT user (old vs new)", render_prompt_diff_table(diffs_construct)),
            ("PROMPT DIFF — CONSTRUCT user (new vs implicit)",
             render_prompt_diff_table(diffs_construct_vs_impl)),
            ("PROMPT DIFF — FEEDBACK user (old vs new)", render_prompt_diff_table(diffs_feedback)),
        ]
        for v in PROMPT_VARIANTS:
            sections.append((f"RENDERED CONSTRUCT USER [{v}]", rendered_construct[v]))
            sections.append((f"RENDERED FEEDBACK USER [{v}]", rendered_feedback[v]))
        _log_save(log_path, sections)
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0,
                "log_path": str(log_path),
                "diffs": {"construct_old_vs_new": diffs_construct,
                          "construct_new_vs_implicit": diffs_construct_vs_impl,
                          "feedback_old_vs_new": diffs_feedback}}

    backend = get_latent_backend()
    chains: dict[str, dict] = {}
    n_attempts = int(os.environ.get("TEST_AB_CODER_ATTEMPTS", "2"))
    for v in PROMPT_VARIANTS:
        print("\n" + "─" * 78)
        print(f"  ▶ variant={v!r}")
        print("─" * 78)
        t0 = time.time()
        chains[v] = run_full_chain(
            backend, v,
            latent_steps=latent_steps,
            do_live_coder=True,
            n_coder_attempts=n_attempts,
            chain_label="ab",
        )
        chains[v]["total_chain_s"] = round(time.time() - t0, 2)

    # ── Ringkasan tabel cross-variant ───────────────────────────────────────
    print("\n── CROSS-VARIANT SUMMARY ──")
    header = (f"{'variant':<9} | {'prop_len':>8} | {'prop_kv':>7} | "
              f"{'ctor_len':>8} | {'ctor_kv':>7} | {'ctor_expr':>30} | "
              f"{'fb_len':>6} | {'fb_ok':>5} | {'total_s':>8}")
    print(header); print("─" * len(header))
    for v in PROMPT_VARIANTS:
        c = chains[v]
        expr = (c["construct"].get("factor_expr") or "")[:30]
        fb = c["feedback"]
        print(f"{v:<9} | {len(c['propose']['text']):>8} | {c['propose']['kv_len']:>7} | "
              f"{len(c['construct']['text']):>8} | {c['construct']['kv_len']:>7} | "
              f"{expr:>30} | {len(fb['text']):>6} | "
              f"{('Y' if fb['json'] else 'N'):>5} | {c['total_chain_s']:>7.2f}s")

    # ── Save log ────────────────────────────────────────────────────────────
    sections: list[tuple[str, str]] = [
        ("PROMPT DIFF — CONSTRUCT user (old vs new)", render_prompt_diff_table(diffs_construct)),
        ("PROMPT DIFF — CONSTRUCT user (new vs implicit)",
         render_prompt_diff_table(diffs_construct_vs_impl)),
        ("PROMPT DIFF — FEEDBACK user (old vs new)", render_prompt_diff_table(diffs_feedback)),
    ]
    for v in PROMPT_VARIANTS:
        c = chains[v]
        sections.append((f"=== VARIANT [{v}]  total={c['total_chain_s']}s ===",
                         f"propose_kv={c['propose']['kv_len']}  construct_kv={c['construct']['kv_len']}"))
        sections.append((f"[{v}] PROPOSE TEXT (len={len(c['propose']['text'])})", c["propose"]["text"]))
        sections.append((f"[{v}] CONSTRUCT TEXT (len={len(c['construct']['text'])})", c["construct"]["text"]))
        if "attempts" in c.get("coder", {}):
            attempts_str = "\n".join(
                f"  a{a['attempt']}: text_len={a['text_len']}  expr={a['expr']!r}  ok={a['json_ok']}"
                for a in c["coder"]["attempts"]
            )
            sections.append((f"[{v}] CODER ATTEMPTS", attempts_str))
        sections.append((f"[{v}] FEEDBACK TEXT (len={len(c['feedback']['text'])})", c["feedback"]["text"]))
        if c.get("probes_construct"):
            sections.extend(format_probes_for_log(c["probes_construct"]))
    _log_save(log_path, sections)
    print(f"\n── LOG SAVED ── {log_path}")

    return {
        "group": group, "case": case, "latent_steps": latent_steps,
        "ok_format": all(chains[v]["feedback"]["json"] is not None for v in PROMPT_VARIANTS),
        "elapsed_s": round(sum(chains[v]["total_chain_s"] for v in PROMPT_VARIANTS), 2),
        "log_path": str(log_path),
        "chains": chains,
        "diffs": {
            "construct_old_vs_new": diffs_construct,
            "construct_new_vs_implicit": diffs_construct_vs_impl,
            "feedback_old_vs_new": diffs_feedback,
        },
        "parsed": None,
    }


# ─── Test E: kv_combine_raw (BUKAN seed-extend — concat mentah, RoPE rusak) ──

def test_kv_combine_raw(latent_steps: int | None = None) -> dict:
    """
    Eksperimen: concat propose_kv + feedback_kv MENTAH (tanpa fix-up posisi
    RoPE), lalu pakai sebagai past_key_values untuk construct.

    Expected behavior (per RoPE math):
      Keys feedback_kv di-encode di RoPE positions [0..L_b], tapi dalam
      combined cache mereka di sequence positions [L_a..L_a+L_b]. Query
      construct akan compute attention scores WRT keys yang RoPE-nya
      mismatch dengan posisi mereka → atensi melenceng → output collapse
      atau halusinasi.

    Diagnostik:
      - Bandingkan output construct dengan SAME prompt ke propose_kv saja
        (baseline) dan ke corrected chain (test_kv_combine_corrected).
      - Probe combined_kv: apakah model masih bisa baca scenario / hipotesis?
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    group, case = "pair_propose_construct", f"kv_combine_raw_m{latent_steps}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  (RAW KV concat — no RoPE fix-up)")
    print("═" * 78)

    if CONFIG.dry_run:
        print("dry_run: would build propose_kv + feedback_kv, concat raw, then construct")
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()

    # Step 1: Build propose_kv (kv_only — KV saja, no text needed)
    print("── Step 1: Propose kv_only → propose_kv ──")
    sys_p, usr_p = build_propose_msgs("new")
    t0 = time.time()
    r_prop = backend.build_messages_and_run(
        user_prompt=usr_p, system_prompt=sys_p,
        mode="kv_only", latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role="kvcomb_raw_propose",
    )
    propose_kv = r_prop.kv_cache
    L_a = kv_length(propose_kv)
    elapsed_a = round(time.time() - t0, 2)
    print(f"  propose_kv len={L_a}  elapsed={elapsed_a}s")

    # Step 2: Build feedback_kv (kv_only, INDEPENDENT — tidak seeded dari propose)
    # Ini krusial untuk eksperimen "raw concat" — kedua KV harus dibangun
    # independent supaya posisi mereka benar-benar tumpang tindih.
    print("── Step 2: Feedback kv_only INDEPENDENT (no seed) → feedback_kv ──")
    sys_f, usr_f = build_feedback_msgs(
        "new",
        hypothesis_text=fx.HYPOTHESIS_DICT["hypothesis"],
        factor_summary="- F1: `RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))` [implemented=True]",
        combined_result=fx.COMBINED_RESULT_STR,
    )
    t1 = time.time()
    r_fb = backend.build_messages_and_run(
        user_prompt=usr_f, system_prompt=sys_f,
        mode="kv_only", latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role="kvcomb_raw_feedback",
    )
    feedback_kv = r_fb.kv_cache
    L_b = kv_length(feedback_kv)
    elapsed_b = round(time.time() - t1, 2)
    print(f"  feedback_kv len={L_b}  elapsed={elapsed_b}s")

    # Step 3: RAW concat
    print(f"── Step 3: concat_kv_raw(propose_kv, feedback_kv) — RoPE positions WILL DRIFT ──")
    combined = concat_kv_raw(propose_kv, feedback_kv)
    L_c = kv_length(combined)
    expected = L_a + L_b
    print(f"  combined_kv len={L_c}  (expected {expected}, drift={L_c - expected})")

    # Step 4: Probe combined_kv — apa yang model "lihat"?
    print("── Step 4: Probing combined_kv ──")
    probe_modes = enabled_modes_from_env() or ["rewrite", "format_check"]
    probes = run_probes_at(backend, combined, kv_label="combined_raw_kv", modes=probe_modes)
    if probes:
        print_probe_summary(probes)

    # Step 5: Run construct dengan combined_kv sebagai past_key_values
    print("── Step 5: Construct dengan combined_kv (raw) ──")
    sys_c, usr_c = build_construct_msgs("new")
    t2 = time.time()
    r_ctor = backend.build_messages_and_run(
        user_prompt=usr_c, system_prompt=sys_c,
        json_mode=False, mode="kv_and_text",
        past_key_values=combined, latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role="kvcomb_raw_construct",
    )
    construct_text = r_ctor.text or ""
    construct_json = _extract_json(construct_text)
    elapsed_c = round(time.time() - t2, 2)
    print(f"  construct: text_len={len(construct_text)}  schema_ok={_has_construct_schema(construct_text)}  "
          f"elapsed={elapsed_c}s")
    print(f"  preview: {construct_text[:200]!r}")

    sections = [
        (f"RAW CONCAT EXPERIMENT  L_a={L_a}  L_b={L_b}  L_combined={L_c}  "
         f"expected={expected}  RoPE drift expected for B's positions", ""),
        ("CONSTRUCT OUTPUT (under raw-combined KV)", construct_text),
        ("PARSED JSON", _json.dumps(construct_json, indent=2, ensure_ascii=False)
         if construct_json else "(parse failed)"),
    ]
    sections.extend(format_probes_for_log(probes))
    _log_save(log_path, sections)
    print(f"\n── LOG SAVED ── {log_path}")

    return {
        "group": group, "case": case, "latent_steps": latent_steps,
        "ok_format": _has_construct_schema(construct_text),
        "elapsed_s": round(elapsed_a + elapsed_b + elapsed_c, 2),
        "log_path": str(log_path),
        "kv_lens": {"propose": L_a, "feedback": L_b, "combined": L_c, "expected": expected},
        "construct_text_len": len(construct_text),
        "parsed": construct_json,
    }


# ─── Test F: kv_combine_corrected (seed-and-extend, RoPE benar) ──────────────

def test_kv_combine_corrected(latent_steps: int | None = None) -> dict:
    """
    Counterfactual ke test_kv_combine_raw: kombinasi dua konteks lewat
    SEED-AND-EXTEND. propose_kv menjadi seed, lalu feedback prompt
    di-PROCESS DI ATAS-nya (positions di-RoPE di posisi yang benar).

    Hasil corrected_kv: model "tahu" propose context + feedback context
    dengan urutan dan positions yang konsisten. Inilah cara codebase
    sekarang melakukan "combining" antar agent.

    Output yang diharapkan:
      - Construct yang melanjutkan corrected_kv harus menghasilkan factor
        yang aligned dengan hypothesis di propose.
      - Probe corrected_kv: model bisa restate scenario + factor + metric.
    """
    if latent_steps is None:
        latent_steps = int(os.environ.get("TEST_LATENT_STEPS", "10"))

    group, case = "pair_propose_construct", f"kv_combine_corrected_m{latent_steps}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}  (CORRECTED — seed-and-extend, RoPE positions benar)")
    print("═" * 78)

    if CONFIG.dry_run:
        return {"group": group, "case": case, "ok_format": True,
                "response": "(dry_run)", "parsed": None, "elapsed_s": 0.0, "log_path": str(log_path)}

    backend = get_latent_backend()

    # Step 1: propose_kv (sama seperti raw case)
    print("── Step 1: Propose kv_only → propose_kv ──")
    sys_p, usr_p = build_propose_msgs("new")
    t0 = time.time()
    r_prop = backend.build_messages_and_run(
        user_prompt=usr_p, system_prompt=sys_p,
        mode="kv_only", latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role="kvcomb_corr_propose",
    )
    propose_kv = r_prop.kv_cache
    L_a = kv_length(propose_kv)
    elapsed_a = round(time.time() - t0, 2)
    print(f"  propose_kv len={L_a}  elapsed={elapsed_a}s")

    # Step 2: Process feedback prompt AT TOP OF propose_kv → corrected combined
    print("── Step 2: Feedback kv_only DENGAN SEED propose_kv → corrected_kv ──")
    sys_f, usr_f = build_feedback_msgs(
        "new",
        hypothesis_text=fx.HYPOTHESIS_DICT["hypothesis"],
        factor_summary="- F1: `RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))` [implemented=True]",
        combined_result=fx.COMBINED_RESULT_STR,
    )
    t1 = time.time()
    r_fb = backend.build_messages_and_run(
        user_prompt=usr_f, system_prompt=sys_f,
        mode="kv_only", past_key_values=propose_kv,
        latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role="kvcomb_corr_feedback",
    )
    corrected_kv = r_fb.kv_cache
    L_c = kv_length(corrected_kv)
    elapsed_b = round(time.time() - t1, 2)
    print(f"  corrected_kv len={L_c}  (>= L_a={L_a} expected)  elapsed={elapsed_b}s")

    # Step 3: Probe corrected_kv
    print("── Step 3: Probing corrected_kv ──")
    probe_modes = enabled_modes_from_env() or ["rewrite", "format_check"]
    probes = run_probes_at(backend, corrected_kv, kv_label="combined_corrected_kv", modes=probe_modes)
    if probes:
        print_probe_summary(probes)

    # Step 4: Construct dengan corrected_kv
    print("── Step 4: Construct dengan corrected_kv ──")
    sys_c, usr_c = build_construct_msgs("new")
    t2 = time.time()
    r_ctor = backend.build_messages_and_run(
        user_prompt=usr_c, system_prompt=sys_c,
        json_mode=False, mode="kv_and_text",
        past_key_values=corrected_kv, latent_steps=latent_steps,
        temperature=0.7, top_p=0.95, role="kvcomb_corr_construct",
    )
    construct_text = r_ctor.text or ""
    construct_json = _extract_json(construct_text)
    elapsed_c = round(time.time() - t2, 2)
    print(f"  construct: text_len={len(construct_text)}  schema_ok={_has_construct_schema(construct_text)}  "
          f"elapsed={elapsed_c}s")
    print(f"  preview: {construct_text[:200]!r}")

    sections = [
        (f"CORRECTED SEED-EXTEND  L_a={L_a}  L_corrected={L_c}  Δ={L_c - L_a}  "
         f"(feedback prompt processed on top of propose KV → positions benar)", ""),
        ("CONSTRUCT OUTPUT (under corrected KV)", construct_text),
        ("PARSED JSON (atau parse attempt dari plain text)", _json.dumps(construct_json, indent=2, ensure_ascii=False)
         if construct_json else "(plain text format — tidak ada JSON)"),
    ]
    sections.extend(format_probes_for_log(probes))
    _log_save(log_path, sections)
    print(f"\n── LOG SAVED ── {log_path}")

    return {
        "group": group, "case": case, "latent_steps": latent_steps,
        "ok_format": _has_construct_schema(construct_text),
        "elapsed_s": round(elapsed_a + elapsed_b + elapsed_c, 2),
        "log_path": str(log_path),
        "kv_lens": {"propose": L_a, "corrected": L_c, "growth": L_c - L_a},
        "construct_text_len": len(construct_text),
        "parsed": construct_json,
    }


# ─── Registry ────────────────────────────────────────────────────────────────

CASES = {
    "fresh_start":           test_fresh_start,
    "mutation_seeded":       test_mutation_seeded,
    "feedback_chained":      test_feedback_chained,
    "prompt_ab_chain":       test_prompt_ab_chain,
    "kv_combine_raw":        test_kv_combine_raw,
    "kv_combine_corrected":  test_kv_combine_corrected,
}
