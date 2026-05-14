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


def _has_construct_schema(parsed: dict | None) -> bool:
    """Cek apakah output construct memiliki nested schema 4-key."""
    if not parsed or not isinstance(parsed, dict):
        return False
    for v in parsed.values():
        if not isinstance(v, dict):
            return False
        if not all(k in v for k in ("description", "variables", "formulation", "expression")):
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
    scen_desc = fx.SCENARIO.get_scenario_all_desc()
    sys_f = _jinja(y["factor_feedback_generation"]["system"], scenario=scen_desc)
    usr_f = _jinja(
        y["factor_feedback_generation"]["user"],
        hypothesis_text=fx.HYPOTHESIS_DICT["hypothesis"],
        task_details=fx.TASK_DETAILS,
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
        json_mode=True, mode="kv_and_text",
        past_key_values=propose_kv, latent_steps=latent_steps,
        temperature=temp, top_p=top_p, role=f"construct_{scenario_label}",
    )
    construct_text = r_ctor.text or ""
    construct_json = _extract_json(construct_text)
    construct_schema_ok = _has_construct_schema(construct_json)
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
    construct_sys, construct_usr = _build_construct_msgs(target_hypothesis="")

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
    construct_sys, construct_usr = _build_construct_msgs(target_hypothesis="")

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
    construct_sys, construct_usr = _build_construct_msgs(target_hypothesis="")

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


# ─── Registry ────────────────────────────────────────────────────────────────

CASES = {
    "fresh_start":      test_fresh_start,
    "mutation_seeded":  test_mutation_seeded,
    "feedback_chained": test_feedback_chained,
}
