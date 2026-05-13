"""
Rekayasa experiments: validate mutation & crossover mechanisms vs QuantaAlpha paper.

MUTATION REKAYASA
  Goal: verify LLM correctly localizes the parent's weakness and generates a
  direction that is ORTHOGONAL (different market hypothesis + different features).
  3 parent types with distinct failure modes:
    - weak_momentum:    IC=0.005, no normalization, no volume filter
    - overfit_meanrev:  IC=0.041 but MDD=31.2%, collapses in trending regimes
    - high_ic_bad_risk: IC=0.058 but MDD=24.1%, vol-scaling amplifies illiquid risk
  Each parent is tested N_TRIALS=3 times. Per trial:
    - novelty_score:      1 - jaccard(parent_hyp, child_hyp) -- higher = more different
    - weakness_ack_score: fraction of parent_feedback keywords in orthogonality_reason

CROSSOVER REKAYASA
  Goal: verify LLM genuinely fuses two parents (not just averaging).
  3 groups:
    - momentum_x_meanrev:        complementary regimes (trending vs ranging)
    - voladj_x_volconfirm:       complementary risk profiles (IC vs MDD)
    - anti_complement_momentum:  near-identical parents -- expect LLM to notice overlap
  Per trial: fusion_coverage = fraction of each parent's hypothesis tokens in fusion_logic.
  Anti-complementary group is a negative control: valid JSON but fusion_logic should
  ideally acknowledge the low orthogonality rather than inventing false novelty.

RECONSTRUCTION AGENT DESIGN (for #3, dedicated 4th agent)
  Problem: coder retry inherits KV from construct -> polluted by construct's broken
  expression -> retry produces similar faulty expressions (mirror collapse observed
  in debug session 075518, attempts 2-5, text_len=7 or 0).

  Solution: a dedicated "Reconstruction Agent" called after N consecutive mirror failures:
    - Input : factor_description + failure_summary (list of failed expressions) + allowed_functions
    - Mode  : text_only (NO KV inheritance from construct or coder)
    - System: full evolving_strategy_factor_implementation_v1_system (with function library)
      since there is no prior KV to carry that context
    - User  : just "Factor: X, Description: Y, Previous attempts failed: [e1, e2, ...]
               Write a NEW expression that avoids these exact patterns."
    - Trigger: after 3 consecutive retries where normalized(expr) == normalized(prev_expr)
      (mirror detection already exists in test_pair_construct_coder.py)
    - Implementation: call get_backend() (not get_latent_backend()) for the reconstruction
      step — this is the critical difference from the current retry path.

  This is NOT the same as mutation. Mutation works at hypothesis level (new market idea).
  Reconstruction works at expression level (same hypothesis, fresh expression). They are
  complementary: mutation diversifies strategy space, reconstruction rescues a stuck coder.

Pemakaian:
    python -m try.run --group evolution_rekayasa --case mutation_rekayasa_weak_momentum
    python -m try.run --group evolution_rekayasa --case crossover_rekayasa_momentum_x_meanrev
    python -m try.run --group evolution_rekayasa                    # semua (6 cases)
    python -m try.run --group evolution_rekayasa --dry-run
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from .common import load_yaml, render_format, get_backend, _extract_json, _save_log, PROMPT_PATHS
from .config import CONFIG
from . import fixtures as fx


# ─── Lazy YAML ───────────────────────────────────────────────────────────────

_EVOLUTION_YAML: dict | None = None


def _evo() -> dict:
    global _EVOLUTION_YAML
    if _EVOLUTION_YAML is None:
        _EVOLUTION_YAML = load_yaml(PROMPT_PATHS["evolution"])
    return _EVOLUTION_YAML


# ─── Scoring helpers ──────────────────────────────────────────────────────────

_STOP = {"a", "an", "the", "of", "in", "to", "and", "or", "is", "are",
         "that", "this", "for", "by", "with", "as", "on", "at", "be",
         "from", "its", "it", "not", "but", "are", "was", "has", "can"}


def _tok(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOP and len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _novelty(parent_hyp: str, child_hyp: str) -> float:
    """1 - jaccard(parent tokens, child tokens). Higher = child is more novel."""
    return round(1.0 - _jaccard(_tok(parent_hyp), _tok(child_hyp)), 3)


def _weakness_ack(parent_feedback: str, orthogonality_reason: str) -> float:
    """Fraction of parent feedback keywords present in orthogonality_reason."""
    fb = _tok(parent_feedback)
    ort = _tok(orthogonality_reason)
    return round(len(fb & ort) / max(len(fb), 1), 3)


def _fusion_cov(p1_hyp: str, p2_hyp: str, fusion_logic: str) -> dict:
    """Fraction of each parent's hypothesis tokens appearing in fusion_logic."""
    p1 = _tok(p1_hyp)
    p2 = _tok(p2_hyp)
    fl = _tok(fusion_logic)
    return {
        "p1": round(len(p1 & fl) / max(len(p1), 1), 3),
        "p2": round(len(p2 & fl) / max(len(p2), 1), 3),
    }


def _extract_parent_hyps(summaries_str: str) -> tuple[str, str]:
    """Extract first two **Hypothesis** values from formatted parent_summaries_str."""
    hyps = re.findall(r"\*\*Hypothesis\*\*:\s*([^\n]+)", summaries_str)
    p1 = hyps[0] if len(hyps) > 0 else ""
    p2 = hyps[1] if len(hyps) > 1 else ""
    return p1, p2


# ─── Single trial: mutation ───────────────────────────────────────────────────

def _mutation_trial(parent: dict, trial: int) -> dict:
    y = _evo()["mutation"]
    sys_p = y["system"]
    usr_p = render_format(
        y["user"],
        parent_hypothesis=parent["parent_hypothesis"],
        parent_factors=parent["parent_factors_str"],
        parent_metrics=parent["parent_metrics_str"],
        parent_feedback=parent["parent_feedback_str"],
    )
    label = parent["label"]
    group = "evolution_rekayasa"
    case = f"mutation_rekayasa_{label}_t{trial}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    expected = ["new_hypothesis", "exploration_direction", "orthogonality_reason", "expected_characteristics"]
    result: dict = {
        "group": group, "case": case,
        "json_mode": True, "ok_format": None,
        "response": None, "parsed": None,
        "elapsed_s": 0.0, "log_path": str(log_path),
        "novelty_score": 0.0, "weakness_ack_score": 0.0,
    }

    print(f"\n  ── mutation {label} trial={trial} ──")

    if CONFIG.dry_run:
        result["ok_format"] = True
        _save_log(log_path, sys_p, usr_p, "(dry_run)", result)
        return result

    backend = get_backend(CONFIG)
    t0 = time.time()
    try:
        response = backend.build_messages_and_create_chat_completion(
            user_prompt=usr_p, system_prompt=sys_p, json_mode=True,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        _save_log(log_path, sys_p, usr_p, f"ERROR: {result['error']}", result)
        return result

    result["elapsed_s"] = round(time.time() - t0, 2)
    result["response"] = response
    parsed = _extract_json(response or "")
    result["parsed"] = parsed

    if parsed is None:
        result["ok_format"] = False
        print(f"     ❌ JSON parse failed  elapsed={result['elapsed_s']}s")
    else:
        missing = [k for k in expected if k not in parsed]
        result["ok_format"] = len(missing) == 0
        if missing:
            print(f"     ❌ missing keys: {missing}  elapsed={result['elapsed_s']}s")
        else:
            new_hyp = parsed.get("new_hypothesis", "")
            orth_r = parsed.get("orthogonality_reason", "")
            result["novelty_score"] = _novelty(parent["parent_hypothesis"], new_hyp)
            result["weakness_ack_score"] = _weakness_ack(parent["parent_feedback_str"], orth_r)
            print(
                f"     ✔ ok  elapsed={result['elapsed_s']}s  "
                f"novelty={result['novelty_score']}  weakness_ack={result['weakness_ack_score']}"
            )
            print(f"     new_hyp: {new_hyp[:100]!r}")

    _save_log(log_path, sys_p, usr_p, response or "", result)
    return result


# ─── Single trial: crossover ──────────────────────────────────────────────────

def _crossover_trial(grp: dict, trial: int) -> dict:
    y = _evo()["crossover"]
    sys_p = y["system"]
    usr_p = render_format(y["user"], parent_summaries=grp["parent_summaries_str"])

    label = grp["label"]
    group = "evolution_rekayasa"
    case = f"crossover_rekayasa_{label}_t{trial}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = CONFIG.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    expected = ["hybrid_hypothesis", "fusion_logic", "innovation_points", "expected_benefits"]
    result: dict = {
        "group": group, "case": case,
        "json_mode": True, "ok_format": None,
        "response": None, "parsed": None,
        "elapsed_s": 0.0, "log_path": str(log_path),
        "fusion_coverage": None,
    }

    print(f"\n  ── crossover {label} trial={trial} [{grp['description']}] ──")

    if CONFIG.dry_run:
        result["ok_format"] = True
        _save_log(log_path, sys_p, usr_p, "(dry_run)", result)
        return result

    backend = get_backend(CONFIG)
    t0 = time.time()
    try:
        response = backend.build_messages_and_create_chat_completion(
            user_prompt=usr_p, system_prompt=sys_p, json_mode=True,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        _save_log(log_path, sys_p, usr_p, f"ERROR: {result['error']}", result)
        return result

    result["elapsed_s"] = round(time.time() - t0, 2)
    result["response"] = response
    parsed = _extract_json(response or "")
    result["parsed"] = parsed

    if parsed is None:
        result["ok_format"] = False
        print(f"     ❌ JSON parse failed  elapsed={result['elapsed_s']}s")
    else:
        missing = [k for k in expected if k not in parsed]
        result["ok_format"] = len(missing) == 0
        if missing:
            print(f"     ❌ missing keys: {missing}  elapsed={result['elapsed_s']}s")
        else:
            p1_hyp, p2_hyp = _extract_parent_hyps(grp["parent_summaries_str"])
            fusion_logic = parsed.get("fusion_logic", "")
            result["fusion_coverage"] = _fusion_cov(p1_hyp, p2_hyp, fusion_logic)
            fc = result["fusion_coverage"]
            print(
                f"     ✔ ok  elapsed={result['elapsed_s']}s  "
                f"fusion_p1={fc['p1']}  fusion_p2={fc['p2']}"
            )
            print(f"     hybrid_hyp: {parsed.get('hybrid_hypothesis', '')[:100]!r}")

    _save_log(log_path, sys_p, usr_p, response or "", result)
    return result


# ─── Multi-trial wrappers (returned to run.py as one summary dict) ────────────

N_TRIALS = 3


def _run_mutation(label: str) -> dict:
    parent = next(p for p in fx.MUTATION_PARENTS if p["label"] == label)
    print(f"\n{'═' * 70}")
    print(f"▶ [evolution_rekayasa] mutation_rekayasa_{label}  ({N_TRIALS} trials)")
    print(f"{'═' * 70}")
    trials = [_mutation_trial(parent, t) for t in range(1, N_TRIALS + 1)]
    n_ok = sum(1 for r in trials if r.get("ok_format"))
    novelty = [r.get("novelty_score", 0.0) for r in trials]
    weakness = [r.get("weakness_ack_score", 0.0) for r in trials]
    elapsed = sum(r.get("elapsed_s", 0.0) for r in trials)
    print(f"\n── summary  {n_ok}/{N_TRIALS} ok  "
          f"novelty={novelty}  weakness_ack={weakness}  total={elapsed:.2f}s")
    return {
        "group": "evolution_rekayasa",
        "case": f"mutation_rekayasa_{label}",
        "ok_format": n_ok == N_TRIALS,
        "elapsed_s": elapsed,
        "n_trials": N_TRIALS, "n_ok": n_ok,
        "novelty_scores": novelty,
        "weakness_ack_scores": weakness,
    }


def _run_crossover(label: str) -> dict:
    grp = next(g for g in fx.CROSSOVER_GROUPS if g["label"] == label)
    print(f"\n{'═' * 70}")
    print(f"▶ [evolution_rekayasa] crossover_rekayasa_{label}  ({N_TRIALS} trials)")
    print(f"{'═' * 70}")
    trials = [_crossover_trial(grp, t) for t in range(1, N_TRIALS + 1)]
    n_ok = sum(1 for r in trials if r.get("ok_format"))
    fcs = [r["fusion_coverage"] for r in trials if r.get("fusion_coverage")]
    elapsed = sum(r.get("elapsed_s", 0.0) for r in trials)
    print(f"\n── summary  {n_ok}/{N_TRIALS} ok  fusion_coverages={fcs}  total={elapsed:.2f}s")
    return {
        "group": "evolution_rekayasa",
        "case": f"crossover_rekayasa_{label}",
        "ok_format": n_ok == N_TRIALS,
        "elapsed_s": elapsed,
        "n_trials": N_TRIALS, "n_ok": n_ok,
        "fusion_coverages": fcs,
    }


# ─── CASES (used by run.py) ───────────────────────────────────────────────────

CASES = {
    "mutation_rekayasa_weak_momentum":        lambda: _run_mutation("weak_momentum"),
    "mutation_rekayasa_overfit_meanrev":      lambda: _run_mutation("overfit_meanrev"),
    "mutation_rekayasa_high_ic_bad_risk":     lambda: _run_mutation("high_ic_bad_risk"),
    "crossover_rekayasa_momentum_x_meanrev":  lambda: _run_crossover("momentum_x_meanrev"),
    "crossover_rekayasa_voladj_x_volconfirm": lambda: _run_crossover("voladj_x_volconfirm"),
    "crossover_rekayasa_anti_complement":     lambda: _run_crossover("anti_complement_momentum"),
}
