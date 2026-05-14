"""KV-cache introspection probes for pair tests.

After an agent A → B chain runs, this module lets you ask "what did B
actually see/know via the KV it received from A?" by issuing a third,
diagnostic LLM call that uses the SAME KV-cache as past_key_values but
with a minimal, generic probe prompt.

Why this works (and its limits):
  KV-cache is not directly decodable to natural language — the W_K/W_V
  projections are many-to-one. But the same model that produced the KV
  can re-attend to it and emit text that approximates the latent content.
  With a deliberately minimal probe prompt (low new tokens, low
  temperature), the readout is biased mostly by the KV itself rather
  than by fresh instructions in the probe. See latent_kvcache_mechanics
  memory: this is the only feasible KV introspection approach.

Probe is RESEARCH/DIAGNOSTIC ONLY:
  - It adds 1+ LLM calls per pair-test scenario.
  - The KV passed in is `.crop()`-restored to its original length after
    the probe to avoid contaminating downstream state.
  - latent_steps=0 by default: we want the model to decode from existing
    KV, not "think more" on top of it (which would bias the readout
    toward the probe prompt rather than the KV content).

Opt-in via env var:
    TEST_PROBE=rewrite             → run rewrite probe at every checkpoint
    TEST_PROBE=rewrite,format_check → run two probes per checkpoint
    TEST_PROBE=all                 → all 4 modes
    (unset / empty)                → no probes (default)

Each probe call also produces its own `.md` snapshot under
`backend/debug/llm_outputs/session_*/` via the regular `_save_output_snapshot`
path — the `role` is set to `probe_<mode>_<kv_label>` for easy filtering.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional


# ─── Probe mode definitions ──────────────────────────────────────────────────
# (system_prompt, user_prompt). Empty string → use whatever default the
# backend chooses. Each prompt is intentionally short so the bulk of the
# response is driven by the KV content, not by re-instruction.

PROBE_MODES: dict[str, tuple[str, str]] = {
    # Literal re-statement: ask the model to surface its current understanding
    # of task + scenario + constraints. Most useful for "what does this agent
    # think it was just told to do?"
    "rewrite": (
        "You are a literal, honest assistant. Reply in plain English. "
        "Do not invent details that were not provided.",
        "Without adding new ideas, restate clearly: (1) what task you were "
        "just given, (2) the scenario you are operating in, (3) the output "
        "format you are expected to produce, and (4) any constraints. "
        "If your context is incomplete, say so.",
    ),
    # Compact summarization: 5 bullets. Quickly comparable across runs.
    "summarize": (
        "You are a concise context summarizer.",
        "Summarize the context you currently hold in exactly 5 bullets: "
        "(a) scenario, (b) task, (c) constraints, (d) recent history, "
        "(e) expected output format. One short line each.",
    ),
    # Free continuation: empty user prompt → model decodes purely from KV.
    # The "drift direction" reveals what the KV biases toward — most
    # sensitive to format pollution (e.g., mutation seed leaking JSON keys).
    "continue": (
        "Continue from your current context. Be brief.",
        "Continue.",
    ),
    # Schema introspection: which JSON keys does the model "remember"?
    # Direct test for mutation pollution (new_hypothesis, <factor_name>, etc.).
    "format_check": (
        "You are a structured introspector. Reply with ONLY a list, no prose.",
        "List every JSON key you remember seeing in the context provided to "
        "you so far. One key per line. No commentary, no examples.",
    ),
}


# ─── Result container ───────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Captured output of one probe call."""

    probe_mode: str
    kv_label: str
    kv_len_before: int
    kv_len_after: int
    probe_system_prompt: str
    probe_user_prompt: str
    response: str
    text_len: int
    elapsed_s: float


# ─── Internals ──────────────────────────────────────────────────────────────

def _past_len(kv) -> int:
    """KV length helper — gracefully degrades if backend imports fail."""
    if kv is None:
        return 0
    try:
        from backend.llm.models import _past_length
        return int(_past_length(kv))
    except Exception:
        try:
            return int(kv.get_seq_length())
        except Exception:
            return -1


def _restore_kv(kv, target_len: int) -> None:
    """Crop KV back to `target_len` so probe does not pollute downstream state."""
    if kv is None or target_len <= 0:
        return
    if not hasattr(kv, "crop"):
        return
    try:
        kv.crop(target_len)
    except Exception:
        # Crop failures are non-fatal for the probe itself; downstream
        # code may still detect length mismatch and react.
        pass


# ─── Public API ─────────────────────────────────────────────────────────────

def enabled_modes_from_env() -> list[str]:
    """Parse TEST_PROBE env var into a list of valid probe-mode names.

    Empty / unset returns ``[]`` — caller is responsible for short-circuiting
    so the probe block is fully skipped (zero LLM calls) when disabled.
    """
    val = os.environ.get("TEST_PROBE", "").strip()
    if not val:
        return []
    if val.lower() == "all":
        return list(PROBE_MODES)
    modes: list[str] = []
    for raw in val.split(","):
        m = raw.strip()
        if m in PROBE_MODES and m not in modes:
            modes.append(m)
    return modes


def run_probe(
    backend,
    kv,
    *,
    probe_mode: str = "rewrite",
    kv_label: str = "kv",
    latent_steps: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.95,
    role: Optional[str] = None,
) -> ProbeResult:
    """Issue one probe call and return its result.

    The supplied ``kv`` is snapshotted via its length and cropped back to
    that length after the call, so the probe does not extend the KV that
    later test steps may continue to use.

    Args:
        backend       : LocalLLMBackend instance (latent-enabled is fine).
        kv            : DynamicCache or None. ``None`` falls back to a
                        text-only probe — the response then reflects how
                        the model behaves on the probe prompt alone, which
                        is useful as a baseline.
        probe_mode    : key in :data:`PROBE_MODES`.
        kv_label      : descriptive label for snapshot filename & log.
        latent_steps  : 0 = decode straight from KV+prompt without extra
                        latent forward passes. Raise only if you explicitly
                        want to study how the probe interacts with new
                        latent reasoning.
        max_new_tokens: cap on probe response length.
        temperature   : low (0.3) to favour fidelity over creativity.
        top_p         : standard.
        role          : override for snapshot filename role tag.

    Returns:
        ProbeResult containing the probe's input + output + KV deltas.
    """
    if probe_mode not in PROBE_MODES:
        raise ValueError(
            f"Unknown probe_mode={probe_mode!r}. Valid: {list(PROBE_MODES)}"
        )

    sys_p, usr_p = PROBE_MODES[probe_mode]
    role = role or f"probe_{probe_mode}_{kv_label}"

    kv_len_before = _past_len(kv)
    t0 = time.time()
    result = backend.build_messages_and_run(
        user_prompt=usr_p,
        system_prompt=sys_p if sys_p else None,
        mode="kv_and_text" if kv is not None else "text_only",
        past_key_values=kv,
        latent_steps=latent_steps,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        role=role,
        json_mode=False,
    )
    elapsed = round(time.time() - t0, 2)
    response = result.text or ""

    _restore_kv(kv, kv_len_before)
    kv_len_after = _past_len(kv)

    return ProbeResult(
        probe_mode=probe_mode,
        kv_label=kv_label,
        kv_len_before=kv_len_before,
        kv_len_after=kv_len_after,
        probe_system_prompt=sys_p,
        probe_user_prompt=usr_p,
        response=response,
        text_len=len(response),
        elapsed_s=elapsed,
    )


def run_probes_at(
    backend,
    kv,
    *,
    kv_label: str,
    modes: Optional[list[str]] = None,
    **probe_kwargs: Any,
) -> list[ProbeResult]:
    """Run every enabled probe mode at one checkpoint.

    If ``modes`` is None, reads from :func:`enabled_modes_from_env`.
    Returns ``[]`` when no modes are enabled — caller can safely skip
    log appending without conditionals.
    """
    modes = modes if modes is not None else enabled_modes_from_env()
    results: list[ProbeResult] = []
    for m in modes:
        try:
            results.append(
                run_probe(backend, kv, probe_mode=m, kv_label=kv_label, **probe_kwargs)
            )
        except Exception as exc:
            # Probes are diagnostic — never block the test on a probe error.
            print(f"  [probe] {m}@{kv_label} skipped: {type(exc).__name__}: {exc}")
    return results


def format_probes_for_log(probes: list[ProbeResult]) -> list[tuple[str, str]]:
    """Convert probe results into ``(title, body)`` sections for log files.

    Designed to be appended to the ``extra_sections`` argument of the
    existing pair-test ``_save_scenario_log`` helper.
    """
    sections: list[tuple[str, str]] = []
    for p in probes:
        title = (
            f"PROBE [{p.probe_mode}] @ {p.kv_label}  "
            f"kv_len={p.kv_len_before}→{p.kv_len_after}  "
            f"text_len={p.text_len}  elapsed={p.elapsed_s}s"
        )
        body = (
            "--- probe system prompt ---\n"
            f"{p.probe_system_prompt or '(default)'}\n"
            "--- probe user prompt ---\n"
            f"{p.probe_user_prompt or '(empty — continue from KV)'}\n"
            "--- probe response ---\n"
            f"{p.response or '(empty)'}"
        )
        sections.append((title, body))
    return sections


def print_probe_summary(probes: list[ProbeResult], indent: str = "    ") -> None:
    """Brief console summary of all probes at one checkpoint."""
    for p in probes:
        preview = (p.response or "").strip().replace("\n", " ⏎ ")[:140]
        print(
            f"{indent}[probe:{p.probe_mode}@{p.kv_label}] "
            f"len={p.text_len} elapsed={p.elapsed_s}s  preview: {preview!r}"
        )
