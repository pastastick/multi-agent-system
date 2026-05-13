"""
Helper bersama untuk semua test:
  - load YAML prompt (sama dengan core.prompts.Prompts, tanpa SingletonBaseClass)
  - render Jinja2 (dipakai propose/construct/coder/evaluator/feedback)
  - render .format() (dipakai planning/mutation/crossover)
  - panggil LLM via LocalLLMBackend (singleton lazy-init)
  - simpan+print hasil ke console + file
  - validasi output JSON (best-effort)

Mengapa singleton backend: load Qwen3-4B/14B makan ~8-15 GB VRAM.
Kita hanya mau load sekali untuk seluruh batch test.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Environment, StrictUndefined

# Tambahkan backend/ ke sys.path agar `from llm.client import ...` bekerja
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from .config import CONFIG, TestConfig


# ═════════════════════════════════════════════════════════════════════════
# YAML + template rendering
# ═════════════════════════════════════════════════════════════════════════

def load_yaml(path: Path) -> dict[str, Any]:
    """Mirror core.prompts.Prompts: load YAML ke dict biasa."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Failed to load prompts from {path}")
    return data


def render_jinja(template_str: str, **vars: Any) -> str:
    """Render Jinja2 template (undefined=StrictUndefined agar missing var error)."""
    return (
        Environment(undefined=StrictUndefined)
        .from_string(template_str)
        .render(**vars)
    )


def render_format(template_str: str, **vars: Any) -> str:
    """Render pakai .format() — dipakai planning/mutation/crossover."""
    return template_str.format(**vars)


# Path ke file YAML yang dipakai di backend
BACKEND_ROOT = Path(__file__).resolve().parent.parent / "backend"

PROMPT_PATHS = {
    "factors_prompts": BACKEND_ROOT / "factors" / "prompts" / "prompts.yaml",
    "factors_coder_qa": BACKEND_ROOT / "factors" / "coder" / "qa_prompts.yaml",
    "factors_coder_base": BACKEND_ROOT / "factors" / "coder" / "prompts.yaml",
    "planning": BACKEND_ROOT / "pipeline" / "prompts" / "planning_prompts.yaml",
    "evolution": BACKEND_ROOT / "pipeline" / "prompts" / "evolution_prompts.yaml",
}


# ═════════════════════════════════════════════════════════════════════════
# LLM backend (lazy-init singleton)
# ═════════════════════════════════════════════════════════════════════════

_BACKEND_INSTANCE = None
_LATENT_BACKEND_INSTANCE = None


def get_backend(config: TestConfig = CONFIG):
    """Lazy-init LocalLLMBackend. Load model hanya sekali per proses.

    Text-only backend (latent_steps=0). Dipakai oleh run_case untuk
    semua test yang hanya butuh text generation biasa.
    """
    global _BACKEND_INSTANCE
    if _BACKEND_INSTANCE is None:
        from backend.llm.client import LocalLLMBackend
        print(f"[common] Loading LocalLLMBackend model={config.model_name} device={config.device} ...")
        _BACKEND_INSTANCE = LocalLLMBackend(
            model_name=config.model_name,
            device=config.device,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            log_tensors=False,
            store_kv=False,
        )
        print("[common] Backend ready.")
    return _BACKEND_INSTANCE


def get_latent_backend(config: TestConfig = CONFIG, latent_steps_init: int = 10):
    """Lazy-init LocalLLMBackend dengan latent support aktif.

    Kenapa terpisah dari get_backend(): latent butuh `use_realign=True` dan
    `latent_steps>0` saat __init__ supaya LatentRealigner (W_a projection
    matrix) dibangun. Model weights di-share via _MODEL_CACHE, jadi hanya
    overhead realigner build (~few seconds, negligible VRAM).

    latent_steps_init hanya menentukan default engine — bisa di-override
    per-call via run(..., latent_steps=N).
    """
    global _LATENT_BACKEND_INSTANCE
    if _LATENT_BACKEND_INSTANCE is None:
        from backend.llm.client import LocalLLMBackend
        print(f"[common] Loading LATENT LocalLLMBackend (use_realign=True) ...")
        _LATENT_BACKEND_INSTANCE = LocalLLMBackend(
            model_name=config.model_name,
            device=config.device,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            latent_steps=latent_steps_init,
            use_realign=True,
            log_tensors=False,
            store_kv=False,
        )
        print("[common] Latent backend ready.")
    return _LATENT_BACKEND_INSTANCE


# ═════════════════════════════════════════════════════════════════════════
# Run + logging
# ═════════════════════════════════════════════════════════════════════════

def kv_shape_report(kv, label: str = "") -> dict:
    """
    Return a lightweight dict describing KV-cache state at any agent boundary.

    KV INSPECTION FEASIBILITY NOTE
    ─────────────────────────────
    What this CAN tell you:
      - n_tokens  : total sequence length accumulated (= tokens seen so far)
      - n_layers  : transformer depth (confirms model loaded correctly)
      - size_mb   : GPU/CPU memory consumed by this KV
    What this CANNOT tell you:
      - Which tokens encode which piece of information (construct hypothesis, scenario, etc.)
        KV stores rotated+projected key/value tensors — NOT the original token embeddings.
        Recovering "which token" requires knowing the exact tokenization at recording time.

    Practical corruption tracing across ALL agents:
      Call kv_shape_report(r.kv_cache, label="propose") after each agent step.
      Plot n_tokens: a sudden spike means an agent added much more context than expected
      (e.g., construct's latent steps + prompt grew the KV by >2000 tokens unexpectedly).
      Then run chain_ablation: remove KV from that step and measure output degradation.
      The step whose removal causes the biggest quality drop owns the critical context.

    Usage:
        r = backend.build_messages_and_run(...)
        info = kv_shape_report(r.kv_cache, label="construct")
        print(info)  # {"label": "construct", "n_tokens": 2048, "n_layers": 36, "size_mb": 295.2}
    """
    if kv is None:
        return {"label": label, "n_tokens": 0, "n_layers": 0, "size_mb": 0.0}
    try:
        from backend.llm.models import _past_length
        n_tokens = _past_length(kv)
    except Exception:
        n_tokens = -1
    try:
        from backend.llm._shared import kv_size_bytes
        size_mb = round(kv_size_bytes(kv) / 1024 / 1024, 1)
    except Exception:
        size_mb = -1.0
    try:
        n_layers = len(kv) if hasattr(kv, "__len__") else -1
    except Exception:
        n_layers = -1
    info = {"label": label, "n_tokens": n_tokens, "n_layers": n_layers, "size_mb": size_mb}
    if label:
        print(f"[kv_shape] {label}: {n_tokens} tokens  {n_layers} layers  {size_mb} MB")
    return info


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n... [truncated, total={len(s)} chars]"


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction dari response LLM."""
    if not text:
        return None
    t = text.strip()
    # Strip ```json fences kalau ada
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    # Cari bracket pertama
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None


def run_case(
    group: str,
    case: str,
    system_prompt: str,
    user_prompt: str,
    *,
    json_mode: bool = False,
    expected_keys: Optional[list[str]] = None,
    config: TestConfig = CONFIG,
) -> dict:
    """
    Jalankan satu test case.

    Args:
        group:          nama group (external / planning_evolution / ...)
        case:           nama case spesifik (propose / construct / feedback / ...)
        system_prompt:  prompt sistem yang sudah dirender
        user_prompt:    prompt user yang sudah dirender
        json_mode:      minta model output JSON ketat
        expected_keys:  list key JSON yang wajib muncul — untuk validasi format

    Return: dict ringkasan {case, ok_format, response, parsed, elapsed_s}.
    File log disimpan ke {output_dir}/{group}/{case}_{ts}.txt
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = config.output_dir / group
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}_{ts}.txt"

    effective_json_mode = json_mode or config.force_json_mode

    # Header ke konsol
    print("\n" + "═" * 78)
    print(f"▶ [{group}] {case}")
    print(f"  json_mode={effective_json_mode}  dry_run={config.dry_run}  log={log_path}")
    print("═" * 78)
    print("── SYSTEM ──")
    print(_truncate(system_prompt, config.console_preview_chars))
    print("── USER ──")
    print(_truncate(user_prompt, config.console_preview_chars))

    result = {
        "group": group,
        "case": case,
        "json_mode": effective_json_mode,
        "ok_format": None,
        "response": None,
        "parsed": None,
        "elapsed_s": 0.0,
        "log_path": str(log_path),
    }

    # Dry-run: hanya print prompt, tidak call LLM
    if config.dry_run:
        print("── RESPONSE ──\n(skipped: dry_run=True)")
        _save_log(log_path, system_prompt, user_prompt, response="(dry_run)", result=result)
        return result

    # Panggil LLM
    try:
        backend = get_backend(config)
        t0 = time.time()
        response = backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=effective_json_mode,
        )
        result["elapsed_s"] = round(time.time() - t0, 2)
        result["response"] = response
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"── ERROR ── {result['error']}")
        _save_log(log_path, system_prompt, user_prompt, response=f"ERROR: {result['error']}", result=result)
        return result

    print(f"── RESPONSE ({result['elapsed_s']}s) ──")
    print(_truncate(response or "", config.console_preview_chars))

    # Validasi JSON (jika diminta)
    if effective_json_mode:
        parsed = _extract_json(response or "")
        result["parsed"] = parsed
        if parsed is None:
            result["ok_format"] = False
            print("── VALIDATION ── ❌ JSON parse failed")
        elif expected_keys:
            missing = [k for k in expected_keys if k not in parsed]
            if missing:
                result["ok_format"] = False
                print(f"── VALIDATION ── ❌ missing keys: {missing}")
            else:
                result["ok_format"] = True
                print(f"── VALIDATION ── ✔ all expected keys present: {expected_keys}")
        else:
            result["ok_format"] = True
            print("── VALIDATION ── ✔ JSON parseable")
    else:
        result["ok_format"] = bool(response and response.strip())
        print(f"── VALIDATION ── {'✔' if result['ok_format'] else '❌'} non-empty text")

    _save_log(log_path, system_prompt, user_prompt, response=response or "", result=result)
    return result


def _save_log(
    log_path: Path,
    system_prompt: str,
    user_prompt: str,
    response: str,
    result: dict,
) -> None:
    """Tulis full prompt + response + metadata ke file."""
    lines = [
        f"# Test case: {result.get('group')}/{result.get('case')}",
        f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# json_mode: {result.get('json_mode')}",
        f"# elapsed_s: {result.get('elapsed_s')}",
        f"# ok_format: {result.get('ok_format')}",
        "",
        "=" * 78,
        "SYSTEM PROMPT",
        "=" * 78,
        system_prompt,
        "",
        "=" * 78,
        "USER PROMPT",
        "=" * 78,
        user_prompt,
        "",
        "=" * 78,
        "RESPONSE",
        "=" * 78,
        response,
    ]
    if result.get("parsed") is not None:
        lines += [
            "",
            "=" * 78,
            "PARSED JSON",
            "=" * 78,
            json.dumps(result["parsed"], indent=2, ensure_ascii=False),
        ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
