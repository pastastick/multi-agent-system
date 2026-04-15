from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

import yaml

from log import logger
from llm.client import LocalLLMBackend

if TYPE_CHECKING:
    # Hindari circular import; hanya untuk type checker
    from eksternal.base import ExternalInsight


def _load_prompts(prompt_file: Path) -> dict[str, str]:
    if not prompt_file.exists():
        return {}
    try:
        return yaml.safe_load(prompt_file.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"Failed to load planning prompts: {exc}")
        return {}


def _extract_json(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    fence = re.search(r"```json\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def _parse_directions(message: str, n: int) -> list[str] | None:
    frag = _extract_json(message)
    try:
        data = json.loads(frag)
    except Exception:
        return None
    arr = data.get("directions") if isinstance(data, dict) else None
    if not isinstance(arr, list):
        return None
    vals = [str(x).strip() for x in arr if isinstance(x, str) and x.strip()]
    return vals if len(vals) >= n else None


def _fallback_directions(initial_direction: str, n: int) -> list[str]:
    base = initial_direction.strip() if initial_direction else "market microstructure"
    patterns = [
        f"{base} + short-term momentum signal with volume confirmation",
        f"{base} + volatility regime switch using rolling variance",
        f"{base} + liquidity/turnover adjustment for noise reduction",
        f"{base} + cross-sectional rank with sector-neutralization",
        f"{base} + intraday reversal vs overnight drift decomposition",
        f"{base} + fundamental proxy alignment (price-to-book, earnings momentum)",
        f"{base} + calendar effects and seasonality-aware normalization",
        f"{base} + risk-adjusted return features (downside volatility focus)",
    ]
    out = []
    for i in range(n):
        out.append(patterns[i % len(patterns)])
    return out


def generate_parallel_directions(
    initial_direction: str,
    n: int,
    prompt_file: Path,
    max_attempts: int = 5,
    use_llm: bool = True,
    allow_fallback: bool = True,
    # ── External agent integration ─────────────────────────────────────
    external_insights: Optional[List["ExternalInsight"]] = None,
    llm_backend: Optional[LocalLLMBackend] = None,
) -> list[str]:
    """
    Generate n parallel exploration directions from an initial direction.

    Args:
        initial_direction : seed direction string
        n                 : number of directions to generate
        prompt_file       : path to planning_prompts.yaml
        max_attempts      : LLM retry attempts
        use_llm           : if False, use fallback patterns only
        allow_fallback    : return fallback patterns if LLM fails
        external_insights : list of ExternalInsight from external agents.
                            - Summaries are prepended to the user prompt as context.
                            - First insight with kv_cache is passed as past_key_values
                              to the LLM, giving it latent macro context.
        llm_backend       : pre-built LocalLLMBackend instance.
                            Reuses the same model that built the KV-cache, which
                            is required for past_key_values to be compatible.
                            If None, a new LocalLLMBackend() is created (existing
                            behaviour, no KV-cache support).
    """
    n = max(1, int(n))
    prompts = _load_prompts(prompt_file)
    sys_tpl = prompts.get("system", "")
    user_tpl = prompts.get("user", "")
    output_format = prompts.get("output_format", "")

    system_prompt = sys_tpl.format(initial_direction=initial_direction, n=n)
    user_prompt = user_tpl.format(initial_direction=initial_direction, n=n)
    if output_format:
        if "{n}" in output_format:
            output_format = output_format.replace("{n}", str(n))
        user_prompt = f"{user_prompt}\n\n{output_format}"

    # ── Inject external context into user prompt ───────────────────────
    past_key_values = None
    if external_insights:
        context_blocks = [ins.to_context_str() for ins in external_insights]
        external_block = "\n\n".join(context_blocks)
        user_prompt = (
            f"[External Market Context]\n{external_block}\n\n"
            f"[Direction Generation Task]\n{user_prompt}"
        )
        # Use the KV-cache from the first insight that has one.
        # The KV-cache encodes the macro analysis latent state — the LLM
        # will generate directions "conditioned" on that context.
        for ins in external_insights:
            if ins.has_kv:
                past_key_values = ins.kv_cache
                logger.info(
                    f"[Planning] Using KV-cache from '{ins.source_agent}' "
                    f"(seq_len={ins.metadata.get('kv_seq_len', '?')})"
                )
                break

    if not use_llm:
        return _fallback_directions(initial_direction, n) if allow_fallback else []

    # ── Resolve backend ────────────────────────────────────────────────
    # Prefer pre-built backend (required when passing KV-cache).
    # Fall back to creating a new instance for backward compatibility.
    backend = llm_backend or LocalLLMBackend()

    for attempt in range(1, max_attempts + 1):
        try:
            resp = backend.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=False,
                past_key_values=past_key_values,
            )
            directions = _parse_directions(resp, n)
            if directions:
                return directions[:n]
            system_prompt += "\n\nStrictly output valid JSON. No extra text."
            logger.warning(f"Planning parse failed (attempt {attempt}), retrying...")
        except Exception as exc:
            logger.warning(f"Planning LLM call failed (attempt {attempt}): {exc}")

    return _fallback_directions(initial_direction, n) if allow_fallback else []


def load_run_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"Failed to load run config: {exc}")
        return {}

