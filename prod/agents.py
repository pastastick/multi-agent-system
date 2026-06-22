"""prod/agents.py — pemuat agent produksi (di atas latent_mas.agent).

Memuat agent dari prod/prompts.yaml. keep_answer_in_kv sudah di-set per-agent di
YAML (NO-CROP). Di sini kita override latent_steps & decode params per RunConfig.
"""
from __future__ import annotations

from typing import Any

from .config import PROMPTS_YAML, RunConfig


def load_prod_agent(name: str, backend: Any, cfg: RunConfig) -> Any:
    """load_agent(name, path=prod/prompts.yaml) + set latent_steps & decode params.

    Semua agent produksi mode kv_and_text (decode + KV). Untuk mode 'text',
    latent_steps efektif = 0 (lihat RunConfig.effective_latent_steps).
    """
    from latent_mas.agent import load_agent

    ag = load_agent(name, backend, strict_vars=False, path=PROMPTS_YAML)
    ag.spec.latent_steps = cfg.effective_latent_steps
    if ag.spec.temperature is None:
        ag.spec.temperature = cfg.decode_temperature
    if ag.spec.max_new_tokens is None:
        ag.spec.max_new_tokens = cfg.max_new_tokens
    return ag
