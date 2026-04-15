"""
llm/config.py
=============
LLM configuration settings used across the pipeline.
"""

from core.conf import ExtendedBaseSettings


class LLMConfig(ExtendedBaseSettings):
    """Settings for LLM backends. Values can be overridden via env vars or config file."""

    # Token limits
    chat_token_limit: int = 8192

    # Timeout
    factor_mining_timeout: int = 3600  # seconds

    # Cache
    init_chat_cache_seed: int = 42

    # Embedding
    embedding_max_str_num: int = 50


LLM_SETTINGS = LLMConfig()
