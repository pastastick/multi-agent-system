"""
llm package
===========
Public API for all LLM backends used in 

Module structure:
    _shared.py      - Shared utilities (KVCache, LatentRealigner, helpers)
    client.py       - LocalLLMBackend (primary backend for pipeline)
    models.py       - Latent formula utilities (called by client.py)
    config.py       - LLM_SETTINGS
"""

# Shared types and utilities
from llm._shared import (
    KVCache,
    OutputMode,
    LatentRealigner,
    _past_length,
    _kv_to_cpu,
    _kv_to_device,
    kv_truncate,
    kv_knn_filter,
    kv_size_bytes,
    robust_json_parse,
    md5_hash,
)

# Primary backend
from llm.client import (
    LocalLLMBackend,
    LLMResult,
    KVCacheStore,
    TensorConvManager,
    LocalChatSession,
    get_local_backend,
    calculate_embedding_distance_between_str_list,
)

__all__ = [
    # Types
    "KVCache",
    "OutputMode",
    # Shared utilities
    "LatentRealigner",
    "_past_length",
    "_kv_to_cpu",
    "_kv_to_device",
    "kv_truncate",
    "kv_knn_filter",
    "kv_size_bytes",
    "robust_json_parse",
    "md5_hash",
    # Primary backend
    "LocalLLMBackend",
    "LLMResult",
    "KVCacheStore",
    "TensorConvManager",
    "LocalChatSession",
    "get_local_backend",
    "calculate_embedding_distance_between_str_list",
]
