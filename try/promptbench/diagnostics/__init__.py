"""Diagnostik Phase B: deteksi collapse KV-cache & degenerasi teks."""
from .collapse import (
    BoundaryRecord,
    detect_kv_growth,
    detect_text_collapse,
    repetition_ratio,
    summarize_chain_health,
)

__all__ = [
    "BoundaryRecord",
    "detect_kv_growth",
    "detect_text_collapse",
    "repetition_ratio",
    "summarize_chain_health",
]
