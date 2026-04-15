"""
events.py - Typed Event Dataclasses for Pipeline Monitoring
============================================================

Setiap event merepresentasikan satu kejadian yang dicatat oleh monitor.
Event bersifat immutable setelah dibuat dan bisa di-serialize ke JSON/JSONL.

Perbedaan dengan Trajectory Pool:
- TrajectoryPool menyimpan HASIL akhir per-round (hipotesis, metrik, feedback)
- Events menyimpan PROSES internal: timing, tensor health, LLM behavior, anomali

Perbedaan dengan TensorConvManager:
- TensorConvManager menyimpan raw tensor (.pt files)
- Events menyimpan STATISTIK ringkas dari tensor (norms, drift, distribution)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """Kategori event yang di-track monitor."""
    # Pipeline flow
    PIPELINE_STEP_START = "pipeline.step.start"
    PIPELINE_STEP_END = "pipeline.step.end"
    PIPELINE_STEP_ERROR = "pipeline.step.error"
    PIPELINE_LOOP_START = "pipeline.loop.start"
    PIPELINE_LOOP_END = "pipeline.loop.end"
    PIPELINE_LOOP_SKIPPED = "pipeline.loop.skipped"
    PIPELINE_LOOP_TRACEBACK = "pipeline.loop.traceback"

    # LLM generation
    LLM_CALL_START = "llm.call.start"
    LLM_CALL_END = "llm.call.end"
    LLM_OUTPUT_QUALITY = "llm.output.quality"

    # Tensor health
    TENSOR_HEALTH = "tensor.health"
    HIDDEN_STATE_DRIFT = "tensor.hidden.drift"

    # KV-cache
    KV_CACHE_STATUS = "kv.cache.status"
    KV_CACHE_PRUNE = "kv.cache.prune"

    # Anomaly
    ANOMALY_DETECTED = "anomaly.detected"

    # Evolution
    EVOLUTION_ROUND = "evolution.round"


@dataclass
class MonitorEvent:
    """
    Base event — semua event punya field ini.

    Fields:
        event_id   : UUID unik per event
        event_type : kategori event (EventType enum)
        timestamp  : waktu event dibuat (epoch float)
        iso_time   : waktu ISO 8601 (human-readable)
        session_id : ID session monitoring (per-run)
        data       : payload spesifik per event type
        tags       : label tambahan untuk filtering (misal: step=propose, phase=mutation)
    """
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    iso_time: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ke dict (JSON-safe)."""
        d = asdict(self)
        # Pastikan semua value JSON-serializable
        return d


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_step_start(
    step_name: str,
    loop_idx: int = 0,
    direction_id: int = 0,
    phase: str = "original",
    round_idx: int = 0,
    has_kv_input: bool = False,
    **extra,
) -> MonitorEvent:
    """Event: pipeline step dimulai."""
    return MonitorEvent(
        event_type=EventType.PIPELINE_STEP_START,
        data={
            "step_name": step_name,
            "loop_idx": loop_idx,
            "has_kv_input": has_kv_input,
            **extra,
        },
        tags={
            "step": step_name,
            "phase": phase,
            "direction_id": str(direction_id),
            "round_idx": str(round_idx),
        },
    )


def pipeline_step_end(
    step_name: str,
    duration_s: float,
    success: bool = True,
    output_summary: Optional[Dict[str, Any]] = None,
    gpu_memory_mb: Optional[float] = None,
    **extra,
) -> MonitorEvent:
    """Event: pipeline step selesai."""
    return MonitorEvent(
        event_type=EventType.PIPELINE_STEP_END,
        data={
            "step_name": step_name,
            "duration_s": round(duration_s, 4),
            "success": success,
            "output_summary": output_summary or {},
            "gpu_memory_mb": gpu_memory_mb,
            **extra,
        },
        tags={"step": step_name},
    )


def pipeline_step_error(
    step_name: str,
    error_type: str,
    error_message: str,
    duration_s: float = 0.0,
) -> MonitorEvent:
    """Event: pipeline step gagal."""
    return MonitorEvent(
        event_type=EventType.PIPELINE_STEP_ERROR,
        data={
            "step_name": step_name,
            "error_type": error_type,
            "error_message": error_message,
            "duration_s": round(duration_s, 4),
        },
        tags={"step": step_name},
    )


def pipeline_loop_start(loop_idx: int, total_steps: int = 0) -> MonitorEvent:
    """Event: satu iterasi loop (loop_idx) dimulai."""
    return MonitorEvent(
        event_type=EventType.PIPELINE_LOOP_START,
        data={"loop_idx": loop_idx, "total_steps": total_steps},
        tags={"loop_idx": str(loop_idx)},
    )


def pipeline_loop_end(
    loop_idx: int,
    duration_s: float,
    completed_steps: int = 0,
) -> MonitorEvent:
    """Event: satu iterasi loop (loop_idx) selesai penuh."""
    return MonitorEvent(
        event_type=EventType.PIPELINE_LOOP_END,
        data={
            "loop_idx": loop_idx,
            "duration_s": round(duration_s, 4),
            "completed_steps": completed_steps,
        },
        tags={"loop_idx": str(loop_idx)},
    )


def pipeline_loop_skipped(
    loop_idx: int,
    step_name: str,
    error_type: str,
    error_message: str,
) -> MonitorEvent:
    """Event: loop di-skip karena skip_loop_error."""
    return MonitorEvent(
        event_type=EventType.PIPELINE_LOOP_SKIPPED,
        data={
            "loop_idx": loop_idx,
            "step_name": step_name,
            "error_type": error_type,
            "error_message": error_message[:500],
        },
        tags={"loop_idx": str(loop_idx), "step": step_name},
    )


def pipeline_loop_traceback(
    loop_idx: int,
    step_name: str,
    error_type: str,
    error_message: str,
) -> MonitorEvent:
    """Event: loop di-traceback (step_idx reset) karena CoderError."""
    return MonitorEvent(
        event_type=EventType.PIPELINE_LOOP_TRACEBACK,
        data={
            "loop_idx": loop_idx,
            "step_name": step_name,
            "error_type": error_type,
            "error_message": error_message[:500],
        },
        tags={"loop_idx": str(loop_idx), "step": step_name},
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def llm_call_start(
    caller: str,
    mode: str = "text",
    input_tokens: int = 0,
    temperature: float = 0.0,
    latent_steps: int = 0,
    has_past_kv: bool = False,
) -> MonitorEvent:
    """Event: LLM call dimulai."""
    return MonitorEvent(
        event_type=EventType.LLM_CALL_START,
        data={
            "caller": caller,
            "mode": mode,
            "input_tokens": input_tokens,
            "temperature": temperature,
            "latent_steps": latent_steps,
            "has_past_kv": has_past_kv,
        },
        tags={"caller": caller, "mode": mode},
    )


def llm_call_end(
    caller: str,
    duration_s: float,
    output_tokens: int = 0,
    tokens_per_sec: float = 0.0,
    total_tokens: int = 0,
    mode: str = "text",
) -> MonitorEvent:
    """Event: LLM call selesai."""
    return MonitorEvent(
        event_type=EventType.LLM_CALL_END,
        data={
            "caller": caller,
            "duration_s": round(duration_s, 4),
            "output_tokens": output_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "total_tokens": total_tokens,
            "mode": mode,
        },
        tags={"caller": caller, "mode": mode},
    )


def llm_output_quality(
    caller: str,
    output_length: int,
    unique_token_ratio: float,
    repetition_ratio: float,
    avg_line_length: float,
    empty_output: bool = False,
    has_json: bool = False,
    has_code: bool = False,
) -> MonitorEvent:
    """
    Event: analisis kualitas output LLM.

    Metrics:
        unique_token_ratio : rasio token unik / total (rendah = repetitif)
        repetition_ratio   : rasio n-gram yang berulang (tinggi = repetitif)
        avg_line_length    : rata-rata panjang baris (terlalu pendek = degenerate)
    """
    return MonitorEvent(
        event_type=EventType.LLM_OUTPUT_QUALITY,
        data={
            "caller": caller,
            "output_length": output_length,
            "unique_token_ratio": round(unique_token_ratio, 4),
            "repetition_ratio": round(repetition_ratio, 4),
            "avg_line_length": round(avg_line_length, 2),
            "empty_output": empty_output,
            "has_json": has_json,
            "has_code": has_code,
        },
        tags={"caller": caller},
    )


# ─────────────────────────────────────────────────────────────────────────────
# TENSOR HEALTH EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def tensor_health(
    name: str,
    shape: List[int],
    dtype: str,
    norm: float,
    mean: float,
    std: float,
    min_val: float,
    max_val: float,
    has_nan: bool = False,
    has_inf: bool = False,
    zero_ratio: float = 0.0,
) -> MonitorEvent:
    """Event: health check sebuah tensor (hidden state, latent vec, dll)."""
    return MonitorEvent(
        event_type=EventType.TENSOR_HEALTH,
        data={
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "norm": round(norm, 6),
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(min_val, 6),
            "max": round(max_val, 6),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "zero_ratio": round(zero_ratio, 4),
        },
        tags={"tensor": name},
    )


def hidden_state_drift(
    step_name: str,
    iteration: int,
    cosine_similarity: float,
    norm_current: float,
    norm_previous: float,
    norm_ratio: float,
) -> MonitorEvent:
    """
    Event: drift antar hidden state dari iterasi berurutan.

    Cosine similarity rendah = model menghasilkan representasi yang sangat berbeda.
    Norm ratio jauh dari 1.0 = magnitude berubah drastis.
    """
    return MonitorEvent(
        event_type=EventType.HIDDEN_STATE_DRIFT,
        data={
            "step_name": step_name,
            "iteration": iteration,
            "cosine_similarity": round(cosine_similarity, 6),
            "norm_current": round(norm_current, 4),
            "norm_previous": round(norm_previous, 4),
            "norm_ratio": round(norm_ratio, 4),
        },
        tags={"step": step_name},
    )


# ─────────────────────────────────────────────────────────────────────────────
# KV-CACHE EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def kv_cache_status(
    step_name: str,
    seq_len: int,
    n_layers: int,
    size_mb: float,
    source: str = "",
) -> MonitorEvent:
    """Event: status KV-cache pada titik tertentu."""
    return MonitorEvent(
        event_type=EventType.KV_CACHE_STATUS,
        data={
            "step_name": step_name,
            "seq_len": seq_len,
            "n_layers": n_layers,
            "size_mb": round(size_mb, 2),
            "source": source,
        },
        tags={"step": step_name},
    )


def kv_cache_prune(
    method: str,
    before_len: int,
    after_len: int,
    tokens_removed: int,
    reason: str = "",
) -> MonitorEvent:
    """Event: KV-cache di-prune (truncate atau KNN filter)."""
    return MonitorEvent(
        event_type=EventType.KV_CACHE_PRUNE,
        data={
            "method": method,
            "before_len": before_len,
            "after_len": after_len,
            "tokens_removed": tokens_removed,
            "reason": reason,
        },
        tags={"method": method},
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY EVENTS
# ─────────────────────────────────────────────────────────────────────────────

class AnomalySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


def anomaly_detected(
    anomaly_type: str,
    severity: str,
    description: str,
    context: Optional[Dict[str, Any]] = None,
) -> MonitorEvent:
    """Event: anomali terdeteksi."""
    return MonitorEvent(
        event_type=EventType.ANOMALY_DETECTED,
        data={
            "anomaly_type": anomaly_type,
            "severity": severity,
            "description": description,
            "context": context or {},
        },
        tags={"anomaly_type": anomaly_type, "severity": severity},
    )


# ─────────────────────────────────────────────────────────────────────────────
# EVOLUTION EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def evolution_round(
    round_idx: int,
    phase: str,
    direction_id: int,
    trajectory_id: str,
    parent_ids: List[str],
    primary_metric: Optional[float] = None,
    is_successful: bool = False,
) -> MonitorEvent:
    """Event: satu round evolusi selesai."""
    return MonitorEvent(
        event_type=EventType.EVOLUTION_ROUND,
        data={
            "round_idx": round_idx,
            "phase": phase,
            "direction_id": direction_id,
            "trajectory_id": trajectory_id,
            "parent_ids": parent_ids,
            "primary_metric": primary_metric,
            "is_successful": is_successful,
        },
        tags={"phase": phase, "direction_id": str(direction_id)},
    )
