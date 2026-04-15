"""
debug/ - Pipeline Monitoring & Debug System
============================================

Sistem monitoring untuk memantau proses internal pipeline:
- Pipeline step timing & bottleneck detection
- LLM output quality analysis (repetition, diversity, anomalies)
- Tensor health monitoring (NaN, Inf, norm collapse/explosion)
- Hidden state drift tracking antar iterasi
- KV-cache size & pruning metrics
- Anomaly detection & alerting

TIDAK menduplikasi:
- TrajectoryPool (menyimpan HASIL: hipotesis, metrik, feedback)
- TensorConvManager (menyimpan RAW tensor: input_ids, output_ids, hidden_last)

Monitor menyimpan PROSES & STATISTIK: timing, quality scores, tensor stats,
drift analysis, anomalies — data yang berguna untuk debugging dan future
training/finetuning.

Quick start:
    from debug import get_monitor
    monitor = get_monitor()

    with monitor.track_step("factor_propose"):
        result = do_work()

    monitor.analyze_llm_output(text, caller="propose")
    monitor.check_tensor(hidden_state, name="hidden_last")

File structure:
    debug/
    ├── __init__.py       ← Public API (this file)
    ├── monitor.py        ← PipelineMonitor orchestrator
    ├── events.py         ← Typed event dataclasses
    ├── collectors.py     ← LLM, Tensor, KV-cache data collectors
    ├── analyzers.py      ← Session analysis, drift analysis
    ├── storage.py        ← JSONL event writer & reader
    └── logs/             ← Output directory (auto-created)
"""

from debug.monitor import (
    PipelineMonitor,
    get_monitor,
    set_monitor,
    reset_monitor,
)
from debug.events import EventType, MonitorEvent, AnomalySeverity
from debug.storage import EventReader

__all__ = [
    "PipelineMonitor",
    "get_monitor",
    "set_monitor",
    "reset_monitor",
    "EventType",
    "MonitorEvent",
    "AnomalySeverity",
    "EventReader",
]
