"""
debug/ - Pipeline Monitoring & Debug System
============================================

Sistem monitoring ringan untuk memantau proses internal pipeline:
- Pipeline step timing & error tracking
- Loop iteration tracking (start/end/skipped/traceback)
- LLM output quality analysis (repetition, diversity, anomaly)
- Evolution round tracking
- GPU memory snapshot per step

TIDAK menduplikasi:
- TrajectoryPool (menyimpan HASIL: hipotesis, metrik, feedback)
- TensorConvManager (menyimpan RAW tensor: input_ids, output_ids, hidden_last)
- LLM output snapshot di llm/client.py (debug/llm_outputs/session_*/...)

Monitor menyimpan PROSES & STATISTIK ringan: timing, quality scores,
anomali — events di JSONL untuk inspect real-time atau post-hoc.

Quick start:
    from debug import get_monitor
    monitor = get_monitor()

    with monitor.track_step("factor_propose"):
        result = do_work()

    monitor.analyze_llm_output(text, caller="propose")

File structure:
    debug/
    ├── __init__.py       ← Public API (this file)
    ├── monitor.py        ← PipelineMonitor orchestrator
    ├── events.py         ← Typed event dataclasses
    ├── collectors.py     ← LLMCollector
    ├── analyzers.py      ← SessionAnalyzer (post-hoc summary)
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
