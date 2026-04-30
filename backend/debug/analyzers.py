"""
analyzers.py - Post-hoc Analysis dari Monitor Events
=====================================================

SessionAnalyzer membaca event history dan menghasilkan insight ringkas:
- Total duration dan per-step breakdown
- Bottleneck detection (step mana paling lambat)
- Anomaly summary (berapa banyak, severity apa)
- LLM call statistics

Hanya dijalankan di akhir session via PipelineMonitor.finalize().
Output dict bisa di-print atau di-save ke summary.json.

Catatan: TensorAnalyzer & DriftAnalyzer di-buang pada simplifikasi
2026-04-28 — sumber datanya (TensorCollector, KVCacheCollector) sudah
dihapus karena tidak actionable selama fase debugging.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from debug.events import EventType, MonitorEvent


class SessionAnalyzer:
    """
    Analisis keseluruhan satu monitoring session.

    Menghasilkan:
    - Total duration dan per-step breakdown
    - Bottleneck detection (step mana paling lambat)
    - Anomaly summary (berapa banyak, severity apa)
    - LLM call statistics
    """

    @staticmethod
    def analyze(events: List[MonitorEvent]) -> Dict[str, Any]:
        """Analisis list of events dan return summary dict."""
        if not events:
            return {"status": "no_events"}

        summary: Dict[str, Any] = {
            "total_events": len(events),
            "event_type_counts": {},
            "pipeline": {},
            "llm": {},
            "anomalies": {},
        }

        # Count event types
        type_counts = defaultdict(int)
        for evt in events:
            type_counts[evt.event_type] += 1
        summary["event_type_counts"] = dict(type_counts)

        # Pipeline timing
        summary["pipeline"] = SessionAnalyzer._analyze_pipeline(events)

        # LLM stats
        summary["llm"] = SessionAnalyzer._analyze_llm(events)

        # Anomalies
        summary["anomalies"] = SessionAnalyzer._analyze_anomalies(events)

        return summary

    @staticmethod
    def _analyze_pipeline(events: List[MonitorEvent]) -> Dict[str, Any]:
        """Analisis pipeline step timing."""
        step_durations: Dict[str, List[float]] = defaultdict(list)
        step_errors: Dict[str, int] = defaultdict(int)

        for evt in events:
            if evt.event_type == EventType.PIPELINE_STEP_END:
                step = evt.data.get("step_name", "unknown")
                duration = evt.data.get("duration_s", 0)
                step_durations[step].append(duration)
            elif evt.event_type == EventType.PIPELINE_STEP_ERROR:
                step = evt.data.get("step_name", "unknown")
                step_errors[step] += 1

        result = {}
        total_time = 0
        for step, durations in step_durations.items():
            avg_dur = sum(durations) / len(durations)
            total_time += sum(durations)
            result[step] = {
                "call_count": len(durations),
                "total_s": round(sum(durations), 3),
                "avg_s": round(avg_dur, 3),
                "min_s": round(min(durations), 3),
                "max_s": round(max(durations), 3),
                "errors": step_errors.get(step, 0),
            }

        # Bottleneck: step dengan total waktu terbesar
        bottleneck = None
        if result:
            bottleneck = max(result.items(), key=lambda x: x[1]["total_s"])
            bottleneck = {
                "step": bottleneck[0],
                "total_s": bottleneck[1]["total_s"],
                "percentage": round(bottleneck[1]["total_s"] / max(total_time, 0.001) * 100, 1),
            }

        return {
            "steps": result,
            "total_pipeline_time_s": round(total_time, 3),
            "bottleneck": bottleneck,
        }

    @staticmethod
    def _analyze_llm(events: List[MonitorEvent]) -> Dict[str, Any]:
        """Analisis LLM call statistics."""
        call_durations: List[float] = []
        total_output_tokens = 0
        callers: Dict[str, int] = defaultdict(int)

        quality_scores: List[float] = []

        for evt in events:
            if evt.event_type == EventType.LLM_CALL_END:
                call_durations.append(evt.data.get("duration_s", 0))
                total_output_tokens += evt.data.get("output_tokens", 0)
                callers[evt.data.get("caller", "unknown")] += 1

            elif evt.event_type == EventType.LLM_OUTPUT_QUALITY:
                # Skor kualitas sederhana: unique_ratio * (1 - repetition_ratio)
                unique = evt.data.get("unique_token_ratio", 0.5)
                rep = evt.data.get("repetition_ratio", 0.0)
                quality_scores.append(unique * (1 - rep))

        result: Dict[str, Any] = {
            "total_calls": len(call_durations),
            "total_output_tokens": total_output_tokens,
            "callers": dict(callers),
        }

        if call_durations:
            result["avg_duration_s"] = round(sum(call_durations) / len(call_durations), 3)
            result["total_duration_s"] = round(sum(call_durations), 3)

        if quality_scores:
            result["avg_quality_score"] = round(sum(quality_scores) / len(quality_scores), 4)
            result["min_quality_score"] = round(min(quality_scores), 4)

        return result

    @staticmethod
    def _analyze_anomalies(events: List[MonitorEvent]) -> Dict[str, Any]:
        """Ringkasan anomali terdeteksi."""
        by_severity: Dict[str, int] = defaultdict(int)
        by_type: Dict[str, int] = defaultdict(int)
        details: List[Dict[str, str]] = []

        for evt in events:
            if evt.event_type == EventType.ANOMALY_DETECTED:
                severity = evt.data.get("severity", "unknown")
                atype = evt.data.get("anomaly_type", "unknown")
                by_severity[severity] += 1
                by_type[atype] += 1
                details.append({
                    "type": atype,
                    "severity": severity,
                    "description": evt.data.get("description", ""),
                    "time": evt.iso_time,
                })

        return {
            "total": sum(by_severity.values()),
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "details": details[-20:],  # Last 20
        }
