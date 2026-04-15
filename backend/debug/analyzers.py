"""
analyzers.py - Post-hoc Analysis dari Monitor Events
=====================================================

Analyzers membaca event history dan menghasilkan insight:
- SessionAnalyzer  : ringkasan satu session (timing, bottleneck, health)
- DriftAnalyzer    : tren hidden state drift across iterations
- QualityScorer    : skor kualitas output LLM per step

Analyzers dijalankan di akhir session atau on-demand.
Output berupa dict summary yang bisa di-print atau di-save.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from debug.events import EventType, MonitorEvent


class SessionAnalyzer:
    """
    Analisis keseluruhan satu monitoring session.

    Menghasilkan:
    - Total duration dan per-step breakdown
    - Bottleneck detection (step mana paling lambat)
    - Anomaly summary (berapa banyak, severity apa)
    - LLM call statistics
    - KV-cache growth pattern
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
            "kv_cache": {},
            "tensor_health": {},
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

        # KV-cache
        summary["kv_cache"] = SessionAnalyzer._analyze_kv_cache(events)

        # Tensor health
        summary["tensor_health"] = SessionAnalyzer._analyze_tensors(events)

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

    @staticmethod
    def _analyze_kv_cache(events: List[MonitorEvent]) -> Dict[str, Any]:
        """KV-cache growth dan pruning pattern."""
        statuses: List[Dict[str, Any]] = []
        prune_events: List[Dict[str, Any]] = []

        for evt in events:
            if evt.event_type == EventType.KV_CACHE_STATUS:
                statuses.append(evt.data)
            elif evt.event_type == EventType.KV_CACHE_PRUNE:
                prune_events.append(evt.data)

        result: Dict[str, Any] = {
            "measurements": len(statuses),
            "prune_count": len(prune_events),
        }

        if statuses:
            seq_lens = [s.get("seq_len", 0) for s in statuses]
            result["max_seq_len"] = max(seq_lens)
            result["final_seq_len"] = seq_lens[-1]
            sizes = [s.get("size_mb", 0) for s in statuses]
            result["max_size_mb"] = round(max(sizes), 2)

        if prune_events:
            total_removed = sum(p.get("tokens_removed", 0) for p in prune_events)
            result["total_tokens_pruned"] = total_removed

        return result

    @staticmethod
    def _analyze_tensors(events: List[MonitorEvent]) -> Dict[str, Any]:
        """Tensor health summary."""
        health_records: Dict[str, List[Dict]] = defaultdict(list)
        drift_records: List[Dict] = []

        for evt in events:
            if evt.event_type == EventType.TENSOR_HEALTH:
                name = evt.data.get("name", "unknown")
                health_records[name].append(evt.data)
            elif evt.event_type == EventType.HIDDEN_STATE_DRIFT:
                drift_records.append(evt.data)

        tensor_summary = {}
        for name, records in health_records.items():
            norms = [r["norm"] for r in records]
            tensor_summary[name] = {
                "check_count": len(records),
                "avg_norm": round(sum(norms) / len(norms), 4),
                "any_nan": any(r.get("has_nan") for r in records),
                "any_inf": any(r.get("has_inf") for r in records),
            }

        drift_summary = {}
        if drift_records:
            cos_sims = [d["cosine_similarity"] for d in drift_records]
            drift_summary = {
                "measurements": len(drift_records),
                "avg_cosine_similarity": round(sum(cos_sims) / len(cos_sims), 4),
                "min_cosine_similarity": round(min(cos_sims), 4),
                "max_cosine_similarity": round(max(cos_sims), 4),
            }

        return {
            "tensors": tensor_summary,
            "drift": drift_summary,
        }


class DriftAnalyzer:
    """
    Analisis tren drift antar iterasi.

    Mendeteksi:
    - Convergence: cosine similarity meningkat → model converge
    - Divergence: cosine similarity menurun → model diverge
    - Stagnation: cosine similarity ≈ 1.0 terus → model stuck
    """

    @staticmethod
    def analyze_trend(events: List[MonitorEvent]) -> Dict[str, Any]:
        """Analisis tren drift dari event history."""
        drift_events = [
            e for e in events
            if e.event_type == EventType.HIDDEN_STATE_DRIFT
        ]

        if len(drift_events) < 2:
            return {"status": "insufficient_data", "points": len(drift_events)}

        cos_sims = [e.data["cosine_similarity"] for e in drift_events]
        iterations = [e.data["iteration"] for e in drift_events]

        # Trend: comparing first half vs second half
        mid = len(cos_sims) // 2
        first_half_avg = sum(cos_sims[:mid]) / mid
        second_half_avg = sum(cos_sims[mid:]) / (len(cos_sims) - mid)

        delta = second_half_avg - first_half_avg

        if all(s > 0.98 for s in cos_sims):
            trend = "stagnation"
            description = (
                "Model menghasilkan representasi yang hampir identik antar iterasi. "
                "Kemungkinan model stuck dan tidak mengeksplorasi strategi baru."
            )
        elif delta > 0.05:
            trend = "converging"
            description = (
                "Cosine similarity meningkat → model converging ke representasi stabil. "
                "Ini normal jika metrik backtest juga membaik."
            )
        elif delta < -0.05:
            trend = "diverging"
            description = (
                "Cosine similarity menurun → model diverging. "
                "Representasi semakin berbeda tiap iterasi. Periksa apakah ini disengaja (explorasi) "
                "atau tidak (instabilitas)."
            )
        else:
            trend = "stable"
            description = "Tidak ada tren signifikan pada drift hidden state."

        return {
            "trend": trend,
            "description": description,
            "first_half_avg_similarity": round(first_half_avg, 4),
            "second_half_avg_similarity": round(second_half_avg, 4),
            "delta": round(delta, 4),
            "data_points": len(cos_sims),
            "timeline": [
                {"iteration": it, "cosine_similarity": round(cs, 4)}
                for it, cs in zip(iterations, cos_sims)
            ],
        }
