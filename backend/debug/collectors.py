"""
collectors.py - Data Collectors for Pipeline Monitoring
========================================================

Collectors mengekstrak sinyal dari output LLM:
- LLMCollector : analisis kualitas teks (repetition, diversity, anomaly)

Collectors TIDAK menyimpan data — mereka menghasilkan MonitorEvent
yang kemudian dikirim ke storage oleh PipelineMonitor.

Catatan: TensorCollector & KVCacheCollector di-buang pada simplifikasi
2026-04-28 — overhead tanpa actionable insight selama fase debugging.
KV-cache size masih bisa diukur dari `pipeline_step_end.gpu_memory_mb`,
NaN/Inf tensor langsung crash di tempat tanpa butuh detector terpisah.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple

from debug.events import (
    MonitorEvent,
    llm_output_quality,
    anomaly_detected,
    AnomalySeverity,
)


# ═════════════════════════════════════════════════════════════════════════════
# LLM OUTPUT COLLECTOR
# ═════════════════════════════════════════════════════════════════════════════

class LLMCollector:
    """
    Analisis kualitas output LLM.

    Mendeteksi:
    - Output kosong atau terlalu pendek
    - Repetisi berlebihan (n-gram repetition)
    - Token diversity rendah
    - Format output (ada JSON? ada kode?)
    """

    @staticmethod
    def analyze_text_output(
        text: str,
        caller: str = "unknown",
        token_ids: Optional[list] = None,
    ) -> Tuple[MonitorEvent, List[MonitorEvent]]:
        """
        Analisis teks output LLM.

        Returns:
            Tuple of (quality_event, list_of_anomaly_events)
        """
        anomalies: List[MonitorEvent] = []

        if not text or not text.strip():
            quality_evt = llm_output_quality(
                caller=caller,
                output_length=0,
                unique_token_ratio=0.0,
                repetition_ratio=1.0,
                avg_line_length=0.0,
                empty_output=True,
            )
            anomalies.append(anomaly_detected(
                anomaly_type="empty_output",
                severity=AnomalySeverity.CRITICAL,
                description=f"LLM menghasilkan output kosong (caller={caller})",
            ))
            return quality_evt, anomalies

        # Basic stats
        words = text.split()
        lines = text.strip().split("\n")
        output_length = len(text)
        avg_line_length = sum(len(l) for l in lines) / max(len(lines), 1)

        # Token diversity (word-level jika token_ids tidak tersedia)
        if token_ids:
            tokens = token_ids
        else:
            tokens = words
        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        unique_ratio = unique_tokens / max(total_tokens, 1)

        # N-gram repetition (bigram)
        repetition_ratio = LLMCollector._bigram_repetition_ratio(words)

        # Format detection
        has_json = bool(re.search(r'[\{\[]\s*"', text))
        has_code = bool(re.search(r'(def |class |import |return |if |for )', text))

        quality_evt = llm_output_quality(
            caller=caller,
            output_length=output_length,
            unique_token_ratio=unique_ratio,
            repetition_ratio=repetition_ratio,
            avg_line_length=avg_line_length,
            has_json=has_json,
            has_code=has_code,
        )

        # Anomaly detection
        if unique_ratio < 0.15 and total_tokens > 20:
            anomalies.append(anomaly_detected(
                anomaly_type="low_diversity",
                severity=AnomalySeverity.WARNING,
                description=(
                    f"Token diversity sangat rendah ({unique_ratio:.2%}). "
                    f"Model mungkin stuck dalam loop repetisi."
                ),
                context={"unique_ratio": unique_ratio, "caller": caller},
            ))

        if repetition_ratio > 0.6 and total_tokens > 30:
            anomalies.append(anomaly_detected(
                anomaly_type="high_repetition",
                severity=AnomalySeverity.WARNING,
                description=(
                    f"Repetisi bigram tinggi ({repetition_ratio:.2%}). "
                    f"Output kemungkinan degenerate."
                ),
                context={"repetition_ratio": repetition_ratio, "caller": caller},
            ))

        if output_length < 20 and not has_json:
            anomalies.append(anomaly_detected(
                anomaly_type="very_short_output",
                severity=AnomalySeverity.INFO,
                description=f"Output sangat pendek ({output_length} chars)",
                context={"length": output_length, "caller": caller},
            ))

        return quality_evt, anomalies

    @staticmethod
    def _bigram_repetition_ratio(words: list) -> float:
        """Hitung rasio bigram yang muncul > 1 kali."""
        if len(words) < 4:
            return 0.0
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        counts = Counter(bigrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / max(len(bigrams), 1)
