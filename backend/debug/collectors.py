"""
collectors.py - Data Collectors for Pipeline Monitoring
========================================================

Collectors mengekstrak sinyal/metrik dari berbagai sumber:
- LLMCollector     : analisis output LLM (kualitas teks, token stats)
- TensorCollector  : health check tensor (norms, NaN, drift)
- KVCacheCollector : metrik KV-cache (size, growth, prune events)

Collectors TIDAK menyimpan data — mereka menghasilkan MonitorEvent
yang kemudian dikirim ke storage oleh PipelineMonitor.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from debug.events import (
    MonitorEvent,
    llm_output_quality,
    tensor_health,
    hidden_state_drift,
    kv_cache_status,
    kv_cache_prune,
    anomaly_detected,
    AnomalySeverity,
)

# Optional torch import (collectors bisa dipakai tanpa GPU)
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


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


# ═════════════════════════════════════════════════════════════════════════════
# TENSOR HEALTH COLLECTOR
# ═════════════════════════════════════════════════════════════════════════════

class TensorCollector:
    """
    Health check untuk tensor (hidden states, latent vectors, embeddings).

    Mendeteksi:
    - NaN / Inf values (tanda training/inference error)
    - Norm explosion atau collapse
    - Zero ratio tinggi (dead neurons)
    - Drift antar iterasi (perubahan distribusi)
    """

    # State untuk tracking drift
    _previous_states: Dict[str, "torch.Tensor"] = {}

    @staticmethod
    def check_tensor(
        t: "torch.Tensor",
        name: str = "tensor",
    ) -> Tuple[MonitorEvent, List[MonitorEvent]]:
        """
        Health check satu tensor.

        Returns:
            Tuple of (health_event, list_of_anomaly_events)
        """
        if not _HAS_TORCH:
            return None, []

        anomalies: List[MonitorEvent] = []
        t_float = t.detach().float()

        has_nan = bool(torch.isnan(t_float).any().item())
        has_inf = bool(torch.isinf(t_float).any().item())
        total_elements = t_float.numel()
        zero_count = (t_float == 0).sum().item()
        zero_ratio = zero_count / max(total_elements, 1)

        # Safe stats (handle NaN)
        if has_nan or has_inf:
            clean = t_float[~torch.isnan(t_float) & ~torch.isinf(t_float)]
            if clean.numel() > 0:
                norm_val = clean.norm().item()
                mean_val = clean.mean().item()
                std_val = clean.std().item()
                min_val = clean.min().item()
                max_val = clean.max().item()
            else:
                norm_val = mean_val = std_val = min_val = max_val = 0.0
        else:
            norm_val = t_float.norm().item()
            mean_val = t_float.mean().item()
            std_val = t_float.std().item()
            min_val = t_float.min().item()
            max_val = t_float.max().item()

        health_evt = tensor_health(
            name=name,
            shape=list(t.shape),
            dtype=str(t.dtype),
            norm=norm_val,
            mean=mean_val,
            std=std_val,
            min_val=min_val,
            max_val=max_val,
            has_nan=has_nan,
            has_inf=has_inf,
            zero_ratio=zero_ratio,
        )

        # Anomaly detection
        if has_nan:
            anomalies.append(anomaly_detected(
                anomaly_type="tensor_nan",
                severity=AnomalySeverity.CRITICAL,
                description=f"NaN terdeteksi di tensor '{name}' (shape={list(t.shape)})",
                context={"tensor_name": name, "shape": list(t.shape)},
            ))

        if has_inf:
            anomalies.append(anomaly_detected(
                anomaly_type="tensor_inf",
                severity=AnomalySeverity.CRITICAL,
                description=f"Inf terdeteksi di tensor '{name}'",
                context={"tensor_name": name},
            ))

        if norm_val > 1e6:
            anomalies.append(anomaly_detected(
                anomaly_type="norm_explosion",
                severity=AnomalySeverity.WARNING,
                description=f"Norm sangat besar ({norm_val:.2e}) pada tensor '{name}'",
                context={"norm": norm_val, "tensor_name": name},
            ))

        if norm_val < 1e-8 and total_elements > 1:
            anomalies.append(anomaly_detected(
                anomaly_type="norm_collapse",
                severity=AnomalySeverity.WARNING,
                description=f"Norm mendekati nol ({norm_val:.2e}) pada tensor '{name}'",
                context={"norm": norm_val, "tensor_name": name},
            ))

        if zero_ratio > 0.95 and total_elements > 10:
            anomalies.append(anomaly_detected(
                anomaly_type="mostly_zeros",
                severity=AnomalySeverity.WARNING,
                description=f"Tensor '{name}' hampir seluruhnya nol ({zero_ratio:.2%})",
                context={"zero_ratio": zero_ratio, "tensor_name": name},
            ))

        return health_evt, anomalies

    @classmethod
    def check_drift(
        cls,
        t: "torch.Tensor",
        name: str,
        step_name: str = "",
        iteration: int = 0,
    ) -> Optional[MonitorEvent]:
        """
        Cek cosine drift antara tensor sekarang dan sebelumnya (untuk nama yang sama).

        Berguna untuk mendeteksi apakah hidden state berubah drastis
        (bisa indikasi model diverge) atau tidak berubah sama sekali
        (bisa indikasi model stuck).
        """
        if not _HAS_TORCH:
            return None

        key = f"{name}_{step_name}"
        t_flat = t.detach().float().flatten()

        if key in cls._previous_states:
            prev_flat = cls._previous_states[key]
            if prev_flat.shape == t_flat.shape:
                cos_sim = torch.nn.functional.cosine_similarity(
                    t_flat.unsqueeze(0), prev_flat.unsqueeze(0)
                ).item()
                norm_current = t_flat.norm().item()
                norm_previous = prev_flat.norm().item()
                norm_ratio = norm_current / max(norm_previous, 1e-10)

                cls._previous_states[key] = t_flat.cpu()

                return hidden_state_drift(
                    step_name=step_name,
                    iteration=iteration,
                    cosine_similarity=cos_sim,
                    norm_current=norm_current,
                    norm_previous=norm_previous,
                    norm_ratio=norm_ratio,
                )

        cls._previous_states[key] = t_flat.cpu()
        return None

    @classmethod
    def reset_drift_state(cls):
        """Reset stored states (misal di awal session baru)."""
        cls._previous_states.clear()


# ═════════════════════════════════════════════════════════════════════════════
# KV-CACHE COLLECTOR
# ═════════════════════════════════════════════════════════════════════════════

class KVCacheCollector:
    """
    Mengumpulkan metrik dari KV-cache.

    Track:
    - Ukuran KV-cache di berbagai titik pipeline
    - Pruning events (truncate, KNN filter)
    - Growth rate antar step
    """

    @staticmethod
    def measure_kv_cache(
        kv: Any,
        step_name: str = "",
        source: str = "",
    ) -> Optional[MonitorEvent]:
        """Ukur status KV-cache."""
        if kv is None or not _HAS_TORCH:
            return None

        try:
            # KV-cache format: Tuple[Tuple[Tensor, Tensor], ...]
            n_layers = len(kv)
            if n_layers == 0:
                return None

            # seq_len dari layer pertama, key tensor
            seq_len = kv[0][0].shape[-2] if len(kv[0]) > 0 else 0

            # Estimasi size dalam MB
            total_bytes = 0
            for layer_kv in kv:
                for tensor in layer_kv:
                    if isinstance(tensor, torch.Tensor):
                        total_bytes += tensor.nelement() * tensor.element_size()
            size_mb = total_bytes / (1024 * 1024)

            return kv_cache_status(
                step_name=step_name,
                seq_len=seq_len,
                n_layers=n_layers,
                size_mb=size_mb,
                source=source,
            )
        except Exception:
            return None

    @staticmethod
    def record_prune(
        method: str,
        before_len: int,
        after_len: int,
        reason: str = "",
    ) -> MonitorEvent:
        """Record KV-cache prune event."""
        return kv_cache_prune(
            method=method,
            before_len=before_len,
            after_len=after_len,
            tokens_removed=before_len - after_len,
            reason=reason,
        )
