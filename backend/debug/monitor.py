"""
monitor.py - PipelineMonitor: Orchestrator untuk Debug & Monitoring System
==========================================================================

PipelineMonitor adalah entry point utama untuk monitoring pipeline.
Dia mengoordinasikan:
  - LLMCollector: analisis kualitas teks output LLM
  - Storage: menyimpan events ke JSONL
  - SessionAnalyzer: ringkasan timing & anomaly di akhir session

Desain prinsip:
  - NON-DUPLIKAT dengan TrajectoryPool (trajectory = hasil, monitor = proses)
  - NON-DUPLIKAT dengan _save_output_snapshot (snapshot = raw text per call)
  - Lightweight: tidak menambah overhead signifikan ke pipeline
  - Safe: semua operasi wrapped dalam try/except agar tidak mengganggu pipeline

Usage:
------
    from debug import get_monitor

    monitor = get_monitor()

    # Pipeline step tracking
    with monitor.track_step("factor_propose", loop_idx=0):
        hypothesis = hypothesis_gen.gen(trace)

    # LLM output analysis
    monitor.analyze_llm_output(text, caller="propose")

    # Session summary
    monitor.finalize()
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from debug.events import (
    MonitorEvent,
    EventType,
    pipeline_step_start,
    pipeline_step_end,
    pipeline_step_error,
    pipeline_loop_start,
    pipeline_loop_end,
    pipeline_loop_skipped,
    pipeline_loop_traceback,
    llm_call_start,
    llm_call_end,
    evolution_round,
)
from debug.collectors import LLMCollector
from debug.analyzers import SessionAnalyzer
from debug.storage import EventWriter

# Optional torch (untuk GPU memory probe saja)
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class PipelineMonitor:
    """
    Orchestrator utama untuk pipeline monitoring.

    Lifecycle:
        1. __init__() → buat session, buka writer
        2. track_step() / analyze_llm_output() / track_llm_call_* → collect events
        3. finalize() → analyze + write summary + close

    All methods are safe — exceptions are caught and logged,
    never propagated to the calling pipeline code.
    """

    def __init__(
        self,
        session_name: Optional[str] = None,
        log_dir: str = "./debug/logs",
        console_echo: bool = False,
        enabled: bool = True,
    ):
        """
        Parameters:
            session_name : nama session (default: auto-timestamp)
            log_dir      : direktori output
            console_echo : print events ke console juga?
            enabled      : False = semua method jadi no-op (zero overhead)
        """
        self.enabled = enabled
        self.console_echo = console_echo
        self._events: List[MonitorEvent] = []
        self._session_start = time.time()

        if enabled:
            self._writer = EventWriter(log_dir=log_dir, session_name=session_name)
            self.session_id = self._writer.session_name
        else:
            self._writer = None
            self.session_id = ""

        # Context: current pipeline state
        self._current_loop_idx = 0
        self._current_direction_id = 0
        self._current_phase = "original"
        self._current_round_idx = 0

    def set_context(
        self,
        loop_idx: int = None,
        direction_id: int = None,
        phase: str = None,
        round_idx: int = None,
    ):
        """Update pipeline context (dipakai oleh loop.py di awal tiap iterasi)."""
        if loop_idx is not None:
            self._current_loop_idx = loop_idx
        if direction_id is not None:
            self._current_direction_id = direction_id
        if phase is not None:
            self._current_phase = phase
        if round_idx is not None:
            self._current_round_idx = round_idx

    # ─────────────────────────────────────────────────────────────────────
    # PIPELINE STEP TRACKING
    # ─────────────────────────────────────────────────────────────────────

    @contextmanager
    def track_step(self, step_name: str, **extra):
        """
        Context manager untuk tracking pipeline step.

        Usage:
            with monitor.track_step("factor_propose"):
                result = hypothesis_gen.gen(trace)
        """
        if not self.enabled:
            yield
            return

        try:
            # Emit start event
            start_evt = pipeline_step_start(
                step_name=step_name,
                loop_idx=self._current_loop_idx,
                direction_id=self._current_direction_id,
                phase=self._current_phase,
                round_idx=self._current_round_idx,
                **extra,
            )
            self._record(start_evt)

            gpu_mem_before = self._get_gpu_memory()
            t0 = time.time()

        except Exception:
            yield
            return

        error_occurred = False
        try:
            yield
        except Exception as e:
            error_occurred = True
            duration = time.time() - t0
            try:
                err_evt = pipeline_step_error(
                    step_name=step_name,
                    error_type=type(e).__name__,
                    error_message=str(e)[:500],
                    duration_s=duration,
                )
                self._record(err_evt)
            except Exception:
                pass
            raise
        finally:
            if not error_occurred:
                duration = time.time() - t0
                try:
                    gpu_mem_after = self._get_gpu_memory()
                    gpu_delta = None
                    if gpu_mem_before is not None and gpu_mem_after is not None:
                        gpu_delta = gpu_mem_after - gpu_mem_before

                    end_evt = pipeline_step_end(
                        step_name=step_name,
                        duration_s=duration,
                        gpu_memory_mb=gpu_mem_after,
                        gpu_memory_delta_mb=gpu_delta,
                    )
                    self._record(end_evt)
                except Exception:
                    pass

    # ─────────────────────────────────────────────────────────────────────
    # LOOP-LEVEL TRACKING (dipanggil oleh LoopBase.run di utils/workflow.py)
    # ─────────────────────────────────────────────────────────────────────

    def track_loop_start(self, loop_idx: int, total_steps: int = 0):
        """Record awal satu iterasi loop dan update context."""
        if not self.enabled:
            return
        try:
            self._current_loop_idx = loop_idx
            self._record(pipeline_loop_start(loop_idx=loop_idx, total_steps=total_steps))
        except Exception:
            pass

    def track_loop_end(self, loop_idx: int, duration_s: float, completed_steps: int = 0):
        """Record akhir iterasi loop yang selesai penuh."""
        if not self.enabled:
            return
        try:
            self._record(pipeline_loop_end(
                loop_idx=loop_idx,
                duration_s=duration_s,
                completed_steps=completed_steps,
            ))
        except Exception:
            pass

    def track_loop_skipped(
        self,
        loop_idx: int,
        step_name: str,
        error_type: str,
        error_message: str,
    ):
        """Record loop yang di-skip karena skip_loop_error."""
        if not self.enabled:
            return
        try:
            self._record(pipeline_loop_skipped(
                loop_idx=loop_idx,
                step_name=step_name,
                error_type=error_type,
                error_message=error_message,
            ))
        except Exception:
            pass

    def track_loop_traceback(
        self,
        loop_idx: int,
        step_name: str,
        error_type: str,
        error_message: str,
    ):
        """Record loop yang di-traceback (step_idx reset) karena CoderError."""
        if not self.enabled:
            return
        try:
            self._record(pipeline_loop_traceback(
                loop_idx=loop_idx,
                step_name=step_name,
                error_type=error_type,
                error_message=error_message,
            ))
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # LLM TRACKING
    # ─────────────────────────────────────────────────────────────────────

    def track_llm_call_start(
        self,
        caller: str,
        mode: str = "text",
        input_tokens: int = 0,
        temperature: float = 0.0,
        latent_steps: int = 0,
        has_past_kv: bool = False,
    ):
        """Record LLM call start."""
        if not self.enabled:
            return
        try:
            evt = llm_call_start(
                caller=caller,
                mode=mode,
                input_tokens=input_tokens,
                temperature=temperature,
                latent_steps=latent_steps,
                has_past_kv=has_past_kv,
            )
            self._record(evt)
        except Exception:
            pass

    def track_llm_call_end(
        self,
        caller: str,
        duration_s: float,
        output_tokens: int = 0,
        tokens_per_sec: float = 0.0,
        total_tokens: int = 0,
        mode: str = "text",
    ):
        """Record LLM call end."""
        if not self.enabled:
            return
        try:
            evt = llm_call_end(
                caller=caller,
                duration_s=duration_s,
                output_tokens=output_tokens,
                tokens_per_sec=tokens_per_sec,
                total_tokens=total_tokens,
                mode=mode,
            )
            self._record(evt)
        except Exception:
            pass

    def analyze_llm_output(
        self,
        text: str,
        caller: str = "unknown",
        token_ids: Optional[list] = None,
    ):
        """Analisis kualitas output LLM dan record events + anomalies."""
        if not self.enabled:
            return
        try:
            quality_evt, anomalies = LLMCollector.analyze_text_output(
                text, caller=caller, token_ids=token_ids,
            )
            self._record(quality_evt)
            for a in anomalies:
                self._record(a)
                if self.console_echo:
                    sev = a.data.get("severity", "")
                    desc = a.data.get("description", "")
                    print(f"[MONITOR ANOMALY] [{sev}] {desc}")
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # EVOLUTION TRACKING
    # ─────────────────────────────────────────────────────────────────────

    def track_evolution_round(
        self,
        round_idx: int,
        phase: str,
        direction_id: int,
        trajectory_id: str,
        parent_ids: List[str],
        primary_metric: Optional[float] = None,
        is_successful: bool = False,
    ):
        """Record evolution round completion."""
        if not self.enabled:
            return
        try:
            evt = evolution_round(
                round_idx=round_idx,
                phase=phase,
                direction_id=direction_id,
                trajectory_id=trajectory_id,
                parent_ids=parent_ids,
                primary_metric=primary_metric,
                is_successful=is_successful,
            )
            self._record(evt)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # SESSION LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────

    def finalize(self) -> Optional[Dict[str, Any]]:
        """
        Finalize session: run analysis, write summary, close files.

        Returns session summary dict.
        """
        if not self.enabled:
            return None

        try:
            # Run analysis
            summary = SessionAnalyzer.analyze(self._events)
            summary["session_duration_s"] = round(time.time() - self._session_start, 2)

            # Write summary
            if self._writer:
                self._writer.write_summary(summary)
                self._writer.close()

            if self.console_echo:
                self._print_summary(summary)

            return summary

        except Exception as e:
            if self._writer:
                self._writer.close()
            return {"error": str(e)}

    def get_events(self) -> List[MonitorEvent]:
        """Get semua events yang sudah di-record (untuk analysis)."""
        return list(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    # ─────────────────────────────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────────────────────────────

    def _record(self, event: MonitorEvent):
        """Record event ke memory dan storage."""
        event.session_id = self.session_id
        self._events.append(event)
        if self._writer:
            self._writer.write(event)

    @staticmethod
    def _get_gpu_memory() -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if not _HAS_TORCH or not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            return None

    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary ke console."""
        print("\n" + "=" * 70)
        print("  PIPELINE MONITOR - SESSION SUMMARY")
        print("=" * 70)

        print(f"  Duration   : {summary.get('session_duration_s', 0):.1f}s")
        print(f"  Events     : {summary.get('total_events', 0)}")

        # Pipeline
        pipeline = summary.get("pipeline", {})
        if pipeline.get("steps"):
            print(f"\n  Pipeline Steps:")
            for step, info in pipeline["steps"].items():
                print(f"    {step:20s} → {info['avg_s']:.3f}s avg × {info['call_count']} calls")
            bn = pipeline.get("bottleneck")
            if bn:
                print(f"    Bottleneck: {bn['step']} ({bn['percentage']}% of total)")

        # LLM
        llm = summary.get("llm", {})
        if llm.get("total_calls"):
            print(f"\n  LLM:")
            print(f"    Calls        : {llm['total_calls']}")
            print(f"    Total tokens : {llm.get('total_output_tokens', 0)}")
            print(f"    Avg quality  : {llm.get('avg_quality_score', 'N/A')}")

        # Anomalies
        anomalies = summary.get("anomalies", {})
        if anomalies.get("total", 0) > 0:
            print(f"\n  Anomalies: {anomalies['total']}")
            for sev, count in anomalies.get("by_severity", {}).items():
                print(f"    {sev:10s} : {count}")

        print("=" * 70 + "\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        return False


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL SINGLETON
# ═════════════════════════════════════════════════════════════════════════════

_global_monitor: Optional[PipelineMonitor] = None


def get_monitor(auto_create: bool = True) -> PipelineMonitor:
    """
    Get global PipelineMonitor instance.

    Kalau belum ada dan auto_create=True, buat instance baru (enabled=True).
    """
    global _global_monitor
    if _global_monitor is None and auto_create:
        _global_monitor = PipelineMonitor()
    return _global_monitor


def set_monitor(monitor: PipelineMonitor):
    """Set global monitor instance (dipanggil di awal pipeline run)."""
    global _global_monitor
    _global_monitor = monitor


def reset_monitor():
    """Reset global monitor (finalize + clear)."""
    global _global_monitor
    if _global_monitor is not None:
        _global_monitor.finalize()
        _global_monitor = None
