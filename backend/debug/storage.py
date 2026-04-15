"""
storage.py - Persistent Storage untuk Monitor Events
=====================================================

EventWriter menyimpan events ke format JSONL (satu event per baris).
JSONL dipilih karena:
- Append-friendly (tidak perlu baca seluruh file untuk menambah event)
- Mudah di-stream untuk training data pipeline
- Bisa di-parse per baris (memory efficient untuk file besar)
- Kompatibel dengan tools ML (pandas, HuggingFace datasets)

File structure:
    debug/logs/
        session_{timestamp}/
            events.jsonl        <- semua events
            summary.json        <- session summary (generated di akhir)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from debug.events import MonitorEvent


class EventWriter:
    """
    Append-only writer untuk MonitorEvent ke JSONL.

    Thread-safe melalui file-level flush (bukan lock).
    Setiap write langsung flush ke disk untuk data safety.
    """

    def __init__(
        self,
        log_dir: str = "./debug/logs",
        session_name: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{timestamp}"
        self.session_dir = self.log_dir / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.session_dir / "events.jsonl"
        self.summary_path = self.session_dir / "summary.json"

        self._event_count = 0
        self._file = None

    def _ensure_open(self):
        """Lazy open file."""
        if self._file is None:
            self._file = open(self.events_path, "a", encoding="utf-8")

    def write(self, event: MonitorEvent):
        """Write satu event ke JSONL."""
        self._ensure_open()
        line = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
        self._file.write(line + "\n")
        self._file.flush()
        self._event_count += 1

    def write_batch(self, events: List[MonitorEvent]):
        """Write multiple events sekaligus."""
        self._ensure_open()
        for event in events:
            line = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
            self._file.write(line + "\n")
            self._event_count += 1
        self._file.flush()

    def write_summary(self, summary: Dict[str, Any]):
        """Write session summary ke JSON file terpisah."""
        summary["session_name"] = self.session_name
        summary["total_events_written"] = self._event_count
        summary["generated_at"] = datetime.now().isoformat()

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    def close(self):
        """Flush dan tutup file."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    @property
    def event_count(self) -> int:
        return self._event_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# READER (untuk loading events dari JSONL)
# ─────────────────────────────────────────────────────────────────────────────

class EventReader:
    """
    Baca events dari JSONL file.

    Berguna untuk:
    - Post-hoc analysis
    - Loading data untuk training/finetuning
    - Debugging specific sessions
    """

    @staticmethod
    def read_events(jsonl_path: str) -> List[MonitorEvent]:
        """Baca semua events dari file JSONL."""
        events = []
        path = Path(jsonl_path)
        if not path.exists():
            return events

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    events.append(MonitorEvent(
                        event_type=d.get("event_type", ""),
                        data=d.get("data", {}),
                        tags=d.get("tags", {}),
                        event_id=d.get("event_id", ""),
                        timestamp=d.get("timestamp", 0),
                        iso_time=d.get("iso_time", ""),
                        session_id=d.get("session_id", ""),
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue

        return events

    @staticmethod
    def read_summary(session_dir: str) -> Optional[Dict[str, Any]]:
        """Baca session summary."""
        summary_path = Path(session_dir) / "summary.json"
        if not summary_path.exists():
            return None
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def list_sessions(log_dir: str = "./debug/logs") -> List[Dict[str, Any]]:
        """List semua session di log directory."""
        log_path = Path(log_dir)
        if not log_path.exists():
            return []

        sessions = []
        for d in sorted(log_path.iterdir()):
            if d.is_dir() and (d / "events.jsonl").exists():
                events_file = d / "events.jsonl"
                line_count = sum(1 for _ in open(events_file))
                sessions.append({
                    "name": d.name,
                    "path": str(d),
                    "event_count": line_count,
                    "has_summary": (d / "summary.json").exists(),
                    "created": datetime.fromtimestamp(
                        events_file.stat().st_ctime
                    ).isoformat(),
                })

        return sessions
