"""
AlphaAgent logging module - compatibility layer.

Maps alphaagent.log to rdagent.log so all alphaagent.log imports work.
Provides AlphaAgent-specific APIs: log_trace_path, set_trace_path.
"""

import pickle
from pathlib import Path
from rdagent.log import rdagent_logger as _rdagent_logger
from rdagent.log.utils import LogColors


class _AlphaAgentLoggerWrapper:
    """
    Wraps rdagent_logger and adds log_trace_path / set_trace_path. Other attributes/methods delegate to rdagent_logger.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner) #* simpan rdagent_logger asli

    # ---------- AlphaAgent extension ----------
    @property
    def log_trace_path(self) -> Path:
        """Return current log trace path."""
        return self._inner.storage.path #* path tempat log disimpan

    def set_trace_path(self, path) -> None:
        """Set new log trace path."""
        from rdagent.log.storage import FileStorage
        self._inner.storage = FileStorage(Path(path)) #*ubah path penyimpanan log

    # ---------- Safe log_object (skip unpicklable) ----------
    def log_object(self, obj, *, tag: str = "") -> None:
        """Wrap rdagent log_object: skip gracefully if object can't be pickled.

        Latent pipeline objects (coder, hypothesis_generator, etc.) hold
        references to PyTorch models with thread locks — these can never
        be pickled. Crashing the pipeline for debug logging is not worth it.
        """
        try:
            self._inner.log_object(obj, tag=tag)
        except (TypeError, pickle.PicklingError) as exc:
            pass  # silently skip unpicklable objects

    # ---------- Delegate to rdagent_logger ----------
    def __getattr__(self, name):
        return getattr(self._inner, name) #* delegasi atribut/metode lain ke rdagent_logger

    def __setattr__(self, name, value):
        if name in ("_inner",):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


logger = _AlphaAgentLoggerWrapper(_rdagent_logger) #* buat instance logger yang membungkus rdagent_logger, sehingga semua fungsi logging tetap bisa digunakan, plus tambahan set_trace_path dan log_trace_path

__all__ = ["logger", "LogColors"]
