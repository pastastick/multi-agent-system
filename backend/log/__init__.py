"""
AlphaAgent logging module - compatibility layer.

Maps alphaagent.log to rdagent.log so all alphaagent.log imports work.
Provides AlphaAgent-specific APIs: log_trace_path, set_trace_path.
"""

import os
import pickle
from pathlib import Path
from loguru import logger as _loguru_logger
from rdagent.log import rdagent_logger as _rdagent_logger
from rdagent.log.utils import LogColors

#* Nama file sink untuk menangkap log konsol (mis. "Evaluated expr" dari regulator).
#* rdagent.log.logger hanya memasang sink konsol (stderr) tanpa sink file, jadi baris
#* logger.info/.warning/.error bersifat ephemeral. Sink ini mempersistenkannya ke disk
#* dengan FORMAT DEFAULT loguru — sama persis dengan yang tampil di terminal.
_CONSOLE_LOG_FILENAME = "console.log"


class _AlphaAgentLoggerWrapper:
    """
    Wraps rdagent_logger and adds log_trace_path / set_trace_path. Other attributes/methods delegate to rdagent_logger.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner) #* simpan rdagent_logger asli
        object.__setattr__(self, "_console_file_sink_id", None) #* id sink file loguru aktif
        self._attach_console_file(inner.storage.path) #* pasang sink awal di trace path saat ini

    # ---------- Console-format file sink ----------
    def _attach_console_file(self, trace_path) -> None:
        """(Re)arahkan satu sink file loguru ke <trace_path>/console.log.

        Memakai format DEFAULT loguru (tanpa argumen format) sehingga isi file
        identik dengan output konsol. Override lokasi via env LOG_CONSOLE_FILE
        (path tetap; tidak ikut berpindah saat set_trace_path)."""
        fixed = os.getenv("LOG_CONSOLE_FILE")
        target = Path(fixed) if fixed else Path(trace_path) / _CONSOLE_LOG_FILENAME

        prev_id = object.__getattribute__(self, "_console_file_sink_id")
        if fixed and prev_id is not None:
            return  #* path tetap: sink sudah terpasang, jangan dipindah

        if prev_id is not None:
            try:
                _loguru_logger.remove(prev_id) #* lepas sink lama sebelum pindah
            except ValueError:
                pass

        target.parent.mkdir(parents=True, exist_ok=True)
        #* level DEBUG agar semua baris (info/warning/error) tertangkap; format default = format konsol
        sink_id = _loguru_logger.add(str(target), level="DEBUG", encoding="utf-8")
        object.__setattr__(self, "_console_file_sink_id", sink_id)

    # ---------- AlphaAgent extension ----------
    @property
    def log_trace_path(self) -> Path:
        """Return current log trace path."""
        return self._inner.storage.path #* path tempat log disimpan

    def set_trace_path(self, path) -> None:
        """Set new log trace path."""
        from rdagent.log.storage import FileStorage
        self._inner.storage = FileStorage(Path(path)) #*ubah path penyimpanan log
        self._attach_console_file(path) #* ikutkan sink file konsol ke trace path baru

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

    # ---------- Compat: rdagent's RDAgentLog has info/warning/error but NO debug ----------
    def debug(self, *args, **kwargs) -> None:
        """RDAgentLog tidak punya .debug → delegasi bila ada, selain itu no-op.
        Tanpa shim ini, setiap `logger.debug(...)` (mis. graceful parse-skip di
        factor_regulator.validate_function_arity) melempar AttributeError yang
        merambat ke gate → gate fail-closed dengan error palsu."""
        fn = getattr(self._inner, "debug", None)
        if callable(fn):
            fn(*args, **kwargs)

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
