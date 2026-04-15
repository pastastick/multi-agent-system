"""
This is a class that try to store/resume/traceback the workflow session


Postscripts:
- Originally, I want to implement it in a more general way with python generator.
  However, Python generator is not picklable (dill does not support pickle as well)

Jadi ketika kita menulis:


class AlphaAgentLoop(LoopBase, metaclass=LoopMeta):
    def factor_propose(self, ...): ...
    def factor_construct(self, ...): ...
    def factor_calculate(self, ...): ...
    def factor_backtest(self, ...): ...
    def feedback(self, ...): ...
    def _get_trajectory_data(self, ...): ...  # dimulai "_" → TIDAK jadi step
Maka AlphaAgentLoop.steps otomatis menjadi:


["factor_propose", "factor_construct", "factor_calculate", "factor_backtest", "feedback"]
Tidak perlu mendaftarkan steps secara manual. Cukup tulis method, dan metaclass yang mengaturnya.


Visualisasi alur run():

Loop 0:
  step 0: factor_propose(prev_out={})                → prev_out = {factor_propose: idea}
  step 1: factor_construct(prev_out)                 → prev_out = {..., factor_construct: factor}
  step 2: factor_calculate(prev_out)                 → prev_out = {..., factor_calculate: coded}
  step 3: factor_backtest(prev_out)                  → prev_out = {..., factor_backtest: result}
  step 4: feedback(prev_out)                         → prev_out = {..., feedback: fb}
  → loop_idx = 1, prev_out = {}, step_idx = 0

Loop 1:
  step 0: factor_propose(prev_out={})                → fresh hypothesis
  ...dst
"""

import datetime
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable  # noqa: F401 — dipakai di LoopMeta.__new__ check

from tqdm.auto import tqdm

from core.exception import CoderError
from log import logger
import threading

# Pipeline monitor (safe import — no-op jika debug/ tidak tersedia)
try:
    from debug import get_monitor as _get_monitor
    _HAS_MONITOR = True
except ImportError:
    _HAS_MONITOR = False
    _get_monitor = lambda: None

class LoopMeta(type):
    @staticmethod
    def _get_steps(bases):
        """
        Recursively get all the `steps` from the base classes and combine them into a single list.

        Args:
            bases (tuple): A tuple of base classes.

        Returns:
            List[Callable]: A list of steps combined from all base classes.
        """
        
        # import pdb; pdb.set_trace()
        steps = []
        for base in bases:
            # misal: LoopBase punya steps [], AlphaAgentLoop punya method
            # factor_propose, factor_construct, dll → semua ditambahkan
            for step in LoopMeta._get_steps(base.__bases__) + getattr(base, "steps", []):
                if step not in steps:
                    steps.append(step)
        return steps

    #* dipanggil saat Python membuat class baru (bukan instance, tapi class itu sendiri)
    def __new__(cls, clsname, bases, attrs):
        """
        Create a new class with combined steps from base classes and current class.

        Args:
            clsname (str): Name of the new class.
            bases (tuple): Base classes.
            attrs (dict): Attributes of the new class.

        Returns:
            LoopMeta: A new instance of LoopMeta.
        """
        steps = LoopMeta._get_steps(bases)  # all the base classes of parents
        for name, attr in attrs.items():
            # Skip methods whose names start with underscore (private/protected)
            if not name.startswith("_") and isinstance(attr, Callable):
                if name not in steps:
                    # NOTE: if we override the step in the subclass
                    # Then it is not the new step. So we skip it.
                    steps.append(name)
        attrs["steps"] = steps
        return super().__new__(cls, clsname, bases, attrs)


@dataclass
class LoopTrace:
    """
    kapan sebuah step mulai dan selesai.
    """
    start: datetime.datetime  # the start time of the trace
    end: datetime.datetime  # the end time of the trace
    # TODO: more information about the trace


class LoopBase:
    steps: list[str]  # a list of step method-names (di-set oleh LoopMeta)
    loop_trace: dict[int, list[LoopTrace]] # riwayat timing per loop

    skip_loop_error: tuple[Exception] = field(
        default_factory=tuple
    )  # you can define a list of error that will skip current loop

    # Subclass bisa override tuple ini untuk exclude atribut non-picklable
    # dari session dump.  Contoh: GPU tensor (_pipeline_kv), loaded model
    # (llm_backend), dan objek proposal yang mengandung KV-cache state.
    _non_picklable_attrs: tuple[str, ...] = ()

    def __init__(self):
        self.loop_idx = 0  # current loop index
        self.step_idx = 0  # the index of next step to be run
        self.loop_prev_out = {}  # the step results of current loop
        self.loop_trace = defaultdict(list[LoopTrace])  # the key is the number of loop
        self.session_folder = logger.log_trace_path / "__session__" # folder untuk menyimpan session(pickle)

    # ── Pickle safety ────────────────────────────────────────────────
    # LoopBase.dump() serializes the ENTIRE object via pickle.
    # Subclasses yang menyimpan GPU tensor (KV-cache) atau loaded model
    # harus mendaftarkan nama atribut tersebut di _non_picklable_attrs
    # agar tidak ikut di-serialize (GPU tensor tidak bisa di-pickle,
    # dan model terlalu besar).
    #
    # Saat di-load kembali (LoopBase.load), atribut tersebut jadi None.
    # Caller harus re-inject llm_backend & past_kv setelah load.

    def __getstate__(self) -> dict:
        """Exclude GPU tensors dan heavy objects dari pickle."""
        state = self.__dict__.copy()
        for attr in self._non_picklable_attrs:
            state.pop(attr, None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state; atribut non-picklable di-set ke None."""
        self.__dict__.update(state)
        for attr in self._non_picklable_attrs:
            if attr not in self.__dict__:
                self.__dict__[attr] = None

    def run(self, step_n: int | None = None, stop_event: threading.Event = None):
        """

        Parameters
        ----------
        step_n : int | None
            How many steps to run;
            `None` indicates to run forever until error or KeyboardInterrupt
        """
        _mon = _get_monitor() if _HAS_MONITOR else None
        _loop_start_ts: dict[int, datetime.datetime] = {}

        with tqdm(total=len(self.steps), desc="Workflow Progress", unit="step") as pbar:
            while True:
                if step_n is not None:
                    if step_n <= 0:
                        break
                    step_n -= 1

                li, si = self.loop_idx, self.step_idx
                #* li = loop index, si = step index

                # Emit loop-start event saat memasuki step pertama dari sebuah loop
                if _mon is not None and si == 0 and li not in _loop_start_ts:
                    _loop_start_ts[li] = datetime.datetime.now(datetime.timezone.utc)
                    _mon.track_loop_start(loop_idx=li, total_steps=len(self.steps))

                start = datetime.datetime.now(datetime.timezone.utc)

                #* ambil nama step, misal: "factor_propose" (si=0), "factor_construct" (si=1)
                name = self.steps[si]

                #* ambil method dari instance, misal: self.factor_propose
                func = getattr(self, name)
                try:
                    self.loop_prev_out[name] = func(self.loop_prev_out)
                    #   PANGGIL step function!
                    #   input: dict output dari step sebelumnya
                    #   output: disimpan ke dict dengan key = nama step
                    #
                    #   Misal step "factor_construct" dipanggil:
                    #     func = self.factor_construct
                    #     self.loop_prev_out = {"factor_propose": idea_object}
                    #     func(self.loop_prev_out)  → akses prev_out["factor_propose"]
                    #     hasilnya disimpan: self.loop_prev_out["factor_construct"] = factor

                    # TODO: Fix the error logger.exception(f"Skip loop {li} due to {e}")
                except self.skip_loop_error as e:
                    logger.warning(f"Skip loop {li} due to {e}")
                    if _mon is not None:
                        _mon.track_loop_skipped(
                            loop_idx=li,
                            step_name=name,
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                    self.loop_idx += 1
                    self.step_idx = 0
                    _loop_start_ts.pop(li, None)
                    continue
                except CoderError as e:
                    logger.warning(f"Traceback loop {li} due to {e}")
                    if _mon is not None:
                        _mon.track_loop_traceback(
                            loop_idx=li,
                            step_name=name,
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                    self.step_idx = 0
                    continue

                end = datetime.datetime.now(datetime.timezone.utc)

                self.loop_trace[li].append(LoopTrace(start, end))

                # Update tqdm progress bar
                pbar.set_postfix(loop_index=li, step_index=si, step_name=name)
                pbar.update(1)

                # index increase and save session
                self.step_idx = (self.step_idx + 1) % len(self.steps)
                if self.step_idx == 0:  # reset to step 0 in next round
                    if _mon is not None:
                        loop_start = _loop_start_ts.pop(li, None)
                        duration_s = (
                            (end - loop_start).total_seconds()
                            if loop_start is not None else 0.0
                        )
                        _mon.track_loop_end(
                            loop_idx=li,
                            duration_s=duration_s,
                            completed_steps=len(self.steps),
                        )
                    self.loop_idx += 1
                    self.loop_prev_out = {}
                    pbar.reset()  # reset the progress bar for the next loop
                
                # save a snapshot after the session
                # path: __session__/0/0_factor_proposepath: __session__/0/0_factor_propose
                self.dump(self.session_folder / f"{li}" / f"{si}_{name}")  
                
                if stop_event is not None and stop_event.is_set():
                    # break
                    raise Exception("Mining stopped by user")

                
    def dump(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f) 
            #* serialize SELURUH object ke file binary
            #* state lengkap tersimpan: loop_idx, step_idx, loop_prev_out, loop_trace, dll

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        with path.open("rb") as f:
            session = pickle.load(f)
            #* load object dari pickle-session langsung punya semua state
            
        logger.set_trace_path(session.session_folder.parent)

        max_loop = max(session.loop_trace.keys())
        logger.storage.truncate(time=session.loop_trace[max_loop][-1].end)
        #* potong log: hanya simpan waktu terakhir session
        return session
