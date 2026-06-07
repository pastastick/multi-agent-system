"""
Model workflow with session control.
"""

import time
import pandas as pd
from typing import Any, Optional

from pipeline.settings import BaseFacSetting
from core.developer import Developer

# Lazy imports for KV-cache / latent pipeline (avoid hard dependency)
try:
    from llm.client import LocalLLMBackend, KVCache
    from llm._shared import kv_truncate
    _HAS_LOCAL_LLM = True
except ImportError:
    _HAS_LOCAL_LLM = False
    LocalLLMBackend = Any   # type: ignore[assignment,misc]
    KVCache = Any           # type: ignore[assignment,misc]
    kv_truncate = None      # type: ignore[assignment]

from core.proposal import (
    Hypothesis2Experiment,              #* ABC: convert hypothesis → experiment/factor
    HypothesisExperiment2Feedback,      #* ABC: generate feedback dari backtest
    HypothesisGen,                      #* ABC: generate hypothesis
    Trace,                              #* Simpan history trace (hypothesis, experiment, feedback) untuk tiap round
)
from core.scenario import Scenario      #* ABC: deskripsi skenario (market, data, interface)
from core.utils import import_class
from log import logger
from log.time import measure_time
from utils.workflow import LoopBase, LoopMeta   #* framework workflow
from core.exception import FactorEmptyError
import threading

# Pipeline monitor (safe import — no-op if unavailable)
try:
    from debug import get_monitor as _get_monitor
    _HAS_MONITOR = True
except ImportError:
    _HAS_MONITOR = False
    _get_monitor = lambda: None

import datetime
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from tqdm.auto import tqdm
from core.exception import CoderError
from log import logger
from contextlib import nullcontext as _nullcontext
from functools import wraps

# Decorator: check stop_event before invoking the function
def stop_event_check(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if STOP_EVENT is not None and STOP_EVENT.is_set():
            raise Exception("Operation stopped due to stop_event flag.")
        return func(self, *args, **kwargs)
    return wrapper


#* Metaclass adalah class yang membuat class lain. 
# LoopMeta otomatis mengumpulkan method publik dari sebuah class dan mendaftarkannya sebagai steps.

# SOTA tracker deterministik (per-proses; reset tiap run). Dipakai
# AlphaAgentLoop._decide_replace_sota untuk keputusan replace-best-result tanpa LLM.
# Konsisten dengan fresh_start TrajectoryPool: SOTA mulai kosong tiap proses.
_SOTA_BEST: dict[str, Any] = {"metric": None}


class AlphaAgentLoop(LoopBase, metaclass=LoopMeta):
    skip_loop_error = (FactorEmptyError,)

    # GPU tensor dan loaded model tidak bisa di-pickle.
    # LoopBase.__getstate__ akan exclude atribut-atribut ini dari session dump.
    # Setelah load, atribut ini jadi None — caller harus re-inject.
    _non_picklable_attrs = (
        "_pipeline_kv",
        "_coder_kv",
        "llm_backend",
        "hypothesis_generator",
        "factor_constructor",
        "coder",
        "summarizer",
        # ── new LatentMAS pipeline (latent_mas) ──
        "_front",        # FrontEndPipeline (holds backend + agents)
        "_front_out",    # last FrontEndOutput (GPU KV tensors)
        "_runlog",       # RunLogger (file handles)
        "_parent_trajectories",  # parent StrategyTrajectory objects (dibaca sbg teks)
    )
    
    @measure_time  #log berapa lama waktu yang dibutuhkan untuk inisialisasi loop
    def __init__(
        self,
        PROP_SETTING: BaseFacSetting,   # config: path ke semua class #! ditimpa dengan ALPHA_AGENT_FACTOR_PROP_SETTING
        potential_direction,            # arah eksplorasi
        stop_event: threading.Event,
        use_local: bool = True,
        strategy_suffix: str = "",      # suffix dari evolution 
        evolution_phase: str = "original",
        trajectory_id: str = "",
        parent_trajectory_ids: list = None,
        direction_id: int = 0,
        round_idx: int = 0,
        quality_gate_config: dict = None,
        external_context: Optional[str] = None,
        llm_backend: Optional["LocalLLMBackend"] = None,
        past_kv: Optional["KVCache"] = None,
        parent_trajectories: Optional[list] = None,
    ):
        with logger.tag("init"): # semua log di sini ditandai "init"
            self.use_local = use_local
            # Store initial direction for factor provenance
            self.potential_direction = potential_direction

            # Evolution-related attributes
            self.strategy_suffix = strategy_suffix
            self.evolution_phase = evolution_phase  # original / mutation / crossover
            self.trajectory_id = trajectory_id
            self.parent_trajectory_ids = parent_trajectory_ids or []
            self.direction_id = direction_id
            self.round_idx = round_idx  # 0=original, 1=mutation, 2=crossover, ...

            # Quality gate config
            self.quality_gate_config = quality_gate_config or {}

            # External agent context (text summary from MacroExternalAgent, etc.)
            self.external_context: Optional[str] = external_context

            # ── KV-cache / latent pipeline ───────────────────────────────
            # llm_backend : shared LocalLLMBackend instance (same model for
            #               all steps — required for KV-cache compatibility).
            # past_kv     : seed KV-cache from planning / external agents.
            #               Passed to factor_propose as initial context.
            self.llm_backend = llm_backend
            self._pipeline_kv = past_kv

            # For trajectory collection
            self._last_hypothesis = None
            self._last_experiment = None
            self._last_feedback = None
            # KV dari coder repair (disimpan factor_calculate, dibaca feedback)
            self._coder_kv = None

            logger.info(f"Initialized AlphaAgentLoop, backtest in {'local' if use_local else 'Docker'}")
            if potential_direction:
                logger.info(f"Initial direction: {potential_direction}")
            if evolution_phase != "original":
                logger.info(f"Evolution phase: {evolution_phase}, round: {round_idx}, trajectory_id: {trajectory_id}")
            if external_context:
                logger.info(f"External context attached ({len(external_context)} chars)")
            if llm_backend is not None:
                logger.info(
                    f"[LatentPipeline] LocalLLMBackend active, "
                    f"past_kv={'yes' if past_kv is not None else 'no'}"
                )
                
            #* consistency: apakah faktor konsisten dengan hipotesis?
            consistency_enabled = self.quality_gate_config.get("consistency_enabled", False)
            #* complexity: apakah faktor terlalu kompleks?
            complexity_enabled = self.quality_gate_config.get("complexity_enabled", True)
            #* redundancy: apakah faktor redundant dengan faktor lain yang sudah ada?
            redundancy_enabled = self.quality_gate_config.get("redundancy_enabled", True)
            
            logger.info(f"Quality gate: consistency={'on' if consistency_enabled else 'off'}, "
                       f"complexity={'on' if complexity_enabled else 'off'}, "
                       f"redundancy={'on' if redundancy_enabled else 'off'}")

            #* buat scenario
            scen: Scenario = import_class(PROP_SETTING.scen)(use_local=use_local)
            # PROP_SETTING.scen = "factors.experiment.QlibAlphaAgentScenario"
            #   import_class() memecah string ini:
            #     module_path = "factors.experiment"
            #     class_name = "QlibAlphaAgentScenario"
            #   lalu: importlib.import_module("factors.experiment")
            #   lalu: getattr(module, "QlibAlphaAgentScenario")
            #   lalu: QlibAlphaAgentScenario(use_local=True)
            
            logger.log_object(scen, tag="scenario")

            # Build effective_direction: base + strategy_suffix + external_context
            effective_direction = potential_direction
            if strategy_suffix:
                effective_direction = (potential_direction or "") + "\n" + strategy_suffix
            if external_context:
                effective_direction = (
                    (effective_direction or "")
                    + "\n\n[External Macro Context]\n"
                    + external_context
                )
            self._effective_direction = effective_direction or ""
            # Direction TANPA strategy_suffix text-operator lama. Dipakai sebagai
            # konteks `direction` untuk run_evolution (mutation/crossover judger);
            # arah revisi/fusi sebenarnya datang dari TEKS parent, bukan suffix lama.
            base_direction = potential_direction or ""
            if external_context:
                base_direction = (base_direction
                                  + "\n\n[External Macro Context]\n" + external_context)
            self._base_direction = base_direction
            # State threaded across the 5 LoopBase steps within one iteration.
            self._front_out = None       # FrontEndOutput dari step propose
            # feedback kini murni evaluatif (tak mengarang hipotesis) → selalu "".
            # Dipertahankan agar tanda tangan FrontEndPipeline.run tetap kompatibel.
            self._prior_feedback = ""
            self._last_factor_name = ""  # diisi _build_experiment, dipakai feedback block
            # Parent trajectories (objek StrategyTrajectory, BUKAN hanya id) untuk
            # _evolution_propose → _format_parents_text (dibaca sebagai TEKS:
            # hypothesis/expr/metrics/feedback; KV parent tidak dipakai lagi).
            self._parent_trajectories = parent_trajectories or []

            # ── KV-cache config dari settings ────────────────────────────
            # Baca per-step latent_steps dan temperature dari PROP_SETTING.
            # getattr() dengan fallback agar tetap kompatibel jika settings
            # belum punya field latent (misal BaseFacSetting).
            self._kv_max_tokens = getattr(PROP_SETTING, 'kv_max_tokens', 20480)  

            # ── Instantiate proposal classes ─────────────────────────────
            # When llm_backend is provided, use Latent variants with
            # KV-cache support.  Otherwise, use standard classes.
            if llm_backend is not None and _HAS_LOCAL_LLM:
                # ── NEW: LatentMAS pipeline (menggantikan factors.latent_proposal) ──
                # proposal→construct→consistency→judger→gate→repair dalam satu
                # FrontEndPipeline; feedback via agent. Substrat backtest/library
                # tetap dipakai ulang (lihat _build_experiment & feedback()).
                from latent_mas.runlog import get_run_logger
                from latent_mas.pipeline import FrontEndPipeline
                self._latent = True
                self._runlog = get_run_logger(
                    run_name=f"loop_{evolution_phase}_{round_idx}_{direction_id}"
                )
                # quality_gate=None + use_regulator=True → FactorRegulator PENUH
                # (complexity SL/PC/ER + redundansi alpha-zoo), bukan sekadar sintaks.
                self._front = FrontEndPipeline(
                    llm_backend, runlog=self._runlog, max_repair_attempts=3,
                )
                # Atribut path-standar di-set None agar pickle-exclusion & getattr aman.
                self.hypothesis_generator = None
                self.factor_constructor = None
                self.summarizer = None
                logger.info(
                    f"[LatentMAS] FrontEndPipeline active "
                    f"(kv_max_tokens={self._kv_max_tokens})"
                )
            else:
                self._latent = False
                self.hypothesis_generator: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen, effective_direction)
                
                #   "factors.proposal.AlphaAgentHypothesis2FactorExpression"
                #   convert hipotesis → list faktor (ekspresi matematika)
                self.factor_constructor: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)(
                    consistency_enabled=consistency_enabled
                )
                
                #   "factors.feedback.AlphaAgentQlibFactorHypothesisExperiment2Feedback"
                #   evaluasi hasil backtest → generate feedback
                self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
                
                # Inject llm_backend into standard classes (mereka pakai
                # self.llm_backend di _call_llm jika tersedia)
                if llm_backend is not None:
                    self.hypothesis_generator.llm_backend = llm_backend
                    self.factor_constructor.llm_backend = llm_backend
                    self.summarizer.llm_backend = llm_backend

            logger.log_object(self.hypothesis_generator, tag="hypothesis generator")
            logger.log_object(self.factor_constructor, tag="experiment generation")

            #   "factors.qlib_coder.QlibFactorParser"
            #   tugas: parse rumus faktor → kode Python yang bisa dieksekusi
            #   Saat latent pipeline aktif, coder menerima llm_backend
            #   agar LLM calls (expression fix, evaluator) menggunakan
            #   shared model dan bisa terima KV-cache dari construct step.
            _coder_kwargs = {}
            if llm_backend is not None and _HAS_LOCAL_LLM:
                _coder_kwargs["llm_backend"] = llm_backend
                _get_ls_c = getattr(PROP_SETTING, 'get_latent_steps_for', None)
                _get_temp_c = getattr(PROP_SETTING, 'get_temperature_for', None)
                if _get_ls_c:
                    _coder_kwargs["latent_steps"] = _get_ls_c("coder")
                if _get_temp_c:
                    _coder_kwargs["temperature"] = _get_temp_c("coder")
            self.coder: Developer = import_class(PROP_SETTING.coder)(scen, **_coder_kwargs)
            logger.log_object(self.coder, tag="coder")

            #   "factors.runner.QlibFactorRunner"
            #   tugas: jalankan kode, hitung faktor, backtest di Qlib
            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")

            logger.log_object(self.summarizer, tag="summarizer")
            self.trace = Trace(scen=scen) #* trace kosong akan diisi setiap loop di step feedback
            
            global STOP_EVENT
            STOP_EVENT = stop_event
            super().__init__()

    @classmethod
    def load(cls, path, use_local: bool = True): #* resume session dari disk
        """Load existing session."""
        instance = super().load(path)
        instance.use_local = use_local
        logger.info(f"Loaded AlphaAgentLoop, backtest in {'local' if use_local else 'Docker'}")
        return instance

    @measure_time
    @stop_event_check
    def factor_propose(self, prev_out: dict[str, Any]):
        """Propose hypothesis as the basis for factor construction."""
        _mon = _get_monitor() if _HAS_MONITOR else None

        # ── NEW LatentMAS path: phase menentukan generator ──────────────────
        #   original → FrontEndPipeline.run (proposal→construct→consistency→judger)
        #   mutation → run_evolution("mutation")  — judger revisi 1 target (seed None)
        #   crossover→ run_evolution("crossover") — judger recombine k parent (seed None)
        if getattr(self, "_latent", False):
            phase = getattr(self, "evolution_phase", "original")
            parents = getattr(self, "_parent_trajectories", []) or []
            with logger.tag("r"):
                if phase == "mutation" and parents:
                    front = self._evolution_propose(parents, crossover=False)
                elif phase == "crossover" and parents:
                    front = self._evolution_propose(parents, crossover=True)
                else:
                    front = self._front.run(
                        direction=self._effective_direction,
                        seed_kv=self._pipeline_kv,
                        prior_feedback=self._prior_feedback,
                    )
            self._front_out = front
            self._last_hypothesis = front.hypothesis
            logger.info(
                f"[LatentMAS] phase={phase}: hypo_len={len(front.hypothesis)}, "
                f"n_expr={len(front.expressions)}, exprs={front.expressions!r}, "
                f"repaired={front.repaired}, gate_error={front.gate_error or 'none'}"
            )
            if not front.expressions:
                raise FactorEmptyError("Front-end produced no expression")
            return front.hypothesis

        with logger.tag("r"):
            # ── KV-cache: inject seed from planning/external agents ──
            self.hypothesis_generator.set_past_kv(self._pipeline_kv)

            if _mon:
                _mon.set_context(loop_idx=getattr(self, 'loop_idx', 0),
                                 direction_id=self.direction_id,
                                 phase=self.evolution_phase,
                                 round_idx=self.round_idx)

            with _mon.track_step("factor_propose", has_kv_input=self._pipeline_kv is not None) if _mon else _nullcontext():
                idea = self.hypothesis_generator.gen(self.trace)

            logger.log_object(idea, tag="hypothesis generation")
            self._last_hypothesis = idea

            if _mon:
                _mon.analyze_llm_output(str(idea), caller="propose")

        return idea

    @measure_time
    @stop_event_check
    def factor_construct(self, prev_out: dict[str, Any]):
        """Construct multiple factors from the hypothesis."""
        _mon = _get_monitor() if _HAS_MONITOR else None

        # ── NEW LatentMAS path: bridge (hypothesis, N expressions) → QlibFactorExperiment
        if getattr(self, "_latent", False):
            with logger.tag("r"):
                front = self._front_out
                exp = self._build_experiment(front.hypothesis, front.expressions)
            logger.log_object(exp.sub_tasks, tag="experiment generation")
            return exp

        with logger.tag("r"):
            self.factor_constructor.set_past_kv(self.hypothesis_generator.last_kv)

            with _mon.track_step("factor_construct") if _mon else _nullcontext():
                factor = self.factor_constructor.convert(prev_out["factor_propose"], self.trace)

            logger.log_object(factor.sub_tasks, tag="experiment generation")

            if _mon and factor.sub_tasks:
                _mon.analyze_llm_output(
                    "\n".join(str(t) for t in factor.sub_tasks),
                    caller="construct",
                )

        return factor

    @measure_time
    @stop_event_check
    def factor_calculate(self, prev_out: dict[str, Any]):  #* tulis kode dari rumus faktor
        """Compute factor values from factor expressions."""
        _mon = _get_monitor() if _HAS_MONITOR else None

        # ── NEW LatentMAS path: kode + workspace sudah dirender di _build_experiment
        # (judger+gate+repair menggantikan coder LLM). Tidak ada yang dikerjakan di sini.
        if getattr(self, "_latent", False):
            return prev_out["factor_construct"]

        with logger.tag("d"):  # develop

            construct_kv = getattr(self.factor_constructor, 'last_kv', None)

            with _mon.track_step("factor_calculate") if _mon else _nullcontext():
                factor = self.coder.develop(prev_out["factor_construct"], past_kv=construct_kv)

            logger.log_object(factor.sub_workspace_list, tag="coder result")

            # Simpan coder_kv untuk dipakai feedback step.
            # coder_kv ada jika LLM repair dipanggil (expression gagal pertama kali).
            # None jika template langsung OK — feedback fallback ke construct_kv.
            coder_kv = getattr(self.coder, 'last_kv', None)
            self._coder_kv = coder_kv
            if construct_kv is not None:
                logger.info(
                    f"[LatentPipeline] Coder KV chain: "
                    f"construct_kv=yes → coder_kv={'yes' if coder_kv is not None else 'no (template OK, no repair)'}"
                )

        return factor
    

    @measure_time
    @stop_event_check
    def factor_backtest(self, prev_out: dict[str, Any]):  #* jalankan backtest
        """Run backtest for factors."""
        _mon = _get_monitor() if _HAS_MONITOR else None
        with logger.tag("ef"):  # evaluate and feedback
            logger.info(f"Start factor backtest (Local: {self.use_local})")

            with _mon.track_step("factor_backtest") if _mon else _nullcontext():
                exp = self.runner.develop(prev_out["factor_calculate"], use_local=self.use_local)

            if exp is None:
                logger.error(f"Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")

            logger.log_object(exp, tag="runner result")
            self._last_experiment = exp
        return exp

    @measure_time
    @stop_event_check
    def feedback(self, prev_out: dict[str, Any]):
        _mon = _get_monitor() if _HAS_MONITOR else None

        if getattr(self, "_latent", False):
            # ── NEW LatentMAS path: feedback agent membaca CLONE kv_consist
            # (anti-bias kv_judger — lihat desain). Mengembalikan dict feedback.
            feedback = self._run_latent_feedback(prev_out)
        else:
            # Feedback menerima KV terbaik yang tersedia:
            #   coder_kv  — jika LLM repair dipanggil: mengandung construct context
            #               + repair reasoning. Lebih informatif karena feedback
            #               seharusnya "tahu" bagaimana expression diperbaiki.
            #   construct_kv — fallback jika tidak ada repair (template langsung OK):
            #               mengandung propose context + factor expressions.
            construct_kv = getattr(self.factor_constructor, 'last_kv', None)
            coder_kv = getattr(self, '_coder_kv', None)
            feedback_input_kv = coder_kv if coder_kv is not None else construct_kv
            self.summarizer.set_past_kv(feedback_input_kv)
            if feedback_input_kv is not None:
                _kv_src = "coder (repair)" if coder_kv is not None else "construct (no repair)"
                logger.info(f"[LatentPipeline] Feedback receives KV from {_kv_src}")

            with _mon.track_step("feedback") if _mon else _nullcontext():
                feedback = self.summarizer.generate_feedback(prev_out["factor_backtest"], prev_out["factor_propose"], self.trace)

        with logger.tag("ef"):
            logger.log_object(feedback, tag="feedback")

        if _mon:
            _mon.analyze_llm_output(str(feedback), caller="feedback")

        self.trace.hist.append((prev_out["factor_propose"], prev_out["factor_backtest"], feedback))
        self._last_feedback = feedback

        # Reset pipeline KV after each iteration — propose starts fresh.
        # Context carryover is handled via self.trace.hist (text), not KV.
        # Chaining feedback_kv → propose biases the model toward feedback
        # output format and accumulates stale context; text trace is enough.
        # Mutation/crossover seeds are injected at __init__ time, consumed in
        # iteration 1 only — resetting here is safe for subsequent iterations.
        self._pipeline_kv = None
        self._coder_kv = None  # reset per-iteration, set ulang di factor_calculate berikutnya

        # Latent: feedback kini MURNI EVALUATIF (tak mengarang New Hypothesis) —
        # generasi arah berikutnya adalah tugas mutation/crossover. Maka tidak ada
        # carryover hipotesis ke propose; biarkan kosong.
        if getattr(self, "_latent", False):
            self._prior_feedback = ""

        #* Auto-save factors to unified factor library
        try:
            import os
            from pathlib import Path
            from factors.library import FactorLibraryManager
            
            # Project root: loop.py -> pipeline/ -> quantaalpha/ -> project_root/
            project_root = Path(__file__).resolve().parent.parent.parent

            experiment_id = "unknown"
            if hasattr(self, 'session_folder') and self.session_folder:
                parts = Path(self.session_folder).parts
                for part in parts:
                    if part.startswith("202") and len(part) > 10:
                        experiment_id = part
                        break

            round_number = self.round_idx

            hypothesis_text = None
            if prev_out.get("factor_propose"):
                hypothesis_text = str(prev_out["factor_propose"])

            planning_direction = getattr(self, 'potential_direction', None)
            user_initial_direction = getattr(self, 'user_initial_direction', None)

            evolution_phase = getattr(self, 'evolution_phase', 'original')
            trajectory_id = getattr(self, 'trajectory_id', '')
            parent_trajectory_ids = getattr(self, 'parent_trajectory_ids', [])

            #* Factor library filename can be customized via env FACTOR_LIBRARY_SUFFIX
            library_suffix = os.environ.get('FACTOR_LIBRARY_SUFFIX', '')
            if library_suffix:
                library_filename = f"all_factors_library_{library_suffix}.json"
            else:
                library_filename = "all_factors_library.json"
                
            factorlib_dir = project_root / "data" / "factorlib"
            factorlib_dir.mkdir(parents=True, exist_ok=True)
            library_path = factorlib_dir / library_filename
            
            manager = FactorLibraryManager(str(library_path))
            manager.add_factors_from_experiment(
                experiment=prev_out["factor_backtest"],
                experiment_id=experiment_id,
                round_number=round_number,
                hypothesis=hypothesis_text,
                feedback=feedback,
                initial_direction=planning_direction,
                user_initial_direction=user_initial_direction,
                planning_direction=planning_direction,
                evolution_phase=evolution_phase,
                trajectory_id=trajectory_id,
                parent_trajectory_ids=parent_trajectory_ids,
            )
            logger.info(f"Saved factors to library: {library_path} (phase={evolution_phase})")
        except Exception as e:
            logger.warning(f"Failed to save factors to library: {e}")
    
    def _get_trajectory_data(self) -> dict[str, Any]:
        """
        Get trajectory data for the current round (used by evolution controller).
        Method name is prefixed with underscore so the workflow system does not treat it as a step.
        Returns:
            Dict with hypothesis, experiment, feedback, etc.
        """
        # Extract hypothesis_embedding from propose step's hidden_last
        hypothesis_embedding = None
        last_result = getattr(self.hypothesis_generator, "last_result", None)
        if last_result is not None and getattr(last_result, "hidden_last", None) is not None:
            try:
                # hidden_last: [1, d] → flatten to list[float]
                hypothesis_embedding = last_result.hidden_last.squeeze(0).float().cpu().tolist()
            except Exception:
                pass

        return {
            "hypothesis": self._last_hypothesis,
            "experiment": self._last_experiment,
            "feedback": self._last_feedback,
            "direction_id": self.direction_id,
            "evolution_phase": self.evolution_phase,
            "trajectory_id": self.trajectory_id,
            "parent_trajectory_ids": self.parent_trajectory_ids,
            "loop_idx": self.loop_idx,
            "round_idx": self.round_idx,
            "hypothesis_embedding": hypothesis_embedding,
            # KV-cache untuk trajectory.kv_cache.
            # Latent: None — evolution (mutation/crossover) sekarang JUDGER-ONLY dan
            #   membaca parent sebagai TEKS (lihat _format_parents_text), jadi KV
            #   parent tak lagi dibutuhkan. Mewariskannya hanya akan mem-pin KV
            #   GPU (ratusan MB) per trajectory di pool → buang. Ini sekaligus
            #   memutus carry-over KV antar-generasi (akar over-KV & salin-loop).
            # Standar (non-latent): _pipeline_kv lama tetap diteruskan.
            "pipeline_kv": (
                None
                if getattr(self, "_latent", False)
                else getattr(self, "_pipeline_kv", None)
            ),
        }

    # ── NEW LatentMAS helpers ────────────────────────────────────────────────

    def _build_experiment(self, hypothesis: str, expressions: list):
        """Bridge: (hypothesis, N ekspresi) → QlibFactorExperiment siap-backtest.

        N ekspresi lolos regulator → N FactorTask/workspace → runner menggabungkan
        jadi SATU model LightGBM (multi-faktor), satu hasil backtest. Mirror
        proposal._build_experiment_from_dict (dedup intra-batch + terhadap history).
        Substrat (rdagent/qlib) dipakai ulang, bukan ditulis ulang.
        """
        import re as _re
        from factors.coder.factor import FactorTask, FactorFBWorkspace
        from factors.experiment import QlibFactorExperiment
        from factors.coder.evolving_strategy import code_template

        base = _re.sub(r"[^a-zA-Z0-9]+", "_", (hypothesis or "").strip())[:32].strip("_")
        base = base or f"latent_factor_{getattr(self, 'loop_idx', 0)}"

        based = (
            [QlibFactorExperiment(sub_tasks=[])]
            + [h[1] for h in self.trace.hist if h[2]]
        )
        # nama faktor yang sudah ada di history → untuk dedup
        existing = {
            getattr(st, "factor_name", None)
            for be in based for st in be.sub_tasks
        }

        tasks, workspaces, names, seen = [], [], [], set()
        single = len([e for e in expressions if e and e.strip()]) == 1
        for i, expr in enumerate(expressions):
            if not expr or not expr.strip():
                continue
            key = expr.replace(" ", "").lower()
            if key in seen:
                continue
            seen.add(key)
            fname = base if single else f"{base}_{i}"
            while fname in existing:    # hindari tabrakan nama dengan history
                fname = f"{fname}_x"
            existing.add(fname)
            task = FactorTask(
                factor_name=fname,
                factor_description=hypothesis,
                factor_formulation=expr,
                factor_expression=expr,
                variables={},
            )
            code = code_template.render(expression=expr, factor_name=fname)
            ws = FactorFBWorkspace(target_task=task)
            ws.inject_code(**{"factor.py": code})
            tasks.append(task)
            workspaces.append(ws)
            names.append(fname)

        self._last_factor_names = names
        self._last_factor_name = names[0] if names else base  # back-compat
        exp = QlibFactorExperiment(tasks)
        exp.based_experiments = based
        exp.sub_workspace_list = workspaces
        return exp

    @staticmethod
    def _parent_expr(parent) -> str:
        """Ambil ekspresi faktor pertama dari parent StrategyTrajectory."""
        factors = getattr(parent, "factors", None) or []
        if factors and isinstance(factors[0], dict):
            return factors[0].get("expression", "") or ""
        return ""

    @staticmethod
    def _parent_metric(parent) -> float:
        """Metric primer parent untuk pengurutan crossover (None → -inf)."""
        getter = getattr(parent, "get_primary_metric", None)
        if callable(getter):
            try:
                m = getter()
                return float(m) if m is not None else float("-inf")
            except Exception:
                return float("-inf")
        return float("-inf")

    def _evolution_propose(self, parents: list, *, crossover: bool):
        """Evolution JUDGER-ONLY → FrontEndPipeline.run_evolution (seed_kv=None).

        mutation  : revisi 1 target (eksploitasi).
        crossover : recombination k parent (eksplorasi).
        Materi parent dikirim sebagai TEKS (lihat _format_parents_text); KV parent
        TIDAK diwariskan → KV per ronde terbatas, tak ada salin-loop. Tidak ada lagi
        reflection / kv_concat / re-entry front-end.
        """
        if crossover:
            # urut ASC by metric → parent terbaik di posisi TERAKHIR (recency dalam teks)
            ordered = sorted(parents, key=self._parent_metric)
            front = self._front.run_evolution(
                kind="crossover",
                parent_text=self._format_parents_text(ordered),
                n_parents=len(ordered),
                direction=self._base_direction,
            )
            label = f"crossover(n={len(ordered)})"
        else:
            front = self._front.run_evolution(
                kind="mutation",
                parent_text=self._format_parents_text([parents[0]]),
                n_parents=1,
                direction=self._base_direction,
            )
            label = "mutation"
        logger.info(f"[LatentMAS] {label} → n_expr={len(front.expressions)} "
                    f"exprs={front.expressions!r} repaired={front.repaired} "
                    f"gate_error={front.gate_error or 'none'}")
        return front

    def _format_parents_text(self, parents: list) -> str:
        """Rangkai parent StrategyTrajectory → TEKS untuk mutation/crossover judger.

        Menyertakan hipotesis, ekspresi PENUH (tak dipangkas seperti
        to_summary_text yang memotong di 100 char), metrik backtest, dan feedback.
        Diberi label [Parent i] saat >1 agar crossover bisa membedakan sumber.
        Inilah SATU-SATUNYA jalur transfer parent→evolution sekarang (KV parent
        tidak lagi dipakai)."""
        blocks: list[str] = []
        single = len(parents) == 1
        for i, p in enumerate(parents, 1):
            hypo = (getattr(p, "hypothesis", "") or "").strip()
            expr_lines: list[str] = []
            for f in (getattr(p, "factors", None) or [])[:5]:
                if isinstance(f, dict) and (f.get("expression") or "").strip():
                    nm = (f.get("name") or "").strip()
                    ex = f["expression"].strip()
                    expr_lines.append(f"  - {nm}: {ex}" if nm else f"  - {ex}")
            if not expr_lines:
                ex = self._parent_expr(p)
                if ex:
                    expr_lines.append(f"  - {ex}")
            metrics = getattr(p, "backtest_metrics", None) or {}
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
                if isinstance(v, (int, float))
            )
            fb = str(getattr(p, "feedback", "") or "").strip()
            if len(fb) > 400:
                fb = fb[:400] + "…"
            parts: list[str] = []
            if hypo:        parts.append(f"Hypothesis: {hypo}")
            if expr_lines:  parts.append("Expression(s):\n" + "\n".join(expr_lines))
            if metric_str:  parts.append(f"Backtest: {metric_str}")
            if fb:          parts.append(f"Feedback: {fb}")
            block = "\n".join(parts) or "(empty parent)"
            blocks.append(block if single else f"[Parent {i}]\n{block}")
        return "\n\n".join(blocks)

    def _run_latent_feedback(self, prev_out: dict[str, Any]) -> dict:
        """Feedback EVALUATIF (support/refute) via agent latent_mas.

        Berbeda dari QuantaAlpha asli: feedback TIDAK lagi mengarang "New
        Hypothesis" — generasi arah berikutnya adalah tugas mutation (revisi 1
        target) & crossover (recombination), keduanya JUDGER-ONLY membaca parent
        sebagai teks. Dua keputusan konsekuensial DICABUT dari LLM menjadi
        deterministik dan disuntik sebagai input:
          - complexity audit  → ComplexityChecker
          - replace-best-result → aturan metrik (_decide_replace_sota)

        Seed = kv_final (kv_repair bila repair jalan, else kv_judger) → feedback
        koheren dengan ekspresi yang benar-benar dijalankan. KV feedback (terminal)
        TIDAK diwariskan ke ronde evolution (evolution baca teks parent, bukan KV).
        """
        from llm._shared import robust_json_parse
        from latent_mas import kv_ops

        front = self._front_out
        exp = prev_out["factor_backtest"]

        # ── DETERMINISTIK (tanpa LLM): complexity audit + replace-SOTA ──────────
        # Multi-ekspresi: audit semua, metrik dari backtest model LightGBM gabungan.
        complexity = self._complexity_audit_multi(front.expressions)
        metrics = self._extract_metrics_safe(exp)
        replace_flag, sota_note = self._decide_replace_sota(metrics, bool(complexity))

        # factor_block: daftar SEMUA ekspresi lolos (jadi referensi bentuk faktor
        # yang masuk model gabungan untuk hipotesis ini).
        names = getattr(self, "_last_factor_names", None) or [self._last_factor_name]
        exprs = front.expressions or [front.expression]
        factor_lines = [f"- {n}: {e}" for n, e in zip(names, exprs)]
        factor_block = (
            f"Hypothesis: {front.hypothesis}\n"
            f"Factors ({len(exprs)}) feeding the combined model:\n"
            + "\n".join(factor_lines)
        )
        if complexity:
            factor_block += f"\n  {complexity}"

        res = getattr(exp, "result", None)
        if res is None:
            backtest_results = "backtest produced no result (factor may have failed)"
        elif hasattr(res, "to_string"):
            backtest_results = res.to_string()
        else:
            backtest_results = str(res)

        # Keputusan replace + audit disuntik sebagai konteks (bukan untuk diputuskan LLM).
        decision_block = (
            f"Replace-best-result (DETERMINISTIC): "
            f"{'REPLACE' if replace_flag else 'KEEP'} — {sota_note}"
        )

        fb_agent = self._front.agents["feedback"]
        with _get_monitor().track_step("feedback") if (_HAS_MONITOR and _get_monitor()) else _nullcontext():
            r = fb_agent.run(
                past_kv=kv_ops.kv_deepcopy(front.kv_final),
                hypothesis_text=front.hypothesis,
                factor_block=factor_block,
                backtest_results=backtest_results,
                sota_block=decision_block,
            )
        # KV feedback (terminal) sengaja TIDAK disimpan/diwariskan: evolution kini
        # judger-only membaca parent sebagai teks, jadi membiarkan r.kv_cache di-GC
        # menghindari pin KV GPU yang sia-sia. _get_trajectory_data → pipeline_kv=None.

        try:
            fb = robust_json_parse(r.text) if r.text else {}
        except Exception:
            fb = {"Observations": r.text or ""}
        if not isinstance(fb, dict):
            fb = {"Observations": str(fb)}
        # Keputusan replace bersifat DETERMINISTIK → override apa pun dari LLM.
        fb["Replace Best Result"] = "yes" if replace_flag else "no"
        return fb

    # ── Audit deterministik (dicabut dari feedback LLM) ──────────────────────
    @staticmethod
    def _complexity_audit(expression: str) -> str:
        """Verdict complexity dari ComplexityChecker (deterministik). '' bila lolos."""
        try:
            from factors.regulator.consistency_checker import ComplexityChecker
            ok, msg = ComplexityChecker().check(expression)
            return "" if ok else f"COMPLEXITY WARNING: {msg}"
        except Exception:
            return ""

    @classmethod
    def _complexity_audit_multi(cls, expressions: list) -> str:
        """Audit complexity untuk N ekspresi → gabung peringatan (jarang muncul
        karena regulator-gate sudah menolak yang terlalu kompleks lebih awal)."""
        warns = []
        for e in (expressions or []):
            w = cls._complexity_audit(e)
            if w:
                warns.append(f"{e}: {w}")
        return " | ".join(warns)

    @staticmethod
    def _extract_metrics_safe(exp) -> dict[str, float]:
        """Ambil annualized_return / RankIC / IC dari exp.result (Series/DataFrame)."""
        res = getattr(exp, "result", None)
        out: dict[str, float] = {}
        if res is None:
            return out
        try:
            import pandas as pd
            wanted = {
                "annualized_return": [
                    "1day.excess_return_with_cost.annualized_return",
                    "1day.excess_return_without_cost.annualized_return",
                    "annualized_return",
                ],
                "RankIC": ["RankIC", "Rank IC", "rank_ic"],
                "IC": ["IC", "ic"],
            }
            idx = getattr(res, "index", None)
            for key, names in wanted.items():
                for n in names:
                    if idx is not None and n in idx:
                        v = res.loc[n]
                        if hasattr(v, "iloc"):
                            v = v.iloc[0]
                        if pd.notna(v):
                            out[key] = float(v)
                        break
        except Exception:
            pass
        return out

    def _decide_replace_sota(self, metrics: dict[str, float],
                             complexity_warning: bool) -> "tuple[bool, str]":
        """Keputusan deterministik 'ganti SOTA?'.

        Aturan (sesuai prioritas paper): primary = annualized_return, fallback
        RankIC. Bila ada complexity warning → JANGAN ganti (risiko overfit),
        sesuai prompt feedback QuantaAlpha. SOTA dilacak per-proses (in-memory).
        """
        cur = metrics.get("annualized_return")
        name = "annualized_return"
        if cur is None:
            cur = metrics.get("RankIC")
            name = "RankIC"
        if cur is None:
            return False, "no comparable metric → keep SOTA"
        if complexity_warning:
            return False, (f"{name}={cur:.4f} but complexity warning "
                           f"→ keep SOTA (overfit risk)")
        best = _SOTA_BEST.get("metric")
        if best is None or cur > best:
            _SOTA_BEST["metric"] = cur
            prev = "none" if best is None else f"{best:.4f}"
            return True, f"{name}={cur:.4f} > SOTA={prev} → replace"
        return False, f"{name}={cur:.4f} <= SOTA={best:.4f} → keep"




class BacktestLoop(LoopBase, metaclass=LoopMeta):
    skip_loop_error = (FactorEmptyError,)
    @measure_time
    def __init__(self, PROP_SETTING: BaseFacSetting, factor_path=None):
        with logger.tag("init"):

            self.factor_path = factor_path

            scen: Scenario = import_class(PROP_SETTING.scen)()
            logger.log_object(scen, tag="scenario")

            self.hypothesis_generator: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)
            logger.log_object(self.hypothesis_generator, tag="hypothesis generator")

            self.factor_constructor: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)(factor_path=factor_path)
            logger.log_object(self.factor_constructor, tag="experiment generation")

            self.coder: Developer = import_class(PROP_SETTING.coder)(scen, with_feedback=False, with_knowledge=False, knowledge_self_gen=False)
            logger.log_object(self.coder, tag="coder")
            
            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")

            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            logger.log_object(self.summarizer, tag="summarizer")
            self.trace = Trace(scen=scen)
            super().__init__()

    def factor_propose(self, prev_out: dict[str, Any]):
        """
        Market hypothesis on which factors are built
        """
        with logger.tag("r"):  
            idea = self.hypothesis_generator.gen(self.trace)
            logger.log_object(idea, tag="hypothesis generation")
        return idea
        

    @measure_time
    def factor_construct(self, prev_out: dict[str, Any]):
        """
        Construct a variety of factors that depend on the hypothesis
        """
        with logger.tag("r"): 
            factor = self.factor_constructor.convert(prev_out["factor_propose"], self.trace)
            logger.log_object(factor.sub_tasks, tag="experiment generation")
        return factor

    @measure_time
    def factor_calculate(self, prev_out: dict[str, Any]):
        """
        Debug factors and calculate their values
        """
        with logger.tag("d"):  # develop
            factor = self.coder.develop(prev_out["factor_construct"])
            logger.log_object(factor.sub_workspace_list, tag="coder result")
        return factor
    

    @measure_time
    def factor_backtest(self, prev_out: dict[str, Any]):
        """
        Conduct Backtesting
        """
        with logger.tag("ef"):  # evaluate and feedback
            exp = self.runner.develop(prev_out["factor_calculate"])
            if exp is None:
                logger.error(f"Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
            logger.log_object(exp, tag="runner result")
        return exp

    @measure_time
    def stop(self, prev_out: dict[str, Any]):
        exit(0)
