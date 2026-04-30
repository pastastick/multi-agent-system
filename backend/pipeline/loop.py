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

class AlphaAgentLoop(LoopBase, metaclass=LoopMeta):
    skip_loop_error = (FactorEmptyError,)

    # GPU tensor dan loaded model tidak bisa di-pickle.
    # LoopBase.__getstate__ akan exclude atribut-atribut ini dari session dump.
    # Setelah load, atribut ini jadi None — caller harus re-inject.
    _non_picklable_attrs = (
        "_pipeline_kv",
        "llm_backend",
        "hypothesis_generator",
        "factor_constructor",
        "coder",
        "summarizer",
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
            consistency_enabled = self.quality_gate_config.get("consistency_enabled", True)
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

            # ── KV-cache config dari settings ────────────────────────────
            # Baca per-step latent_steps dan temperature dari PROP_SETTING.
            # getattr() dengan fallback agar tetap kompatibel jika settings
            # belum punya field latent (misal BaseFacSetting).
            self._kv_max_tokens = getattr(PROP_SETTING, 'kv_max_tokens', 2048)  

            # ── Instantiate proposal classes ─────────────────────────────
            # When llm_backend is provided, use Latent variants with
            # KV-cache support.  Otherwise, use standard classes.
            if llm_backend is not None and _HAS_LOCAL_LLM:
                from factors.latent_proposal import (
                    LatentHypothesisGen,
                    LatentHypothesis2Experiment,
                    LatentFeedback,
                )
                # Per-step config dari settings (None = pakai default engine)
                _get_ls = getattr(PROP_SETTING, 'get_latent_steps_for', None)
                _get_temp = getattr(PROP_SETTING, 'get_temperature_for', None)

                self.hypothesis_generator = LatentHypothesisGen(
                    scen, effective_direction, llm_backend=llm_backend,
                    latent_steps=_get_ls("propose") if _get_ls else None,
                    temperature=_get_temp("propose") if _get_temp else None,
                )
                self.factor_constructor = LatentHypothesis2Experiment(
                    consistency_enabled=consistency_enabled,
                    llm_backend=llm_backend,
                    latent_steps=_get_ls("construct") if _get_ls else None,
                    # Construct: temperature rendah → formula lebih presisi.
                    # Output di-parse oleh QlibFactorParser menjadi Python code.
                    temperature=_get_temp("construct") if _get_temp else None,
                    # Guided JSON decoding — struktur output dipaksa valid
                    # via lm-format-enforcer (prefix_allowed_tokens_fn).
                    # Default True: model kecil butuh constraint struktural.
                    guided_decoding=getattr(
                        PROP_SETTING, "guided_construct_enabled", True,
                    ),
                )
                self.summarizer = LatentFeedback(
                    scen, llm_backend=llm_backend,
                    latent_steps=_get_ls("feedback") if _get_ls else None,
                    temperature=_get_temp("feedback") if _get_temp else None,
                )
                logger.info(
                    f"[LatentPipeline] Using Latent proposal classes with KV-cache chaining "
                    f"(kv_max_tokens={self._kv_max_tokens})"
                )
            else:
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
        with logger.tag("d"):  # develop

            construct_kv = getattr(self.factor_constructor, 'last_kv', None)

            with _mon.track_step("factor_calculate") if _mon else _nullcontext():
                factor = self.coder.develop(prev_out["factor_construct"], past_kv=construct_kv)

            logger.log_object(factor.sub_workspace_list, tag="coder result")

            if construct_kv is not None:
                coder_kv = getattr(self.coder, 'last_kv', None)
                logger.info(
                    f"[LatentPipeline] Coder KV chain: "
                    f"construct_kv={'yes'} → coder_kv={'yes' if coder_kv is not None else 'no'}"
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

        coder_kv = getattr(self.coder, 'last_kv', None)
        construct_kv = getattr(self.factor_constructor, 'last_kv', None)
        feedback_input_kv = coder_kv if coder_kv is not None else construct_kv
        self.summarizer.set_past_kv(feedback_input_kv)
        if feedback_input_kv is not None:
            source = "coder" if coder_kv is not None else "construct"
            logger.info(f"[LatentPipeline] Feedback receives KV from {source} step")

        with _mon.track_step("feedback") if _mon else _nullcontext():
            feedback = self.summarizer.generate_feedback(prev_out["factor_backtest"], prev_out["factor_propose"], self.trace)

        with logger.tag("ef"):
            logger.log_object(feedback, tag="feedback")

        if _mon:
            _mon.analyze_llm_output(str(feedback), caller="feedback")

        self.trace.hist.append((prev_out["factor_propose"], prev_out["factor_backtest"], feedback))
        self._last_feedback = feedback

        # ── KV-cache: chain feedback → next iteration's propose ──
        feedback_kv = self.summarizer.last_kv
        if feedback_kv is not None and kv_truncate is not None:
            self._pipeline_kv = kv_truncate(feedback_kv, self._kv_max_tokens)
            logger.info(
                f"[LatentPipeline] Chained feedback KV → next propose "
                f"(truncated to {self._kv_max_tokens} tokens)"
            )

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
            # KV-cache dari akhir loop (setelah feedback + truncate).
            # Digunakan oleh evolution controller untuk meneruskan konteks
            # latent ke mutation/crossover round berikutnya.
            "pipeline_kv": getattr(self, "_pipeline_kv", None),
        }




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
