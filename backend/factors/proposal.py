"""
berisi implementasi dari "propose hypothesis" dan
"convert hypothesis -> factor".
di loop.py, class disini di-load via import_class() dari string path di setting.py
"""

import json
import re
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined

from factors.coder.factor import FactorExperiment, FactorTask
from components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from core.prompts import Prompts
from core.proposal import Hypothesis, Scenario, Trace
from core.experiment import Experiment
from factors.experiment import QlibFactorExperiment
from llm.client import LocalLLMBackend, robust_json_parse
from utils.prompt_markers import wrap as _mv
import os
import pandas as pd
from log import logger
from factors.regulator.factor_regulator import FactorRegulator

# TODO belum menerapkan debug/monitor.py

DEFAULT_HISTORY_LIMIT = 6  # max riwayat yang dimasukkan ke prompt
MIN_HISTORY_LIMIT = 1   # min riwayat


def render_hypothesis_and_feedback(prompt_dict, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> str:
    """Render hypothesis_and_feedback with configurable history limit."""
    if len(trace.hist) > 0:
        # buat trace baru kosong(scenario sama)
        limited_trace = Trace(scen=trace.scen)
        
        # masukkan hanya sebagian riwayat sesuai history_limit
        limited_trace.hist = trace.hist[-history_limit:] if history_limit > 0 else trace.hist
        
        # render dengan trace terbatas
        # setiap entry jadi teks: "Round 1: Hypothesis: ... Feedback: ... Decision: ..."
        return (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_and_feedback"])
            .render(trace=limited_trace)
        )
    else:
        return "No previous hypothesis and feedback available since it's the first round."


def is_input_length_error(error_msg: str) -> bool:
    """Check if error is due to input length limit."""
    error_indicators = [
        "input length",
        "context length", 
        "maximum context",
        "token limit",
        "InvalidParameter",
        "Range of input length",
        "max_tokens",
        "too long"
    ]
    error_str = str(error_msg).lower()
    return any(indicator.lower() in error_str for indicator in error_indicators)

# QlibFactorHypothesis = Hypothesis dari core/proposal.py
# dipakai oleh QlibFactorHypothesisGen
QlibFactorHypothesis = Hypothesis

qa_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "proposal.yaml")

# hypothesis khusus AlphaAgent
#? mengapa ada hypothesis khusus?
#? ada berapa macam hypothesis? apa bedanya?
class AlphaAgentHypothesis(Hypothesis):
    """
    AlphaAgentHypothesis extends the Hypothesis class to include a potential_direction,
    which represents the initial idea or starting point for the hypothesis.
    """

    def __init__(
        self,
        hypothesis: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        concise_specification: str #FIELD BARU untuk menyimpan hipotesis lebih lengkap
    ) -> None:
        super().__init__(
            hypothesis,
            "",
            "",
            concise_observation,
            concise_justification,
            concise_knowledge,
        )
        self.concise_specification = concise_specification
        
    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
                Concise Observation: {self.concise_observation}
                Concise Justification: {self.concise_justification}
                Concise Knowledge: {self.concise_knowledge}
                concise Specification: {self.concise_specification}
                """

base_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "prompts.yaml")

#! dipakai oleh FactorBasePropSetting, BUKAN AlphaAgent
class QlibFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    #* render riwayat + format output -> context dict
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(base_prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": base_prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": base_prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True   # True = json_mode

    #* parse JSON string -> dict
    def convert_response(self, response: str) -> Hypothesis:
        response_dict = robust_json_parse(response)
        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            reason=response_dict.get("reason", ""),
            concise_reason=response_dict.get("concise_reason", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
        )
        return hypothesis   # QlibFactorHypothesis = Hypothesis

# convert hypothesis -> factor
class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()   #* deskripsi lengkap skenario
        
        #format JSON yang LLM harus ikuti untuk generate faktor
        experiment_output_format = base_prompt_dict["experiment_output_format"]

        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(base_prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        #* t = (hypothesis, experiment, feedback)
        #* t[1] = experiment dari setiap loop sebelumnya
        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]

        factor_list = []
        #* mengumpulkan semua faktor dari eksperimen sebelumnya sebagai conteks untuk LLM
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
            "RAG": None,
        }, True

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = robust_json_parse(response)
        tasks = []

        for factor_name in response_dict:
            factor_data = response_dict.get(factor_name, {})
            if not isinstance(factor_data, dict):
                continue    # skip entry yang bukan dict
            description = factor_data.get("description", "")
            formulation = factor_data.get("formulation", "")
            # expression = factor_data.get("expression", "")
            variables = factor_data.get("variables", {})
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    # factor_expression=expression,
                    variables=variables,
                )
            )

        #* buat experiment dengan list FactorTask
        exp = QlibFactorExperiment(tasks)
        
        # experiment2 sebelumnya yang berhasil(feedback=True)
        # ditambah satu experiment kosong di awal (baseline)
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

        unique_tasks = []

        #*  DEDUPLIKASI: buang faktor yang namanya sama dengan yang sudah ada
        #*  di experiment sebelumnya — jangan buat faktor yang sudah pernah dibuat
        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp


#! prompt sekarang dari prompt.yaml BUKAN proposal.
qa_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "prompts.yaml")

# prompt_dict not as attribute: class instance is pickled later, prompt_dict cannot be pickled
#! dipakai di loop.py : AlphaAgentLoop.factor_propose()
#* FactorHypothesisGen -> LLMHypothesisGen -> HypothesisGen
class AlphaAgentHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario, potential_direction: str=None) -> Tuple[dict, bool]:
        super().__init__(scen)
        self.potential_direction = potential_direction
        #* direction yang dibuat di AlphagentLoop.__init__
        #* base direction + strategy_suffix + external_context

        # ── KV-cache state (opsional) ─────────────────────────────
        # Dipakai saat llm_backend mendukung build_messages_and_run().
        # Latent subclass (LatentHypothesisGen) override _call_llm()
        # untuk selalu pakai latent path; base class auto-detect.
        self._past_kv = None        # KV-cache input untuk panggilan berikutnya
        self.last_result = None     # LLMResult lengkap dari panggilan terakhir

    def set_past_kv(self, kv) -> None:
        """Set KV-cache yang akan dipakai di _call_llm berikutnya."""
        self._past_kv = kv

    @property
    def last_kv(self):
        """KV-cache output dari panggilan _call_llm terakhir."""
        if self.last_result is not None:
            return getattr(self.last_result, "kv_cache", None)
        return None

    # siapkan input untuk LLM
    def prepare_context(self, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> Tuple[dict, bool]:
        
        # ada riwayat: render N entry terakhir jadi teks
        if len(trace.hist) > 0:
            hypothesis_and_feedback = render_hypothesis_and_feedback(
                qa_prompt_dict, trace, history_limit
            )
            
        #* round pertama + ada direction:
        #  mengubah direction -> format yang dipahami LLM
        elif self.potential_direction is not None: 
            hypothesis_and_feedback = (
                Environment(undefined=StrictUndefined)
                .from_string(qa_prompt_dict["potential_direction_transformation"])
                .render(potential_direction=self.potential_direction)
            )
        
        # round pertama + tanpa direction => LLM explore bebas
        else:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round. You are encouraged to propose an innovative hypothesis that diverges significantly from existing perspectives."
            
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": qa_prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": qa_prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> AlphaAgentHypothesis:
        """
        Convert LLM JSON to AlphaAgentHypothesis; handles both standard keys and
        mutation/feedback-style keys that emerge when past_kv primes the model in
        feedback mode (e.g. "New Hypothesis", "Observations", "Key Knowledge").
        """
        response_dict = robust_json_parse(response)

        def _first(*keys, default=""):
            for k in keys:
                v = response_dict.get(k)
                if v:
                    return v if isinstance(v, str) else str(v)
            return default

        def _strip_data_colrefs(text: str) -> str:
            # Remove parenthetical mentions of data column names like
            # "(1day.excess_return_without_cost)" — identified by dot + underscore
            # pattern (typical of data columns). Leaves "(0.05)" etc. intact.
            return re.sub(r'\s*\([^)]*\b\w+\.\w*_\w+[^)]*\)', '', text).strip()

        hypothesis_text = _first("hypothesis", "New Hypothesis", "new_hypothesis")
        concise_observation = _strip_data_colrefs(
            _first("concise_observation", "Observations", "observations")
        )
        concise_justification = _first(
            "concise_justification", "Feedback for Hypothesis", "Reasoning", "reasoning"
        )
        # Key Knowledge may be a dict; serialize it
        ck = (
            response_dict.get("concise_knowledge")
            or response_dict.get("Key Knowledge")
            or response_dict.get("key_knowledge")
            or response_dict.get("Reasoning")
            or ""
        )
        if isinstance(ck, dict):
            ck = "; ".join(f"{k}: {v}" for k, v in ck.items())
        concise_knowledge = ck if isinstance(ck, str) else str(ck)

        cs = (
            response_dict.get("concise_specification")
            or response_dict.get("Evaluation Metrics")
            or response_dict.get("evaluation_metrics")
            or ""
        )
        if isinstance(cs, dict):
            cs = str(cs)
        concise_specification = cs

        return AlphaAgentHypothesis(
            hypothesis=hypothesis_text,
            concise_observation=concise_observation,
            concise_knowledge=concise_knowledge,
            concise_justification=concise_justification,
            concise_specification=concise_specification,
        )
    
    def _call_llm(self, user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        """
        Call LLM via self.llm_backend (LocalLLMBackend).

        self.llm_backend di-set oleh AlphaAgentLoop.__init__.
        Auto-detect: jika _past_kv tersedia dan backend mendukung
        build_messages_and_run, pakai latent path (KV consumed + generated).
        Override di subclass Latent untuk parameter lanjutan (role, latent_steps).
        """
        # ── Latent path: auto-detect KV-cache support ──
        if self._past_kv is not None and hasattr(self.llm_backend, "build_messages_and_run"):
            result = self.llm_backend.build_messages_and_run(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=json_mode,
                past_key_values=self._past_kv,
                mode="kv_and_text",
            )
            self.last_result = result
            logger.info(
                f"[HypothesisGen] KV auto-detect: has_kv={result.has_kv}, "
                f"text_len={len(result.text or '')}"
            )
            return result.text or ""

        # ── Standard text-only path ──
        # Reset last_result agar last_kv tidak return KV stale
        # dari panggilan latent sebelumnya.
        self.last_result = None
        return self.llm_backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt,
            json_mode=json_mode,
        )

    def _get_scenario_desc(self) -> str:
        """Scenario description untuk system prompt.

        Override di LatentHypothesisGen untuk pakai get_compact_desc("propose")
        yang lebih ringkas — prompt lebih pendek menyisakan kapasitas
        KV-cache untuk latent reasoning.
        """
        return self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")

    def gen(self, trace: Trace) -> AlphaAgentHypothesis:
        """Generate hypothesis; supports dynamic history limit for input length."""
        history_limit = DEFAULT_HISTORY_LIMIT   # max 6
        scenario_desc = self._get_scenario_desc()

        while history_limit >= MIN_HISTORY_LIMIT: #hitung mundur dari 6
            try:
                context_dict, json_flag = self.prepare_context(trace, history_limit)
                system_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis_gen"]["system_prompt"])
                    .render(
                        targets=_mv("targets", self.targets),
                        scenario=_mv("scenario", scenario_desc),
                        hypothesis_output_format=_mv("hypothesis_output_format", context_dict["hypothesis_output_format"]),
                        hypothesis_specification=_mv("hypothesis_specification", context_dict["hypothesis_specification"]),
                    )
                )
                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis_gen"]["user_prompt"])
                    .render(
                        targets=_mv("targets", self.targets),
                        hypothesis_and_feedback=_mv("hypothesis_and_feedback", context_dict["hypothesis_and_feedback"]),
                        RAG=_mv("RAG", context_dict["RAG"]),
                        round=len(trace.hist)
                    )
                )

                resp = self._call_llm(user_prompt, system_prompt, json_flag)
                hypothesis = self.convert_response(resp)
                return hypothesis

            except Exception as e:
                if is_input_length_error(str(e)) and history_limit > MIN_HISTORY_LIMIT:
                    history_limit -= 1
                    logger.warning(f"Input length exceeded, retrying with history_limit={history_limit}...")
                else:
                    raise

        # Last attempt with minimum history limit
        context_dict, json_flag = self.prepare_context(trace, MIN_HISTORY_LIMIT)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis_gen"]["system_prompt"])
            .render(
                targets=_mv("targets", self.targets),
                scenario=_mv("scenario", scenario_desc),
                hypothesis_output_format=_mv("hypothesis_output_format", context_dict["hypothesis_output_format"]),
                hypothesis_specification=_mv("hypothesis_specification", context_dict["hypothesis_specification"]),
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis_gen"]["user_prompt"])
            .render(
                targets=_mv("targets", self.targets),
                hypothesis_and_feedback=_mv("hypothesis_and_feedback", context_dict["hypothesis_and_feedback"]),
                RAG=_mv("RAG", context_dict["RAG"]),
                round=len(trace.hist)
            )
        )
        resp = self._call_llm(user_prompt, system_prompt, json_flag)
        hypothesis = self.convert_response(resp)
        return hypothesis


#* return hipotesis kosong(dipakai di backtestloop)
class EmptyHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)
        
    def convert_response(self, *args, **kwargs) -> AlphaAgentHypothesis: 
        return super().convert_response(*args, **kwargs)  
    
    def prepare_context(self, *args, **kwargs) -> Tuple[dict | bool]:
        return super().prepare_context(*args, **kwargs)

    def gen(self, trace: Trace) -> AlphaAgentHypothesis:

        hypothesis = AlphaAgentHypothesis(
            hypothesis="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
            concise_specification=""
        )

        return hypothesis



#! convert hypothesis → faktor + validasi.
#! dipanggil di AlphaAgentLoop.factor_construct()
# inherit: FactorHypothesis2Experiment → LLMHypothesis2Experiment → Hypothesis2Experiment
class AlphaAgentHypothesis2FactorExpression(FactorHypothesis2Experiment):

    def __init__(self, *args, consistency_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        #* self.targets = "factors" (dari FactorHypothesis2Experiment)

        # ── KV-cache state (opsional) ─────────────────────────────
        self._past_kv = None        # KV-cache input untuk panggilan berikutnya
        self.last_result = None     # LLMResult lengkap dari panggilan terakhir
        
        # Initialize FactorRegulator with config settings
        from factors.coder.config import FACTOR_COSTEER_SETTINGS
        
        #*  FactorRegulator: mengevaluasi ekspresi faktor
        #   - apakah bisa di-parse?
        #   - apakah terlalu mirip dengan faktor yang sudah ada (alpha zoo)?
        #   - apakah terlalu kompleks?
        self.factor_regulator = FactorRegulator(
            factor_zoo_path=FACTOR_COSTEER_SETTINGS.factor_zoo_path,
            duplication_threshold=FACTOR_COSTEER_SETTINGS.duplication_threshold,
            symbol_length_threshold=FACTOR_COSTEER_SETTINGS.symbol_length_threshold,
            base_features_threshold=FACTOR_COSTEER_SETTINGS.base_features_threshold,
        )
        
        # Initialize consistency checker if enabled
        self.consistency_enabled = consistency_enabled
        self._quality_gate = None

    def set_past_kv(self, kv) -> None:
        """Set KV-cache yang akan dipakai di _call_llm berikutnya."""
        self._past_kv = kv

    @property
    def last_kv(self):
        """KV-cache output dari panggilan _call_llm terakhir."""
        if self.last_result is not None:
            return getattr(self.last_result, "kv_cache", None)
        return None

    @property
    def quality_gate(self):
        """Lazy-load FactorQualityGate."""
        
        #*  lazy-load: hanya buat kalau dibutuhkan DAN consistency_enabled=True
        #*  FactorQualityGate: cek apakah faktor konsisten dengan hipotesis
        if self._quality_gate is None and self.consistency_enabled:
            try:
                from factors.regulator.consistency_checker import FactorQualityGate
                self._quality_gate = FactorQualityGate(
                    consistency_enabled=self.consistency_enabled,
                    complexity_enabled=True,
                    redundancy_enabled=True
                )
            except ImportError as e:
                logger.warning(f"Could not load consistency checker: {e}")
                self._quality_gate = None
        return self._quality_gate
        
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()   # all skenario dari QLIB
        
        experiment_output_format = qa_prompt_dict["experiment_output_format"]
        function_lib_description = qa_prompt_dict['function_lib_description']
        hypothesis_and_feedback = render_hypothesis_and_feedback(
            qa_prompt_dict, trace, history_limit
        )

        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)    # kumpulan faktor dari eksperimen sebelumnya sebagai konteks untuk LLM

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "function_lib_description": function_lib_description,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
            "RAG": None,
        }, True

    def _call_llm(self, user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        """
        Call LLM via self.llm_backend (LocalLLMBackend).

        self.llm_backend di-set oleh AlphaAgentLoop.__init__.
        Auto-detect: jika _past_kv tersedia dan backend mendukung
        build_messages_and_run, pakai latent path.
        Override di subclass Latent untuk parameter lanjutan (role, latent_steps).
        """
        # ── Latent path: auto-detect KV-cache support ──
        if self._past_kv is not None and hasattr(self.llm_backend, "build_messages_and_run"):
            result = self.llm_backend.build_messages_and_run(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=json_mode,
                past_key_values=self._past_kv,
                mode="kv_and_text",
            )
            self.last_result = result
            logger.info(
                f"[Hypothesis2Expr] KV auto-detect: has_kv={result.has_kv}, "
                f"text_len={len(result.text or '')}"
            )
            return result.text or ""

        # ── Standard text-only path ──
        # Reset last_result agar last_kv tidak return KV stale
        # dari panggilan latent sebelumnya.
        self.last_result = None
        return self.llm_backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt,
            json_mode=json_mode,
        )

    def _get_scenario_desc(self, trace: Trace) -> str:
        """Scenario description untuk construct step system prompt.

        Berbeda dari hypothesis_gen yang punya self.scen, class ini
        mengakses scenario via trace.scen (Hypothesis2Experiment tidak
        menerima scen di __init__).

        Default: background only (current behavior).
        Override di LatentHypothesis2Experiment untuk pakai
        get_compact_desc("construct") yang menyertakan interface +
        output_format — lebih relevan untuk formula generation.
        """
        return trace.scen.background

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        """Convert hypothesis to factor expressions; supports dynamic history limit."""
        history_limit = DEFAULT_HISTORY_LIMIT

        while history_limit >= MIN_HISTORY_LIMIT:
            try:
                return self._convert_with_history_limit(hypothesis, trace, history_limit)
            except Exception as e:
                if is_input_length_error(str(e)) and history_limit > MIN_HISTORY_LIMIT:
                    history_limit -= 1
                    logger.warning(f"Input length exceeded, retrying with history_limit={history_limit}...")
                else:
                    raise

        # Last attempt with minimum history limit
        return self._convert_with_history_limit(hypothesis, trace, MIN_HISTORY_LIMIT)

    #* generator + validasi faktor
    def _convert_with_history_limit(self, hypothesis: Hypothesis, trace: Trace, history_limit: int) -> Experiment:
        """Convert with given history limit.

        KV-cache note (retry loop):
            Attempt 1 menggunakan full prompts + propose KV.
            Attempt 2+: minimal retry prompt (error+hypothesis+functions) tanpa
            repeating full scenario/guidance — mencegah prompt overload Qwen3-4B.
            Attempt 3+: juga reset past_kv ke None (KV clean slate).
        """
        context, json_flag = self.prepare_context(hypothesis, trace, history_limit)
        scenario_desc = self._get_scenario_desc(trace)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=_mv("targets", self.targets),
                scenario=_mv("scenario", scenario_desc),
                experiment_output_format=_mv("experiment_output_format", context["experiment_output_format"]),
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=_mv("targets", self.targets),
                target_hypothesis=_mv("target_hypothesis", context["target_hypothesis"]),
                hypothesis_and_feedback=_mv("hypothesis_and_feedback", context["hypothesis_and_feedback"]),
                function_lib_description=_mv("function_lib_description", context["function_lib_description"]),
                target_list=_mv("target_list", context["target_list"]),
                RAG=_mv("RAG", context["RAG"]),
                expression_duplication=None,
            )
        )

        # Strategy B: templates untuk retry minimal prompt
        _retry_sys_tpl  = qa_prompt_dict.get("hypothesis2experiment_retry_system_prompt", "")
        _retry_user_tpl = qa_prompt_dict.get("hypothesis2experiment_retry_user_prompt", "")
        _compact_schema  = qa_prompt_dict.get("factor_experiment_compact_schema", "")
        _hypothesis_title = getattr(hypothesis, 'hypothesis', str(hypothesis))

        #* Detect duplicated sub-expressions
        flag = False    # semua faktor valid?
        expression_duplication_prompt = None    # feedback duplikasi untuk attempt 1
        error_log: list[str] = []               # error log ringkas untuk retry prompt
        _MAX_CONSTRUCT_RETRIES = 6
        _construct_retries = 0
        while True:
            if flag:
                break   #* semua faktor sudah valid -> keluar loop

            if _construct_retries >= _MAX_CONSTRUCT_RETRIES:
                # Setelah N retry tanpa perbaikan, terima faktor yang sudah
                # valid (jika ada) atau paksa keluar agar pipeline tidak macet.
                logger.warning(
                    f"[Construct] max retries ({_MAX_CONSTRUCT_RETRIES}) reached, "
                    f"accepting {len(proposed_names)} valid factor(s) found so far."
                )
                break
            _construct_retries += 1

            #* panggil LLM => generate faktor
            # Attempt 1: full prompt + propose KV (normal path)
            # Attempt 2+: minimal retry prompt — mencegah prompt overload pada small model
            # Attempt 3+: juga drop past_kv (KV clean slate)
            if _construct_retries >= 2 and _retry_sys_tpl and _retry_user_tpl:
                error_summary = " | ".join(error_log[-3:]) or "format invalid"
                _cur_sys = _retry_sys_tpl
                _cur_user = (
                    Environment(undefined=StrictUndefined)
                    .from_string(_retry_user_tpl)
                    .render(
                        attempt_n=_construct_retries,
                        error_log=_mv("error_log", error_summary),
                        hypothesis_title=_mv("hypothesis_title", _hypothesis_title),
                        function_lib_description=_mv("function_lib_description", context["function_lib_description"]),
                        compact_schema=_mv("compact_schema", _compact_schema),
                    )
                )
                logger.info(
                    f"[Construct] attempt {_construct_retries}: minimal retry prompt"
                    f" (error: {error_summary[:80]})"
                )
                if _construct_retries >= 3 and self._past_kv is not None:
                    # KV reset: hapus kontaminasi dari propose step
                    _saved_kv = self._past_kv
                    self._past_kv = None
                    try:
                        resp = self._call_llm(_cur_user, _cur_sys, json_flag)
                    finally:
                        self._past_kv = _saved_kv
                else:
                    resp = self._call_llm(_cur_user, _cur_sys, json_flag)
            else:
                resp = self._call_llm(user_prompt, system_prompt, json_flag)

            try:
                # parse JSON => dict
                response_dict = robust_json_parse(resp)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed: {e}, retrying...")
                error_log.append(f"JSON parse failed: output must be raw JSON object starting with {{\"FactorName\"")
                continue

            proposed_names = [] # nama faktor yang valid
            proposed_exprs = [] # ekspresi faktor yang valid

            for i, factor_name in enumerate(response_dict):
                factor_data = response_dict.get(factor_name, {})
                if not isinstance(factor_data, dict):
                    continue

                expr = factor_data.get("expression", "")
                description = factor_data.get("description", "")
                formulation = factor_data.get("formulation", "")
                variables = factor_data.get("variables", {})

                #* Check if expression is parsable
                if not self.factor_regulator.is_parsable(expr):
                    logger.warning(f"[Construct] retry {_construct_retries}/{_MAX_CONSTRUCT_RETRIES}: not parsable: {expr!r}")
                    error_log.append(f"not parsable: {expr[:60]!r} — use only $open/$close/$high/$low/$volume/$return")
                    parse_fb = (
                        f"- Expression Not Parsable: `{expr}` could not be parsed. "
                        f"ONLY use variables starting with `$` ($open, $close, $high, "
                        f"$low, $volume, $return). "
                        f"Do NOT use dot-notation names like `1day.excess_return_without_cost` "
                        f"or any undeclared variable. Use `$return` for daily returns."
                    )
                    expression_duplication_prompt = (
                        '\n\n'.join([expression_duplication_prompt, parse_fb])
                        if expression_duplication_prompt else parse_fb
                    )
                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=_mv("targets", self.targets),
                            target_hypothesis=_mv("target_hypothesis", context["target_hypothesis"]),
                            hypothesis_and_feedback=_mv("hypothesis_and_feedback", context["hypothesis_and_feedback"]),
                            function_lib_description=_mv("function_lib_description", context["function_lib_description"]),
                            target_list=_mv("target_list", context["target_list"]),
                            RAG=_mv("RAG", context["RAG"]),
                            expression_duplication=_mv("expression_duplication", expression_duplication_prompt),
                        )
                    )
                    break

                #   evaluate() melakukan:
                #      1. parse AST dari ekspresi
                #      2. match_alphazoo: cek subtree mana yang sudah ada di alpha zoo
                #      3. hitung: num_free_args, num_unique_vars, num_all_nodes
                #      4. hitung: symbol_length, num_base_features
                #   return: (success, dict metrik)
                success, eval_dict = self.factor_regulator.evaluate(expr)
                if not success:
                    logger.warning(f"[Construct] retry {_construct_retries}/{_MAX_CONSTRUCT_RETRIES}: evaluate failed: {expr!r}")
                    error_log.append(f"eval failed: {expr[:60]!r} — check variable names and syntax")
                    eval_fb = (
                        f"- Expression Evaluation Failed: `{expr}` failed evaluation. "
                        f"ONLY use variables starting with `$` ($open, $close, $high, "
                        f"$low, $volume, $return) and allowed operators. "
                        f"Do NOT use dot-notation names (e.g., `1day.excess_return_without_cost`). "
                        f"Use `$return` for daily returns."
                    )
                    expression_duplication_prompt = (
                        '\n\n'.join([expression_duplication_prompt, eval_fb])
                        if expression_duplication_prompt else eval_fb
                    )
                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=_mv("targets", self.targets),
                            target_hypothesis=_mv("target_hypothesis", context["target_hypothesis"]),
                            hypothesis_and_feedback=_mv("hypothesis_and_feedback", context["hypothesis_and_feedback"]),
                            function_lib_description=_mv("function_lib_description", context["function_lib_description"]),
                            target_list=_mv("target_list", context["target_list"]),
                            RAG=_mv("RAG", context["RAG"]),
                            expression_duplication=_mv("expression_duplication", expression_duplication_prompt),
                        )
                    )
                    break
                
                #* Consistency check (if enabled)
                if self.consistency_enabled and self.quality_gate is not None:
                    try:
                        passed, feedback, results = self.quality_gate.evaluate(
                            hypothesis=str(hypothesis),
                            factor_name=factor_name,
                            factor_description=description,
                            factor_formulation=formulation,
                            factor_expression=expr,
                            variables=variables
                        )
                        
                        # Use corrected expression from consistency check if provided
                        # update dengan yang sudah di cek
                        if results.get("corrected_expression") and results["corrected_expression"] != expr:
                            logger.info(f"Consistency check corrected expression: {expr} -> {results['corrected_expression']}")
                            expr = results["corrected_expression"]
                            factor_data["expression"] = expr
                            response_dict[factor_name] = factor_data
                            
                            # Re-check corrected expression
                            if not self.factor_regulator.is_parsable(expr):
                                logger.warning(f"Corrected expression could not be parsed: {expr}")
                                break
                            success, eval_dict = self.factor_regulator.evaluate(expr)
                            if not success:
                                break
                        
                        if not passed:
                            logger.warning(f"Consistency check failed: {factor_name}, feedback: {feedback}")
                    except Exception as e:
                        logger.warning(f"Consistency check error: {e}")
                
                # If expression has problems, regenerate with feedback
                # check acceptability (duplikasi + kompleksity)
                if not self.factor_regulator.is_expression_acceptable(eval_dict):
                    logger.warning(
                        f"[Construct] retry {_construct_retries}/{_MAX_CONSTRUCT_RETRIES}: "
                        f"not acceptable: {expr!r} | "
                        f"dup_size={eval_dict.get('duplicated_subtree_size')}, "
                        f"sl={eval_dict.get('symbol_length')}, "
                        f"base_feat={eval_dict.get('num_base_features')}"
                    )
                    sl = eval_dict.get('symbol_length', 0)
                    dup = eval_dict.get('duplicated_subtree_size', 0)
                    error_log.append(
                        f"not acceptable: sl={sl}, dup={dup} — use different structure"
                    )
                    
                    # Calculate ratios for feedback
                    num_all_nodes = eval_dict['num_all_nodes']
                    free_args_ratio = float(eval_dict['num_free_args']) / float(num_all_nodes) if num_all_nodes > 0 else 0.0
                    unique_vars_ratio = float(eval_dict['num_unique_vars']) / float(num_all_nodes) if num_all_nodes > 0 else 0.0
                    
                    # Get symbol length and base features count for complexity feedback
                    symbol_length = eval_dict.get('symbol_length', 0)
                    num_base_features = eval_dict.get('num_base_features', 0)
                    symbol_length_threshold = self.factor_regulator.symbol_length_threshold
                    base_features_threshold = self.factor_regulator.base_features_threshold
                    
                    #* feedback template => beritahu LLM apa yang salah
                    feedback_item = (
                            Environment(undefined=StrictUndefined)
                            .from_string(qa_prompt_dict["expression_duplication"])
                            .render(
                                prev_expression=expr,
                                duplicated_subtree_size=eval_dict['duplicated_subtree_size'],
                            duplication_threshold=self.factor_regulator.duplication_threshold,
                            duplicated_subtree=eval_dict.get('duplicated_subtree', ''),
                            matched_alpha=eval_dict.get('matched_alpha', ''),
                            free_args_ratio=free_args_ratio,
                            num_free_args=eval_dict['num_free_args'],
                            unique_vars_ratio=unique_vars_ratio,
                            num_unique_vars=eval_dict['num_unique_vars'],
                            num_all_nodes=num_all_nodes,
                            symbol_length=symbol_length,
                            symbol_length_threshold=symbol_length_threshold,
                            num_base_features=num_base_features,
                            base_features_threshold=base_features_threshold
                            )
                        )
                    
                    # kumpulkan dan gabung feedback
                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = '\n\n'.join([expression_duplication_prompt, feedback_item])
                    else:
                        expression_duplication_prompt = feedback_item
                    
                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=_mv("targets", self.targets),
                            target_hypothesis=_mv("target_hypothesis", context["target_hypothesis"]),
                            hypothesis_and_feedback=_mv("hypothesis_and_feedback", context["hypothesis_and_feedback"]),
                            function_lib_description=_mv("function_lib_description", context["function_lib_description"]),
                            target_list=_mv("target_list", context["target_list"]),
                            RAG=_mv("RAG", context["RAG"]),
                            expression_duplication=_mv("expression_duplication", expression_duplication_prompt),
                        )
                    )
                    break       #* break 'for loop' -> mulai lagi 'while loop'
                else:
                    #* Intra-response duplicate check: tolak bila expression
                    # ini identik dengan factor lain di response yang SAMA.
                    # `factor_regulator.is_expression_acceptable` hanya cek
                    # duplikasi terhadap alpha-zoo (factor history lintas
                    # iterasi), bukan duplikat antar factor di satu response.
                    # Construct kadang menghasilkan 2-3 factor dengan
                    # expression identik (cuma beda nama+deskripsi) → coder
                    # retry mencari variasi dan sering collapse karena task
                    # nya redundant.
                    expr_norm = re.sub(r"\s+", "", expr).lower()
                    existing_norms = [re.sub(r"\s+", "", e).lower() for e in proposed_exprs]
                    if expr_norm in existing_norms:
                        dup_idx = existing_norms.index(expr_norm)
                        dup_name = proposed_names[dup_idx]
                        dup_expr = proposed_exprs[dup_idx]
                        logger.warning(
                            f"[Construct] retry {_construct_retries}/{_MAX_CONSTRUCT_RETRIES}: "
                            f"intra-response duplicate: {factor_name!r} == {dup_name!r}, "
                            f"expr={expr!r}"
                        )
                        error_log.append(
                            f"intra-duplicate: {factor_name!r}=={dup_name!r} — each factor must have different expression"
                        )
                        intra_dup_feedback = (
                            f"- Intra-Response Duplicate: factor `{factor_name}` and "
                            f"`{dup_name}` share the IDENTICAL expression `{dup_expr}`. "
                            f"Each factor in your output MUST have a structurally DIFFERENT "
                            f"expression — different name and description alone do not "
                            f"qualify. Change at least one of: operator family (e.g., "
                            f"TS_STD → TS_MAD), window size, or base variable."
                        )
                        if expression_duplication_prompt is not None:
                            expression_duplication_prompt = "\n\n".join(
                                [expression_duplication_prompt, intra_dup_feedback]
                            )
                        else:
                            expression_duplication_prompt = intra_dup_feedback
                        user_prompt = (
                            Environment(undefined=StrictUndefined)
                            .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                            .render(
                                targets=_mv("targets", self.targets),
                                target_hypothesis=_mv("target_hypothesis", context["target_hypothesis"]),
                                hypothesis_and_feedback=_mv("hypothesis_and_feedback", context["hypothesis_and_feedback"]),
                                function_lib_description=_mv("function_lib_description", context["function_lib_description"]),
                                target_list=_mv("target_list", context["target_list"]),
                                RAG=_mv("RAG", context["RAG"]),
                                expression_duplication=_mv("expression_duplication", expression_duplication_prompt),
                            )
                        )
                        break       #* break for-loop, while-loop akan retry construct

                    proposed_names.append(factor_name)
                    proposed_exprs.append(expr)
                    if i == len(response_dict) - 1:
                        flag = True
                    else:
                        continue        #* jika masih ada faktor lain -> loop lanjut
        

        #* Add valid factors to the factor regulator
        self.factor_regulator.add_factor(proposed_names, proposed_exprs)
                
        #* parse JSON terakhir -> buat FactorExperiment
        return self.convert_response(resp, trace)
    

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = robust_json_parse(response)
        tasks = []

        for factor_name in response_dict:
            factor_data = response_dict.get(factor_name, {})
            if not isinstance(factor_data, dict):
                continue
            description = factor_data.get("description", "")
            formulation = factor_data.get("formulation", "")
            expression = factor_data.get("expression", "")
            variables = factor_data.get("variables", {})
            
            # setiap entry JSON -> satu FactorTask
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    factor_expression=expression,
                    variables=variables,
                )
            )
            
        #* buat eksperimen baru
        exp = QlibFactorExperiment(tasks)
        
        #* based_experiments = experiment sebelumnya yang sukses
        # dipakai untuk cek duplikasi nama faktor
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

        unique_tasks = []

        for task in tasks:
            duplicate = False
            
            # proses pengecekan duplikasinya
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp


#* Dipakai oleh FactorBackTestBasePropSetting — load faktor dari CSV file.
class BacktestHypothesis2FactorExpression(FactorHypothesis2Experiment):
    def __init__(self, factor_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_path = factor_path      # path ke CSV file berisi faktor yang sudah jadi
        
    def convert_response(self, *args, **kwargs) -> FactorExperiment:
        return super().convert_response(*args, **kwargs)
        
    def prepare_context(self, *args, **kwargs) -> Tuple[dict | bool]:
        return super().prepare_context(*args, **kwargs)
        
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        if os.path.exists(self.factor_path):
            tasks = []
            factor_df = pd.read_csv(self.factor_path, usecols=["factor_name", "factor_expression"], index_col=None)
            for index, row in factor_df.iterrows():
                
                # cukup nama faktor dan ekspresinya yang diperlukan untuk backtest
                tasks.append(
                    FactorTask(
                        factor_name=row["factor_name"],
                        factor_description="",
                        factor_formulation="",
                        factor_expression=row["factor_expression"],
                        variables="",
                    )
                )
            
            #* pengecekan duplikasi lagi
            exp = QlibFactorExperiment(tasks)
            exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

            unique_tasks = []

            for task in tasks:
                duplicate = False
                for based_exp in exp.based_experiments:
                    for sub_task in based_exp.sub_tasks:
                        if task.factor_name == sub_task.factor_name:
                            duplicate = True
                            break
                    if duplicate:
                        break
                if not duplicate:
                    unique_tasks.append(task)

            exp.tasks = unique_tasks
            return exp
            
        else:
            raise ValueError(f"File {self.factor_csv_path} does not exist. ")
        
    