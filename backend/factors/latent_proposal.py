"""
latent_proposal.py — KV-cache aware proposal classes for LatentMAS pipeline
============================================================================

Extends the standard proposal classes (AlphaAgentHypothesisGen,
AlphaAgentHypothesis2FactorExpression, AlphaAgentQlibFactorHypothesisExperiment2Feedback)
to use LocalLLMBackend.build_messages_and_run() for KV-cache chaining
between pipeline steps.

KV-cache flow within one AlphaAgentLoop iteration:

    past_kv (from planning / external agents / previous iteration's feedback)
        |
    LatentHypothesisGen.gen()
        mode = kv_and_text
        build KV from trace + direction, generate hypothesis text
        |  kv_propose
    LatentHypothesis2Experiment.convert()
        mode = kv_and_text
        continue KV from propose, generate factor expression text
        |  kv_construct
    LatentFeedback.generate_feedback()
        mode = kv_and_text
        generate feedback text, produce KV for next iteration
        |  kv_feedback
    loop.py: truncate kv_feedback → self._pipeline_kv (next iteration)

After each step, ``last_kv`` holds the output KV-cache, which the next
step reads via ``set_past_kv()``.  This implements the "latent reasoning"
pipeline from the LatentMAS paper: agents communicate through latent
KV-cache in addition to explicit text, enabling richer context transfer
with fewer tokens.

Usage (handled automatically by AlphaAgentLoop when llm_backend is set):

    gen = LatentHypothesisGen(scen, direction, llm_backend=backend)
    gen.set_past_kv(insight_kv)
    hypothesis = gen.gen(trace)

    con = LatentHypothesis2Experiment(llm_backend=backend)
    con.set_past_kv(gen.last_kv)
    experiment = con.convert(hypothesis, trace)

    fb = LatentFeedback(scen, llm_backend=backend)
    fb.set_past_kv(con.last_kv)
    feedback = fb.generate_feedback(experiment, hypothesis, trace)
"""

from __future__ import annotations

from typing import Any, Optional

from log import logger

from llm.client import LocalLLMBackend, LLMResult, KVCache, OutputMode
from factors.proposal import (
    AlphaAgentHypothesisGen,
    AlphaAgentHypothesis2FactorExpression,
)
from factors.feedback import AlphaAgentQlibFactorHypothesisExperiment2Feedback


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: KV-cache state management (shared by all three Latent classes)
# ═══════════════════════════════════════════════════════════════════════════

class _LatentMixin:
    """
    Mixin providing KV-cache state management for Latent proposal classes.

    Provides:
        set_past_kv(kv)        — set KV-cache input for next _call_llm
        set_mode(mode)         — set output mode for next _call_llm
        set_latent_steps(n)    — override latent steps for next _call_llm
        last_kv                — property: KV-cache output from last _call_llm
        last_result            — full LLMResult from last _call_llm
    """

    def _init_latent_state(
        self,
        llm_backend: LocalLLMBackend,
        default_mode: OutputMode = "kv_and_text",
        latent_steps: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.llm_backend: LocalLLMBackend = llm_backend
        self.last_result: Optional[LLMResult] = None
        self._past_kv: Optional[KVCache] = None
        self._mode: OutputMode = default_mode
        self._latent_steps: Optional[int] = latent_steps
        # Per-step temperature override.
        # None = pakai default dari LocalLLMBackend.
        # Construct step sebaiknya lebih rendah (misal 0.3) agar
        # formula expression lebih presisi dan bisa di-parse.
        self._temperature: Optional[float] = temperature

    def set_past_kv(self, kv: Optional[KVCache]) -> None:
        """Set KV-cache yang akan dipakai di _call_llm berikutnya."""
        self._past_kv = kv

    def set_mode(self, mode: OutputMode) -> None:
        """Set output mode (kv_and_text / text_only / kv_only)."""
        self._mode = mode

    def set_latent_steps(self, n: Optional[int]) -> None:
        """Override jumlah latent steps (None = pakai default engine)."""
        self._latent_steps = n

    @property
    def last_kv(self) -> Optional[KVCache]:
        """KV-cache dari panggilan _call_llm terakhir."""
        if self.last_result is not None:
            return self.last_result.kv_cache
        return None

    def __getstate__(self):
        """Exclude unpicklable attributes saat rdagent logger.log_object() dipanggil.

        llm_backend berisi PyTorch model dengan thread locks → tidak bisa di-pickle.
        _past_kv dan last_result berisi GPU tensors → tidak bisa di-pickle.
        Pola ini sama dengan LoopBase._non_picklable_attrs di utils/workflow.py.
        """
        state = self.__dict__.copy()
        for key in ("llm_backend", "_past_kv", "last_result"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.llm_backend = None
        self._past_kv = None
        self.last_result = None


# ═══════════════════════════════════════════════════════════════════════════
# LatentHypothesisGen
# ═══════════════════════════════════════════════════════════════════════════

class LatentHypothesisGen(_LatentMixin, AlphaAgentHypothesisGen):
    """
    KV-cache aware hypothesis generator.

    Overrides ``_call_llm()`` to use ``build_messages_and_run()`` sehingga
    setiap panggilan LLM menghasilkan KV-cache yang bisa di-chain ke step
    berikutnya (LatentHypothesis2Experiment).

    Juga override ``_get_scenario_desc()`` untuk pakai get_compact_desc("propose")
    — prompt lebih pendek menyisakan lebih banyak kapasitas KV-cache
    untuk latent reasoning via virtual tokens.

    Default mode: kv_and_text (build KV + generate text).
    """

    def __init__(self, scen, potential_direction=None, *,
                 llm_backend: LocalLLMBackend, latent_steps: Optional[int] = None,
                 temperature: Optional[float] = None):
        AlphaAgentHypothesisGen.__init__(self, scen, potential_direction)
        self._init_latent_state(llm_backend, default_mode="kv_and_text",
                                latent_steps=latent_steps, temperature=temperature)

    def _get_scenario_desc(self) -> str:
        """Compact scenario untuk latent mode.

        Pakai get_compact_desc("propose") yang hanya menyertakan
        background + strategy dengan section markers — jauh lebih
        ringkas dari get_scenario_all_desc(). Prompt lebih pendek
        menyisakan kapasitas KV-cache untuk latent reasoning.
        """
        if hasattr(self.scen, "get_compact_desc"):
            return self.scen.get_compact_desc("propose")
        # Fallback jika scenario bukan QlibAlphaAgentScenario
        return self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment")

    def _call_llm(self, user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        result = self.llm_backend.build_messages_and_run(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=json_mode,
            past_key_values=self._past_kv,
            mode=self._mode,
            role="propose",
            latent_steps=self._latent_steps,
            temperature=self._temperature,
        )
        self.last_result = result
        logger.info(
            f"[LatentHypothesisGen] mode={self._mode}, "
            f"has_kv={result.has_kv}, text_len={len(result.text or '')}"
        )
        return result.text or ""


# ═══════════════════════════════════════════════════════════════════════════
# LatentHypothesis2Experiment
# ═══════════════════════════════════════════════════════════════════════════

class LatentHypothesis2Experiment(_LatentMixin, AlphaAgentHypothesis2FactorExpression):
    """
    KV-cache aware factor expression constructor.

    Receives KV-cache dari LatentHypothesisGen dan melanjutkan konteks
    latent sambil menghasilkan ekspresi faktor.

    Output mode: kv_and_text — SELALU menghasilkan teks DAN KV-cache:
      - Teks: ekspresi faktor (di-parse QlibFactorParser → Python code,
        disimpan di trajectory pool, ditampilkan ke user untuk monitoring)
      - KV-cache: representasi latent dari konteks construct (di-chain
        ke feedback step via loop.py, memperkaya latent reasoning)

    Catatan retry loop:
        Jika expression validation gagal dan retry terjadi di
        _convert_with_history_limit, setiap retry menggunakan past_kv
        yang sama (dari propose step) — bukan KV dari attempt sebelumnya.
        Feedback duplikasi dikirim via teks di user_prompt.
        last_result di-update setiap _call_llm() sehingga last_kv
        selalu KV dari attempt terakhir (untuk feedback step chain).
    """

    def __init__(self, *args, llm_backend: LocalLLMBackend,
                 latent_steps: Optional[int] = None,
                 temperature: Optional[float] = None, **kwargs):
        AlphaAgentHypothesis2FactorExpression.__init__(self, *args, **kwargs)
        self._init_latent_state(llm_backend, default_mode="kv_and_text",
                                latent_steps=latent_steps, temperature=temperature)

    def _get_scenario_desc(self, trace) -> str:
        """Compact scenario untuk latent construct mode.

        Pakai get_compact_desc("construct") yang menyertakan interface +
        output_format dengan section markers — konteks yang paling relevan
        untuk formula expression generation. Background sudah ada di
        KV-cache dari propose step (di-chain via _past_kv).

        Prompt lebih ringkas → lebih banyak kapasitas KV-cache untuk
        latent reasoning di construct step, dimana model perlu "berpikir"
        lebih dalam untuk menghasilkan ekspresi matematika yang presisi.
        """
        if hasattr(trace.scen, "get_compact_desc"):
            return trace.scen.get_compact_desc("construct")
        # Fallback jika scenario bukan QlibAlphaAgentScenario
        return trace.scen.background

    def _call_llm(self, user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        result = self.llm_backend.build_messages_and_run(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=json_mode,
            past_key_values=self._past_kv,
            mode=self._mode,
            role="construct",
            latent_steps=self._latent_steps,
            temperature=self._temperature,
        )
        self.last_result = result
        logger.info(
            f"[LatentHypothesis2Experiment] mode={self._mode}, "
            f"has_kv={result.has_kv}, text_len={len(result.text or '')}"
        )
        return result.text or ""


# ═══════════════════════════════════════════════════════════════════════════
# LatentFeedback
# ═══════════════════════════════════════════════════════════════════════════

class LatentFeedback(_LatentMixin, AlphaAgentQlibFactorHypothesisExperiment2Feedback):
    """
    KV-cache aware feedback generator.

    Receives KV-cache dari LatentHypothesis2Experiment.
    Default mode: kv_and_text — feedback melakukan latent reasoning
    DAN menghasilkan KV-cache yang di-chain ke propose step
    di iterasi berikutnya (via loop.py's _pipeline_kv).

    Ini memungkinkan akumulasi konteks latent antar iterasi:
    feedback_kv → (truncate) → next propose's past_kv.
    """

    def __init__(self, scen, *, llm_backend: LocalLLMBackend,
                 latent_steps: Optional[int] = None,
                 temperature: Optional[float] = None):
        AlphaAgentQlibFactorHypothesisExperiment2Feedback.__init__(self, scen)
        # Mode kv_and_text (bukan text_only):
        #   Feedback melakukan latent reasoning DAN menghasilkan KV-cache.
        #   KV ini di-chain kembali ke propose step di iterasi berikutnya,
        #   sehingga model memiliki "memori latent" antar iterasi loop.
        #   Ini berbeda dari Latent-MAS original di mana hanya judger
        #   yang generate text — di sini SEMUA step butuh text output
        #   karena pipeline downstream (coder, runner) memerlukan teks.
        self._init_latent_state(llm_backend, default_mode="kv_and_text",
                                latent_steps=latent_steps, temperature=temperature)

    def _get_scenario_desc(self) -> str:
        """Compact scenario untuk latent feedback mode.

        Pakai get_compact_desc("feedback") yang menyertakan background +
        experiment_setting dengan section markers — cukup untuk konteks
        evaluasi tanpa overhead full scenario dump.

        Background sudah ada di KV-cache dari construct step (di-chain
        via _past_kv), tapi disertakan kembali di teks karena feedback
        perlu menyatakan ulang konteks market secara eksplisit untuk
        evaluasi yang tepat.
        """
        if hasattr(self.scen, "get_compact_desc"):
            return self.scen.get_compact_desc("feedback")
        return self.scen.get_scenario_all_desc()

    def _call_llm(self, user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        result = self.llm_backend.build_messages_and_run(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=json_mode,
            past_key_values=self._past_kv,
            mode=self._mode,
            role="feedback",
            latent_steps=self._latent_steps,
            temperature=self._temperature,
        )
        self.last_result = result
        logger.info(
            f"[LatentFeedback] mode={self._mode}, "
            f"text_len={len(result.text or '')}"
        )
        return result.text or ""
