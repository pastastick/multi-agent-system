from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Optional
from jinja2 import Environment, StrictUndefined

from coder.costeer.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from coder.costeer.knowledge_management import (
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from factors.coder.config import FACTOR_COSTEER_SETTINGS
from factors.coder.factor import FactorFBWorkspace, FactorTask
from core.prompts import Prompts
from core.template import CodeTemplate
from llm.config import LLM_SETTINGS
from llm.client import LocalLLMBackend
from llm._shared import _past_length
from core.utils import multiprocessing_wrapper
from core.conf import RD_AGENT_SETTINGS
from log import logger
from utils.prompt_markers import wrap as _mv

code_template = CodeTemplate(template_path=Path(__file__).parent / "template.jinjia2")
implement_prompts = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

#* untuk FactorCoSTEER
# tradisional = full LLM generate kode Python
class FactorMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False


    def error_summary(
        self,
        target_task: FactorTask,
        queried_former_failed_knowledge_to_render: list,
        queried_similar_error_knowledge_to_render: list,
    ) -> str:
        error_summary_system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(implement_prompts["evolving_strategy_error_summary_v2_system"])
            .render(
                scenario=_mv("scenario", self.scen.get_scenario_all_desc(target_task)),
                factor_information_str=_mv("factor_information_str", target_task.get_task_information()),
                code_and_feedback=_mv("code_and_feedback", queried_former_failed_knowledge_to_render[-1].get_implementation_and_feedback_str()),
            )
            .strip("\n")
        )
        for _ in range(10):  # max attempt to reduce the length of error_summary_user_prompt
            error_summary_user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(implement_prompts["evolving_strategy_error_summary_v2_user"])
                .render(
                    queried_similar_error_knowledge=_mv("queried_similar_error_knowledge", queried_similar_error_knowledge_to_render),
                )
                .strip("\n")
            )
            if (
                LocalLLMBackend().build_messages_and_calculate_token(
                    user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt
                )
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        error_summary_critics = LocalLLMBackend().build_messages_and_create_chat_completion(
            user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt, json_mode=False
        )
        return error_summary_critics

    # generate kode untuk SATU faktor task
    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
    ) -> str:
        target_factor_task_information = target_task.get_task_information()

        # Knowledge dari CosSTEER
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_factor_task_information]
            if queried_knowledge is not None
            else []
        )  # A list, [success task implement knowledge]

        if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2):
            queried_similar_error_knowledge = (
                queried_knowledge.task_to_similar_error_successful_knowledge[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )  # A dict, {{error_type:[[error_imp_knowledge, success_imp_knowledge],...]},...}
        else:
            queried_similar_error_knowledge = {}

        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][0]
            if queried_knowledge is not None
            else []
        )

        queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

        latest_attempt_to_latest_successful_execution = queried_knowledge.task_to_former_failed_traces[
            target_factor_task_information
        ][1]

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(
                implement_prompts["evolving_strategy_factor_implementation_v1_system"],
            )
            .render(
                scenario=_mv("scenario", self.scen.get_scenario_all_desc(target_task, filtered_tag="feature")),
                queried_former_failed_knowledge=_mv("queried_former_failed_knowledge", queried_former_failed_knowledge_to_render),
            )
        )
        
        queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
        queried_similar_error_knowledge_to_render = queried_similar_error_knowledge
        
        #* buid user prompt dengan semua knowledge -> check token count
        for _ in range(10):
            # Optional error summary
            if (
                isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
                and FACTOR_COSTEER_SETTINGS.v2_error_summary
                and len(queried_similar_error_knowledge_to_render) != 0
                and len(queried_former_failed_knowledge_to_render) != 0
            ):
                error_summary_critics = self.error_summary(
                    target_task,
                    queried_former_failed_knowledge_to_render,
                    queried_similar_error_knowledge_to_render,
                )
            else:
                error_summary_critics = None
            similar_successful_factor_description = ""
            similar_successful_expression = ""
            if len(queried_similar_successful_knowledge_to_render) > 0:
                similar_successful_factor_description = queried_similar_successful_knowledge_to_render[0].target_task.get_task_description()
                similar_successful_expression = self.extract_expr(queried_similar_successful_knowledge_to_render[0].implementation.code)
            
            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    implement_prompts["evolving_strategy_factor_implementation_v2_user"],
                )
                .render(
                    factor_information_str=_mv("factor_information_str", target_task.get_task_description()),
                    queried_similar_error_knowledge=_mv("queried_similar_error_knowledge", queried_similar_error_knowledge_to_render),
                    error_summary_critics=_mv("error_summary_critics", error_summary_critics),
                    similar_successful_factor_description=_mv("similar_successful_factor_description", similar_successful_factor_description),
                    similar_successful_expression=_mv("similar_successful_expression", similar_successful_expression),
                    latest_attempt_to_latest_successful_execution=_mv("latest_attempt_to_latest_successful_execution", latest_attempt_to_latest_successful_execution),
                )
                .strip("\n")
            )
            if (
                LocalLLMBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_former_failed_knowledge_to_render) > 1:
                queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
            elif len(queried_similar_successful_knowledge_to_render) > len(
                queried_similar_error_knowledge_to_render,
            ):
                queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[:-1]
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        for _ in range(10):
            try:
                code = json.loads(
                    LocalLLMBackend().build_messages_and_create_chat_completion(
                        user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True
                    )
                )["code"]
                return code
            except json.decoder.JSONDecodeError:
                pass
        else:
            return ""  # return empty code if failed to get code after 10 attempts

    # inject kode ke workspace masing-masing task
    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_code(**{"factor.py": code_list[index]})
        return evo


#! dipakai QuantaAlpha
#* parsing template dulu, LLM hanya fix expression jika error
qa_implement_prompts = Prompts(file_path=Path(__file__).parent / "qa_prompts.yaml")
class FactorParsingStrategy(MultiProcessEvolvingStrategy):
    """
    Evolving strategy untuk AlphaAgent pipeline.

    Run pertama: render template dari ekspresi (tanpa LLM).
    Jika gagal: panggil LLM untuk perbaiki ekspresi.

    Latent pipeline (llm_backend is not None):
      - LLM calls menggunakan build_messages_and_run() dengan KV-cache
      - KV dari construct step di-inject sebagai konteks awal
      - KV akumulasi antar LLM calls dalam evolve loop
      - Evolve berjalan sequential (bukan multiprocessing) karena
        GPU tensor tidak bisa cross process boundaries
      - last_kv property expose KV terakhir untuk downstream (feedback)
    """

    def __init__(self, *args,
                 llm_backend: Optional[Any] = None,
                 latent_steps: Optional[int] = None,
                 temperature: Optional[float] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False

        # ── Latent pipeline state ────────────────────────────────────
        self._llm_backend: Optional[Any] = llm_backend
        self._past_kv: Optional[Any] = None        # KV input dari construct step
        self._last_kv: Optional[Any] = None         # KV output terakhir
        self._latent_steps: Optional[int] = latent_steps
        self._temperature: Optional[float] = temperature

    # ── Latent setters (dipanggil dari CoSTEER.develop) ──────────────

    def set_llm_backend(self, backend: Any) -> None:
        self._llm_backend = backend

    def set_past_kv(self, kv: Optional[Any]) -> None:
        """Set KV-cache dari construct step."""
        self._past_kv = kv

    @property
    def last_kv(self) -> Optional[Any]:
        """KV-cache output terakhir dari LLM call dalam evolve loop."""
        return self._last_kv

    @property
    def _is_latent(self) -> bool:
        """Apakah latent pipeline aktif."""
        return (
            self._llm_backend is not None
            and hasattr(self._llm_backend, "build_messages_and_run")
        )

    # ── Helper: get backend for LLM calls ────────────────────────────

    def _get_backend(self, use_cache: bool = True) -> LocalLLMBackend:
        """Return shared llm_backend jika latent, else buat baru."""
        if self._llm_backend is not None:
            return self._llm_backend
        return LocalLLMBackend()

    def _call_llm(self, user_prompt: str, system_prompt: str,
                   json_mode: bool = True, reasoning_flag: bool = False,
                   temperature: Optional[float] = None) -> str:
        """
        Unified LLM call — auto-detect latent vs text-only.

        Latent path: build_messages_and_run() dengan KV-cache.
          - Menerima _past_kv (dari construct atau LLM call sebelumnya)
          - Update _last_kv setelah call
          - Mode kv_and_text: generate text DAN KV-cache

        Text-only path: build_messages_and_create_chat_completion().
          - Behavior identik dengan kode original
        """
        if self._is_latent:
            result = self._llm_backend.build_messages_and_run(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=json_mode,
                past_key_values=self._past_kv,
                mode="kv_and_text",
                role="coder",
                latent_steps=self._latent_steps,
                temperature=temperature if temperature is not None else self._temperature,
                max_new_tokens=512,
            )
            text_out = result.text or ""
            logger.info(
                f"[LatentCoder] mode=kv_and_text, "
                f"has_kv={result.has_kv}, text_len={len(text_out)}"
            )
            # KV besar dari construct menyebabkan model collapse (<think> only,
            # text_len=0). Fallback ke text-only agar pipeline tidak mandeg.
            if not text_out.strip():
                logger.warning(
                    "[LatentCoder] kv_and_text collapse detected (text_len=0), "
                    "fallback to text_only"
                )
                text_out = LocalLLMBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=json_mode,
                    reasoning_flag=reasoning_flag,
                )
                return text_out
            # Update KV state hanya kalau tidak fallback
            self._last_kv = result.kv_cache
            self._past_kv = result.kv_cache
            return text_out
        else:
            return LocalLLMBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=json_mode,
                reasoning_flag=reasoning_flag,
            )

    def error_summary(
        self,
        target_task: FactorTask,
        queried_former_failed_knowledge_to_render: list,
        queried_similar_error_knowledge_to_render: list,
    ) -> str:
        """Summarize errors from previous attempts. Latent-aware."""
        error_summary_system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(implement_prompts["evolving_strategy_error_summary_v2_system"])
            .render(
                scenario=_mv("scenario", self.scen.get_scenario_all_desc(target_task)),
                factor_information_str=_mv("factor_information_str", target_task.get_task_information()),
                code_and_feedback=_mv("code_and_feedback", queried_former_failed_knowledge_to_render[-1].get_implementation_and_feedback_str()),
            )
            .strip("\n")
        )
        for _ in range(10):
            error_summary_user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(implement_prompts["evolving_strategy_error_summary_v2_user"])
                .render(
                    queried_similar_error_knowledge=_mv("queried_similar_error_knowledge", queried_similar_error_knowledge_to_render),
                )
                .strip("\n")
            )
            if (
                self._get_backend().build_messages_and_calculate_token(
                    user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt
                )
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]

        return self._call_llm(
            user_prompt=error_summary_user_prompt,
            system_prompt=error_summary_system_prompt,
            json_mode=False,
        )

    def extract_expr(self, code_str: str) -> str:
        """Extract expr from code (expr = \"...\" or expr = '...')."""
        pattern = r'expr\s*=\s*["\']([^"\']*)["\']'
        match = re.search(pattern, code_str)
        if match:
            return match.group(1)
        else:
            return ""


    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
    ) -> str:
        """Generate code for one factor task. First run: template; on error: give LLM feedback and cases."""
        target_factor_task_information = target_task.get_task_information()

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_factor_task_information]
            if queried_knowledge is not None
            else []
        )

        if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2):
            queried_similar_error_knowledge = (
                queried_knowledge.task_to_similar_error_successful_knowledge[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )  # A dict, {{error_type:[[error_imp_knowledge, success_imp_knowledge],...]},...}
        else:
            queried_similar_error_knowledge = {}

        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][0]
            if queried_knowledge is not None
            else []
        )

        queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

        #* RUN PERTAMA langsung render template TANPA LLM
        if len(queried_former_failed_knowledge) == 0:
            logger.info(f"[LatentCoder] first-run template path, expr={target_task.factor_expression}")
            
            rendered_code = code_template.render(
                expression=target_task.factor_expression,
                factor_name=target_task.factor_name
            )
            return rendered_code

        #* RETRY(setelah gagal): panggil LLM untuk perbaiki
        else:
            logger.info(f"[LatentCoder] retry path, former_expr={self.extract_expr(queried_former_failed_knowledge[-1].implementation.code)}")

            latest_attempt_to_latest_successful_execution = queried_knowledge.task_to_former_failed_traces[
                target_factor_task_information
            ][1]

            # KV dari construct step sudah encode scenario + function lib →
            # pakai system prompt ringkas. Text-only fallback tetap butuh full prompt.
            if self._past_kv is not None:
                system_prompt = qa_implement_prompts["evolving_strategy_coder_system_kv"]
                logger.info("[LatentCoder] using compact KV-aware system prompt")
            else:
                system_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(
                        qa_implement_prompts["evolving_strategy_factor_implementation_v1_system"],
                    )
                    .render(
                        scenario=_mv("scenario", self.scen.get_scenario_all_desc(target_task, filtered_tag="feature")),
                    )
                )

            queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
            queried_similar_error_knowledge_to_render = queried_similar_error_knowledge

            for _ in range(10):
                if (
                    isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
                    and FACTOR_COSTEER_SETTINGS.v2_error_summary
                    and len(queried_similar_error_knowledge_to_render) != 0
                    and len(queried_former_failed_knowledge_to_render) != 0
                ):
                    error_summary_critics = self.error_summary(
                        target_task,
                        queried_former_failed_knowledge_to_render,
                        queried_similar_error_knowledge_to_render,
                    )
                else:
                    error_summary_critics = None

                similar_successful_factor_description = ""
                similar_successful_expression = ""
                if len(queried_similar_successful_knowledge_to_render) > 0:
                    similar_successful_factor_description = queried_similar_successful_knowledge_to_render[-1].target_task.get_task_description()
                    similar_successful_expression = self.extract_expr(queried_similar_successful_knowledge_to_render[-1].implementation.code)

                # Pisahkan execution log (raw traceback) dan code_comment (LLM review)
                # agar coder langsung melihat error aktual di bagian atas user prompt.
                last_fb = queried_former_failed_knowledge_to_render[-1].feedback
                execution_log = getattr(last_fb, "execution_feedback", None) or ""
                code_comment = getattr(last_fb, "code_feedback", None) or ""

                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(
                        qa_implement_prompts["evolving_strategy_factor_implementation_v2_user"],
                    )
                    .render(
                        factor_information_str=_mv("factor_information_str", target_task.get_task_description()),
                        queried_similar_error_knowledge=_mv("queried_similar_error_knowledge", queried_similar_error_knowledge_to_render),
                        former_expression=_mv("former_expression", self.extract_expr(queried_former_failed_knowledge_to_render[-1].implementation.code)),
                        execution_log=_mv("execution_log", execution_log),
                        code_comment=_mv("code_comment", code_comment),
                        error_summary_critics=_mv("error_summary_critics", error_summary_critics),
                        similar_successful_factor_description=_mv("similar_successful_factor_description", similar_successful_factor_description),
                        similar_successful_expression=_mv("similar_successful_expression", similar_successful_expression),
                        latest_attempt_to_latest_successful_execution=_mv("latest_attempt_to_latest_successful_execution", latest_attempt_to_latest_successful_execution),
                    )
                    .strip("\n")
                )

                if (
                    self._get_backend().build_messages_and_calculate_token(
                        user_prompt=user_prompt, system_prompt=system_prompt
                    )
                    < LLM_SETTINGS.chat_token_limit
                ):
                    break
                elif len(queried_former_failed_knowledge_to_render) > 1:
                    queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
                elif len(queried_similar_successful_knowledge_to_render) > len(
                    queried_similar_error_knowledge_to_render,
                ):
                    queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[:-1]
                elif len(queried_similar_error_knowledge_to_render) > 0:
                    queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]

            # Capture former_expr untuk validasi anti-mirroring.
            # Simpan dua versi: yang asli (untuk fallback render) dan yang
            # ter-normalisasi (untuk pembanding mirror).
            former_expr_raw = self.extract_expr(
                queried_former_failed_knowledge_to_render[-1].implementation.code
            )
            former_expr_norm = former_expr_raw.replace(" ", "").lower()

            # Snapshot KV-cache length SEBELUM retry loop. DynamicCache di-mutasi
            # in-place oleh latent_pass (append prompt + latent steps) — sekedar
            # menyimpan referensi `kv_baseline = self._past_kv` lalu reassign
            # `self._past_kv = kv_baseline` TIDAK mengembalikan panjang asli.
            # Setelah attempt 1 KV sudah memuat prompt_1 + latent_1; attempt 2
            # akan menambah prompt_2 + latent_2 di atasnya, sehingga konteks
            # menumpuk dan model collapse ke output `<think>` saja (output_tokens=3
            # lalu EOS) karena polusi konteks.
            kv_baseline = self._past_kv
            kv_baseline_len = _past_length(kv_baseline) if kv_baseline is not None else 0

            # Paksa model output HANYA JSON — tanpa penjelasan sebelumnya.
            # Qwen3-4B di large KV context (>13k tok) cenderung generate verbose
            # analysis dulu baru JSON di akhir. _fix_json bisa gagal ekstrak
            # kalau JSON tertimbun di teks panjang.
            json_only_suffix = (
                "\n\nOUTPUT INSTRUCTION: Respond with ONLY the raw JSON object "
                "on a single line. No explanation, no preamble, no analysis. "
                'Example: {"expr": "TS_STD($close, 20)"}'
            )

            # Temperature escalation: 0.3 → 0.7 → 1.0 setelah mirror / JSON fail
            _MAX_ATTEMPTS = 5
            temp_schedule = [None, 0.7, 0.9, 1.0, 1.0]
            mirror_hint = ""
            last_expr = None

            for attempt in range(_MAX_ATTEMPTS):
                try:
                    # Reset KV ke baseline pre-retry. Penting: crop kembali ke
                    # panjang baseline supaya attempt ini melihat KV yang sama
                    # dengan attempt 1 — bukan KV yang sudah ter-append oleh
                    # attempt sebelumnya.
                    if kv_baseline is not None and hasattr(kv_baseline, "crop"):
                        try:
                            kv_baseline.crop(kv_baseline_len)
                        except Exception as crop_err:
                            logger.warning(
                                f"[LatentCoder] attempt {attempt+1}: kv_baseline.crop "
                                f"failed ({crop_err}); proceeding with current KV"
                            )
                    self._past_kv = kv_baseline

                    # Inject mirror-warning ke user prompt bila attempt sebelumnya mirror
                    effective_user_prompt = user_prompt + json_only_suffix + mirror_hint

                    raw = self._call_llm(
                        user_prompt=effective_user_prompt,
                        system_prompt=system_prompt,
                        json_mode=True,
                        reasoning_flag=False,
                        temperature=temp_schedule[attempt],
                    )
                    expr = json.loads(raw)["expr"]
                    expr_norm = expr.replace(" ", "").lower()

                    # Validasi: tolak kalau identik dengan former_expr
                    if expr_norm == former_expr_norm:
                        last_expr = expr
                        mirror_hint = (
                            f"\n\n**PREVIOUS ATTEMPT RETURNED THE SAME EXPRESSION "
                            f"({last_expr}) — THIS IS A FAILURE. You MUST change operator, "
                            f"window size, or base variable. Try a structurally different "
                            f"expression now.**"
                        )
                        logger.warning(
                            f"[LatentCoder] attempt {attempt+1}: LLM mirrored former_expr, retrying with temp={temp_schedule[min(attempt+1, _MAX_ATTEMPTS-1)]}"
                        )
                        continue

                    logger.info(
                        f"[LatentCoder] attempt {attempt+1}: new expr accepted (diff from former)"
                    )
                    rendered_code = code_template.render(
                        expression=expr,
                        factor_name=target_task.factor_name
                    )
                    return rendered_code

                except json.decoder.JSONDecodeError:
                    logger.warning(f"[LatentCoder] attempt {attempt+1}: JSON parse failed, retrying")
                    continue

            # Fallback: semua attempt mirror/fail → pakai expr terakhir dengan
            # warning agar pipeline tetap jalan (evaluator akan reject kalau beneran bad)
            if last_expr is not None:
                logger.error(
                    f"[LatentCoder] all {_MAX_ATTEMPTS} attempts mirrored former_expr, "
                    f"falling back to last output: {last_expr}"
                )
                return code_template.render(
                    expression=last_expr,
                    factor_name=target_task.factor_name
                )
            # Kalau semua JSON fail, return template dengan former_expr ASLI
            # (bukan yang ter-normalisasi lower+nospace, karena parser DSL
            # case-sensitive — TS_STD bukan ts_std).
            logger.error(f"[LatentCoder] all {_MAX_ATTEMPTS} attempts failed JSON parse, using former_expr")
            return code_template.render(
                expression=former_expr_raw,
                factor_name=target_task.factor_name
            )

    def evolve(
        self,
        *,
        evo,
        queried_knowledge=None,
        **kwargs,
    ):
        """Override evolve() untuk sequential mode saat latent aktif.

        GPU tensor (KV-cache) tidak bisa cross process boundaries
        via multiprocessing.Queue. Saat latent pipeline aktif,
        jalankan implement_one_task secara sequential (n=1).
        Text-only path tetap pakai multiprocessing seperti biasa.
        """
        from coder.costeer.evolvable_subjects import EvolvingItem

        # Find tasks to evolve
        to_be_finished_task_index = []
        for index, target_task in enumerate(evo.sub_tasks):
            target_task_desc = target_task.get_task_information()
            if queried_knowledge is not None and target_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                evo.sub_workspace_list[index] = queried_knowledge.success_task_to_knowledge_dict[
                    target_task_desc
                ].implementation
            elif (
                queried_knowledge is None
                or (
                    target_task_desc not in queried_knowledge.success_task_to_knowledge_dict
                    and target_task_desc not in queried_knowledge.failed_task_info_set
                )
            ):
                to_be_finished_task_index.append(index)

        # Selection: if over limit, select a subset
        if self.settings.select_threshold < len(to_be_finished_task_index):
            to_be_finished_task_index = self.select_one_round_tasks(
                to_be_finished_task_index, evo, self.settings.select_threshold, queried_knowledge, self.scen
            )

        if self._is_latent:
            # ── Latent: sequential mode ──────────────────────────────
            # Tiap task masuk dengan KV "pre-batch" yang sama (snapshot
            # sebelum task pertama). KV residual dari task sebelumnya
            # (termasuk retry attempts yang sukses/gagal) di-crop kembali
            # supaya tidak meracuni baseline task berikutnya — pernah bikin
            # task-2 retry collapse ke `<think>` saja walau per-attempt
            # crop di implement_one_task() sudah jalan.
            batch_baseline = self._past_kv
            batch_baseline_len = (
                _past_length(batch_baseline) if batch_baseline is not None else 0
            )
            result = []
            for target_index in to_be_finished_task_index:
                if batch_baseline is not None and hasattr(batch_baseline, "crop"):
                    try:
                        batch_baseline.crop(batch_baseline_len)
                    except Exception as crop_err:
                        logger.warning(
                            f"[LatentCoder] evolve: batch_baseline.crop "
                            f"failed ({crop_err}); proceeding with current KV"
                        )
                self._past_kv = batch_baseline
                code = self.implement_one_task(evo.sub_tasks[target_index], queried_knowledge)
                result.append(code)
            logger.info(
                f"[LatentCoder] Sequential evolve: {len(to_be_finished_task_index)} tasks, "
                f"baseline_len={batch_baseline_len}, "
                f"has_kv={self._last_kv is not None}"
            )
        else:
            # ── Text-only: parallel mode (original behavior) ─────────
            result = multiprocessing_wrapper(
                [
                    (self.implement_one_task, (evo.sub_tasks[target_index], queried_knowledge))
                    for target_index in to_be_finished_task_index
                ],
                n=RD_AGENT_SETTINGS.multi_proc_n,
            )

        code_list = [None for _ in range(len(evo.sub_tasks))]
        for index, target_index in enumerate(to_be_finished_task_index):
            code_list[target_index] = result[index]

        evo = self.assign_code_list_to_evo(code_list, evo)
        evo.corresponding_selection = to_be_finished_task_index

        return evo

    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_code(**{"factor.py": code_list[index]})
        return evo
    
    
    
class FactorRunningStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False


    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
    ) -> str:

        rendered_code = code_template.render(
            expression=target_task.factor_expression, 
            factor_name=target_task.factor_name 
        )
        return rendered_code
        
    
    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_code(**{"factor.py": code_list[index]})
        return evo
    
    
    def evolve(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        **kwargs,
    ) -> EvolvingItem:
        # Find tasks to evolve
        to_be_finished_task_index = []
        for index, target_task in enumerate(evo.sub_tasks):
            to_be_finished_task_index.append(index)

        result = multiprocessing_wrapper(
            [
                (self.implement_one_task, (evo.sub_tasks[target_index], queried_knowledge))
                for target_index in to_be_finished_task_index
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )
        code_list = [None for _ in range(len(evo.sub_tasks))]
        for index, target_index in enumerate(to_be_finished_task_index):
            code_list[target_index] = result[index]

        evo = self.assign_code_list_to_evo(code_list, evo)
        evo.corresponding_selection = to_be_finished_task_index

        return evo
