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
            user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt,
            json_mode=False, role="coder_error_summary",
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
                        user_prompt=user_prompt, system_prompt=system_prompt,
                        json_mode=True, role="coder",
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
    Jika gagal: panggil coder agent (repair-or-pass) — output PASS atau FIXED.

    Latent pipeline (llm_backend is not None):
      - LLM calls menggunakan build_messages_and_run() dengan KV-cache
      - KV dari construct step di-inject sebagai konteks awal
      - Evolve berjalan sequential (bukan multiprocessing) karena
        GPU tensor tidak bisa cross process boundaries
      - last_kv property expose KV terakhir untuk downstream (feedback)
      - coder_retry chain dari construct_kv (_coder_kv) — KV yang memuat
        scenario + function lib + hipotesis original. Tidak ada perantara
        eval_kv lagi karena reviewer/final_decision sudah dihapus dari
        pipeline (semantic gate digabung ke coder agent itu sendiri).
      - Eskalasi per-attempt via system prompt: minimal → different → bold.
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
        """Set KV-cache dari construct step (baseline untuk retry attempts)."""
        self._past_kv = kv
        if self._llm_backend is not None:
            self._llm_backend._coder_kv = kv

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
                    role="coder",
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
                role="coder",
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
            .from_string(qa_implement_prompts["evolving_strategy_error_summary_v2_system"])
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
                .from_string(qa_implement_prompts["evolving_strategy_error_summary_v2_user"])
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

    # Sentinel yang dikembalikan _parse_repair_output kalau model balas "PASS"
    # (kontrak repair-or-pass — LLM judge bahwa ekspresi sebenarnya sudah OK).
    PASS_SENTINEL = "__PASS__"

    @staticmethod
    def _parse_repair_output(raw: str) -> Optional[str]:
        """Parse repair-or-pass LLM output.

        Kontrak: satu baris, salah satu dari:
          PASS                          → keep ekspresi lama (PASS_SENTINEL)
          FIXED: <expression>           → expression string

        Parser permissive untuk FIXED (case-insensitive FIXED/EXPR/RESULT,
        strip quote/backtick wrapper, paren balancing). Return:
          - PASS_SENTINEL bila model balas "PASS"
          - ekspresi string bila FIXED valid
          - None bila tidak bisa diparse sama sekali
        """
        if not raw:
            return None
        text = raw.strip()
        # Form A: "PASS" (case-insensitive, opsional tanda baca terminal).
        # Periksa baris pertama saja untuk hindari false match di tengah teks.
        first_line = text.splitlines()[0].strip() if text.splitlines() else ""
        if re.fullmatch(r'pass[.!]?', first_line, flags=re.IGNORECASE):
            return FactorParsingStrategy.PASS_SENTINEL
        # Form B: "FIXED: <expr>" (EXPR/RESULT diterima sebagai legacy).
        keyword_re = re.compile(
            r'^\s*(?:fixed|expr|result)\s*:\s*(.+?)\s*$', flags=re.IGNORECASE
        )
        for line in text.splitlines():
            if not line.strip():
                continue
            m = keyword_re.match(line)
            if m:
                expr = m.group(1).strip()
                # Strip pembungkus umum (backtick, quote) di luar ekspresi.
                for q in ('`', '"', "'"):
                    if len(expr) >= 2 and expr.startswith(q) and expr.endswith(q):
                        expr = expr[1:-1].strip()
                # Kalau model menambah keyword kedua di line yang sama
                # (mis. "FOO(...) FIXED: BAR(...)"), potong di kemunculan
                # keyword berikutnya.
                m2 = re.search(
                    r'\s+(?:fixed|expr|result)\s*:', expr, flags=re.IGNORECASE,
                )
                if m2:
                    expr = expr[:m2.start()].rstrip()
                # Paren balancing: kalau penutup ")" lebih banyak dari pembuka,
                # potong di posisi seimbang (cegah trailing junk yang sering
                # diappend model 4B di akhir line).
                depth = 0
                for i, ch in enumerate(expr):
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth < 0:
                            expr = expr[:i].rstrip()
                            break
                if expr:
                    return expr
        # Fallback: legacy JSON shape {"expr": "..."} — regex non-greedy.
        m = re.search(r'"(?:expr|fixed)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            return m.group(1).strip()
        return None


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

        #* RETRY(setelah gagal): panggil coder agent (repair-or-pass)
        else:
            former_expr_raw = self.extract_expr(
                queried_former_failed_knowledge[-1].implementation.code
            )
            logger.info(f"[LatentCoder] retry path, former_expr={former_expr_raw}")

            latest_attempt_to_latest_successful_execution = queried_knowledge.task_to_former_failed_traces[
                target_factor_task_information
            ][1]

            # Eskalasi per-attempt (minimal → different → bold). Render system
            # prompt per attempt karena varian beda di-render dari template
            # yang sama dengan attempt_mode berbeda.
            _MAX_ATTEMPTS = 3
            _ATTEMPT_MODES = ["minimal", "different", "bold"]
            temp_schedule = [None, 0.7, 0.9]

            # Pilih template system prompt: KV-aware compact bila KV ada,
            # full-context bila fallback text-only.
            if self._past_kv is not None:
                system_prompt_tpl = qa_implement_prompts["evolving_strategy_coder_system_kv"]
                system_prompt_extra_render = {}
                logger.info("[LatentCoder] using KV-aware compact system prompt")
            else:
                system_prompt_tpl = qa_implement_prompts["evolving_strategy_factor_implementation_v1_system"]
                system_prompt_extra_render = {
                    "scenario": _mv("scenario", self.scen.get_scenario_all_desc(target_task, filtered_tag="feature")),
                }

            queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
            queried_similar_error_knowledge_to_render = queried_similar_error_knowledge

            # Ambil execution_log dan value_feedback dari last failed attempt.
            # code_feedback ditiadakan (tidak ada reviewer LLM lagi).
            last_fb = queried_former_failed_knowledge_to_render[-1].feedback
            execution_log = getattr(last_fb, "execution_feedback", None) or ""
            value_feedback = getattr(last_fb, "value_feedback", None) or ""

            # Error summary dan prior-attempt expression dihitung SEKALI di sini,
            # bukan di dalam token-budget loop (mencegah LLM call berulang per trimming iteration).
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

            latest_attempt_expr = ""
            if latest_attempt_to_latest_successful_execution is not None:
                latest_attempt_expr = self.extract_expr(
                    latest_attempt_to_latest_successful_execution.implementation.code
                )

            for _ in range(10):
                similar_successful_factor_description = ""
                similar_successful_expression = ""
                if len(queried_similar_successful_knowledge_to_render) > 0:
                    similar_successful_factor_description = queried_similar_successful_knowledge_to_render[-1].target_task.get_task_description()
                    similar_successful_expression = self.extract_expr(queried_similar_successful_knowledge_to_render[-1].implementation.code)

                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(
                        qa_implement_prompts["evolving_strategy_factor_implementation_v2_user"],
                    )
                    .render(
                        factor_information_str=_mv("factor_information_str", target_task.get_task_information()),
                        queried_similar_error_knowledge=_mv("queried_similar_error_knowledge", queried_similar_error_knowledge_to_render),
                        former_expression=_mv("former_expression", former_expr_raw),
                        execution_log=_mv("execution_log", execution_log),
                        value_feedback=_mv("value_feedback", value_feedback),
                        error_summary_critics=_mv("error_summary_critics", error_summary_critics),
                        similar_successful_factor_description=_mv("similar_successful_factor_description", similar_successful_factor_description),
                        similar_successful_expression=_mv("similar_successful_expression", similar_successful_expression),
                        latest_attempt_to_latest_successful_execution=_mv("latest_attempt_to_latest_successful_execution", latest_attempt_to_latest_successful_execution),
                        latest_attempt_expr=_mv("latest_attempt_expr", latest_attempt_expr),
                    )
                    .strip("\n")
                )

                # Token-budget check pakai system prompt attempt-1 (minimal)
                # sebagai proxy — varian lain panjangnya kira-kira sama.
                system_prompt_probe = (
                    Environment(undefined=StrictUndefined)
                    .from_string(system_prompt_tpl)
                    .render(attempt_mode=_ATTEMPT_MODES[0], **system_prompt_extra_render)
                )
                if (
                    self._get_backend().build_messages_and_calculate_token(
                        user_prompt=user_prompt, system_prompt=system_prompt_probe
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

            former_expr_norm = former_expr_raw.replace(" ", "").lower()

            # Set LENGKAP semua ekspresi yang pernah dicoba untuk task ini
            # (dari semua CoSTEER loop sebelumnya, bukan hanya former_expr_raw).
            # Ini mencegah cross-loop oscillation: model bergantian antara dua
            # ekspresi A↔B karena check lama hanya bandingkan dengan former_expr_norm
            # (yang berubah tiap loop), bukan dengan seluruh riwayat gagal.
            all_tried_exprs_norm: set = {former_expr_norm}
            for _k in queried_former_failed_knowledge:
                try:
                    _e = self.extract_expr(_k.implementation.code)
                    if _e:
                        all_tried_exprs_norm.add(_e.replace(" ", "").lower())
                except Exception:
                    pass
            if latest_attempt_expr:
                all_tried_exprs_norm.add(latest_attempt_expr.replace(" ", "").lower())
            logger.info(
                f"[LatentCoder] full-history set built: {len(all_tried_exprs_norm)} unique tried exprs"
            )

            # Snapshot KV-cache length SEBELUM retry loop. DynamicCache di-mutasi
            # in-place oleh latent_pass (append prompt + latent steps). Tanpa crop
            # per attempt, KV menumpuk dan model collapse ke "<think>" saja.
            # Baseline = construct_kv (_past_kv) langsung — eval_kv tidak ada lagi
            # karena reviewer/final_decision sudah dihapus.
            kv_baseline = self._past_kv
            kv_baseline_len = _past_length(kv_baseline) if kv_baseline is not None else 0

            mirror_hint = ""
            last_expr = None

            for attempt in range(_MAX_ATTEMPTS):
                # Reset KV ke baseline pre-retry: crop kembali ke panjang
                # baseline supaya attempt ini melihat KV yang sama dengan
                # attempt 1 — bukan KV yang sudah ter-append attempt sebelumnya.
                if kv_baseline is not None and hasattr(kv_baseline, "crop"):
                    try:
                        kv_baseline.crop(kv_baseline_len)
                    except Exception as crop_err:
                        logger.warning(
                            f"[LatentCoder] attempt {attempt+1}: kv_baseline.crop "
                            f"failed ({crop_err}); proceeding with current KV"
                        )
                self._past_kv = kv_baseline

                # Render system prompt sesuai attempt_mode (minimal/different/bold).
                system_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(system_prompt_tpl)
                    .render(attempt_mode=_ATTEMPT_MODES[attempt], **system_prompt_extra_render)
                )

                effective_user_prompt = user_prompt + mirror_hint

                raw = self._call_llm(
                    user_prompt=effective_user_prompt,
                    system_prompt=system_prompt,
                    json_mode=False,
                    reasoning_flag=False,
                    temperature=temp_schedule[attempt],
                )
                expr = self._parse_repair_output(raw)
                if expr is None:
                    logger.warning(
                        f"[LatentCoder] attempt {attempt+1}: failed to parse "
                        f"'PASS' or 'FIXED:' line from output (head=%r), retrying",
                        (raw or "").strip()[:160],
                    )
                    continue

                # PASS: LLM judge ekspresi sebenarnya sudah OK, biarkan apa adanya.
                if expr == self.PASS_SENTINEL:
                    logger.info(
                        f"[LatentCoder] attempt {attempt+1}: model returned PASS, "
                        f"keeping former_expr={former_expr_raw}"
                    )
                    return code_template.render(
                        expression=former_expr_raw,
                        factor_name=target_task.factor_name,
                    )

                expr_norm = expr.replace(" ", "").lower()
                # BUG-FIX: bandingkan dengan SEMUA ekspresi yang pernah dicoba
                # (all_tried_exprs_norm), bukan hanya former_expr_norm.
                # Ini menghentikan oscillasi A→B→A→B lintas CoSTEER loop.
                if expr_norm in all_tried_exprs_norm:
                    last_expr = expr
                    mirror_hint = (
                        f"\n\n**PREVIOUS ATTEMPT RETURNED EXPRESSION "
                        f"({expr}) WHICH HAS ALREADY BEEN TRIED AND FAILED "
                        f"IN A PREVIOUS ROUND — THIS IS A FAILURE. "
                        f"You MUST use a completely different operator family, "
                        f"window size, or base variable. Do NOT re-use any "
                        f"expression from prior rounds.**"
                    )
                    logger.warning(
                        f"[LatentCoder] attempt {attempt+1}: expr already tried before "
                        f"(expr={expr[:60]!r}), "
                        f"retrying with mode={_ATTEMPT_MODES[min(attempt+1, _MAX_ATTEMPTS-1)]}"
                    )
                    continue

                logger.info(
                    f"[LatentCoder] attempt {attempt+1} ({_ATTEMPT_MODES[attempt]}): "
                    f"new expr accepted"
                )
                return code_template.render(
                    expression=expr,
                    factor_name=target_task.factor_name,
                )

            # Fallback: semua attempt mirror/fail → pakai expr terakhir kalau ada,
            # else former_expr_raw. Evaluator akan reject kalau benar-benar bad.
            if last_expr is not None:
                logger.error(
                    f"[LatentCoder] all {_MAX_ATTEMPTS} attempts mirrored former_expr, "
                    f"falling back to last output: {last_expr}"
                )
                return code_template.render(
                    expression=last_expr,
                    factor_name=target_task.factor_name,
                )
            logger.error(f"[LatentCoder] all {_MAX_ATTEMPTS} attempts failed parse, using former_expr")
            return code_template.render(
                expression=former_expr_raw,
                factor_name=target_task.factor_name,
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
