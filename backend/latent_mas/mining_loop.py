"""
latent_mas/mining_loop.py
====================
Loop mining baru: FrontEndPipeline (generasi laten) → SUBSTRAT lama (backtest).

Ini pengganti `pipeline/loop.py::AlphaAgentLoop` dengan arsitektur LatentMAS.
Prinsip (sesuai keputusan): **pakai ulang substrat eksekusi faktor** —
FactorTask / QlibFactorExperiment / template.jinjia2 / runner / library —
dan hanya GANTI lapisan generasi-LLM dengan `latent_mas`.

Alur satu iterasi:

    FrontEndPipeline.run(direction, seed_kv)
        → (hypothesis, expression, kv_consist, kv_judger)         [latent_mas]
    bridge:
        FactorTask(factor_expression=expression)
        → code_template.render → FactorFBWorkspace.inject_code
        → QlibFactorExperiment(sub_tasks=[task], based_experiments=SOTA)  [substrat]
    runner.develop(exp, use_local) → exp.result                   [backtest Qlib]
    feedback agent (deepcopy(kv_consist)) → feedback JSON          [latent_mas]
    FactorLibraryManager.add_factors_from_experiment              [substrat]

Catatan: modul mengimpor substrat (rdagent/qlib) secara LAZY di dalam method
agar bagian latent bisa diimpor tanpa qlib. BELUM diverifikasi di GPU/Qlib —
itu langkah berikutnya sebelum kode lama dihapus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from llm.client import LocalLLMBackend, KVCache
from latent_mas import kv_ops
from latent_mas.pipeline import FrontEndPipeline, FrontEndOutput, default_quality_gate
from latent_mas.parsers import HypothesisExpr


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IterationResult:
    hypothesis: str
    expression: str
    experiment: Any                 # QlibFactorExperiment (dengan .result terisi)
    feedback: dict                  # parsed feedback JSON
    feedback_text: str
    kv_consist: Optional[KVCache]   # baseline untuk evolution
    kv_judger: Optional[KVCache]    # untuk crossover antar-trajectory
    repaired: bool = False
    backtest_ok: bool = False
    replace_sota: bool = False


def _slug(text: str, fallback: str) -> str:
    """factor_name yang aman untuk filesystem/qlib dari teks hipotesis."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip())[:40].strip("_")
    return s or fallback


# ─────────────────────────────────────────────────────────────────────────────

class MiningLoop:
    """Loop mining LatentMAS — front-end laten + backtest substrat lama."""

    def __init__(
        self,
        backend: LocalLLMBackend,
        *,
        scenario: Any,
        runner: Any,
        runlog: Any = None,
        use_local: bool = True,
        library_path: Optional[str] = None,
        max_repair_attempts: int = 3,
    ) -> None:
        self.backend = backend
        self.scenario = scenario
        self.runner = runner
        self.runlog = runlog
        self.use_local = use_local
        self.library_path = library_path

        self.front = FrontEndPipeline(
            backend, runlog=runlog,
            quality_gate=self._quality_gate,
            max_repair_attempts=max_repair_attempts,
        )
        # history: list of (hypothesis, experiment, feedback) seperti trace.hist lama,
        # dipakai membangun based_experiments (SOTA).
        self.history: List[tuple] = []

    @classmethod
    def from_settings(cls, backend: LocalLLMBackend, setting: Any, *,
                      runlog: Any = None, use_local: bool = True) -> "MiningLoop":
        """Bangun scenario + runner dari PROP_SETTING (seperti loop.py lama)."""
        from core.utils import import_class
        scen = import_class(setting.scen)(use_local=use_local)
        runner = import_class(setting.runner)(scen)
        return cls(backend, scenario=scen, runner=runner,
                   runlog=runlog, use_local=use_local)

    # ── quality gate: AST/arity + sinyal kompleksitas ────────────────────────

    def _quality_gate(self, expression: str) -> "tuple[bool, str]":
        return default_quality_gate(expression)

    @staticmethod
    def _complexity_note(expression: str) -> str:
        """Sinyal kompleksitas (constraint gate QuantaAlpha) untuk feedback."""
        try:
            from factors.regulator.consistency_checker import ComplexityChecker
            ok, msg = ComplexityChecker().check(expression)
            return "" if ok else f"COMPLEXITY WARNING: {msg}"
        except Exception:
            n = len(expression or "")
            return f"COMPLEXITY WARNING: expression length {n} > 250" if n > 250 else ""

    # ── bridge: (hypothesis, expression) → QlibFactorExperiment ──────────────

    def _build_experiment(self, hypothesis: str, expression: str, idx: int) -> Any:
        from factors.coder.factor import FactorTask, FactorFBWorkspace
        from factors.experiment import QlibFactorExperiment
        from factors.coder.evolving_strategy import code_template

        factor_name = _slug(hypothesis, f"latent_factor_{idx}")
        task = FactorTask(
            factor_name=factor_name,
            factor_description=hypothesis,
            factor_formulation=expression,
            factor_expression=expression,
            variables={},
        )
        exp = QlibFactorExperiment([task])
        # based_experiments = SOTA dari history (faktor yang feedback-nya ada) —
        # pola identik dengan proposal.convert_response lama.
        exp.based_experiments = (
            [QlibFactorExperiment(sub_tasks=[])]
            + [h[1] for h in self.history if h[2]]
        )

        # render kode dari ekspresi → inject ke workspace (pola coder lama).
        code = code_template.render(expression=expression, factor_name=factor_name)
        ws = FactorFBWorkspace(target_task=task)
        ws.inject_code(**{"factor.py": code})
        exp.sub_workspace_list = [ws]
        return exp

    # ── format konteks feedback ──────────────────────────────────────────────

    @staticmethod
    def _format_result(exp: Any) -> str:
        res = getattr(exp, "result", None)
        if res is None:
            return "backtest produced no result (factor may have failed to execute)"
        try:
            return res.to_string()
        except Exception:
            return str(res)

    def _sota_block(self) -> str:
        if not self.history:
            return "none yet"
        # ambil hypothesis+result terakhir yang punya feedback sebagai SOTA proxy
        for hyp, exp, fb in reversed(self.history):
            if fb is not None:
                return f"Hypothesis: {hyp}\nResult:\n{self._format_result(exp)}"
        return "none yet"

    # ── satu iterasi penuh ───────────────────────────────────────────────────

    def run_iteration(
        self,
        *,
        direction: str,
        seed_kv: Optional[KVCache] = None,
        idx: int = 0,
        market_context: str = "",
        prior_feedback: str = "",
    ) -> IterationResult:
        from llm._shared import robust_json_parse

        rl = self.runlog
        front: FrontEndOutput = self.front.run(
            direction=direction, seed_kv=seed_kv,
            market_context=market_context, prior_feedback=prior_feedback,
        )
        if not front.expression:
            if rl: rl.error("front-end produced no expression; skipping backtest")
            return IterationResult(
                hypothesis=front.hypothesis, expression="", experiment=None,
                feedback={}, feedback_text="", kv_consist=front.kv_consist,
                kv_judger=front.kv_judger, repaired=front.repaired,
            )

        # bridge + backtest (substrat lama)
        exp = self._build_experiment(front.hypothesis, front.expression, idx)
        backtest_ok = True
        with (rl.step("backtest") if rl else _null()):
            try:
                exp = self.runner.develop(exp, use_local=self.use_local)
            except Exception as e:  # noqa: BLE001
                backtest_ok = False
                if rl: rl.error("backtest failed", err=repr(e))

        # feedback (latent_mas) — baca CLONE dari kv_consist (anti-bias kv_judger)
        complexity = self._complexity_note(front.expression)
        factor_block = (
            f"- {_slug(front.hypothesis, 'factor')}: {front.hypothesis}\n"
            f"  Expression: {front.expression}"
        )
        if complexity:
            factor_block += f"\n  {complexity}"
        fb_agent = self.front.agents["feedback"]
        r_fb = fb_agent.run(
            past_kv=kv_ops.kv_deepcopy(front.kv_consist),
            hypothesis_text=front.hypothesis,
            factor_block=factor_block,
            backtest_results=self._format_result(exp),
            sota_block=self._sota_block(),
        )
        feedback_text = r_fb.text or ""
        try:
            feedback = robust_json_parse(feedback_text) if feedback_text else {}
        except Exception:
            feedback = {"Observations": feedback_text}

        replace = str(feedback.get("Replace Best Result", "no")).strip().lower().startswith("y")

        # update history + library
        self.history.append((front.hypothesis, exp, feedback))
        self._save_to_library(exp, front.hypothesis, feedback)

        return IterationResult(
            hypothesis=front.hypothesis, expression=front.expression,
            experiment=exp, feedback=feedback, feedback_text=feedback_text,
            kv_consist=front.kv_consist, kv_judger=front.kv_judger,
            repaired=front.repaired, backtest_ok=backtest_ok, replace_sota=replace,
        )

    def run(self, *, direction: str, n_iterations: int = 5,
            seed_kv: Optional[KVCache] = None) -> List[IterationResult]:
        """Jalankan beberapa iterasi berurutan (propose→…→feedback × n)."""
        results: List[IterationResult] = []
        prior_fb = ""
        for i in range(n_iterations):
            if self.runlog: self.runlog.info(f"iteration {i+1}/{n_iterations}", direction=direction)
            res = self.run_iteration(
                direction=direction, seed_kv=seed_kv, idx=i, prior_feedback=prior_fb,
            )
            results.append(res)
            prior_fb = res.feedback.get("New Hypothesis", "") if res.feedback else ""
            seed_kv = None  # antar-iterasi mulai fresh (KV via text trace, bukan KV chain)
        return results

    def _save_to_library(self, exp: Any, hypothesis: str, feedback: dict) -> None:
        if not self.library_path:
            return
        try:
            from factors.library import FactorLibraryManager
            mgr = FactorLibraryManager(self.library_path)
            mgr.add_factors_from_experiment(
                experiment=exp, experiment_id="latent", round_number=0,
                hypothesis=hypothesis, feedback=feedback,
            )
        except Exception as e:  # noqa: BLE001
            if self.runlog: self.runlog.warn("library save failed", err=repr(e))


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False
