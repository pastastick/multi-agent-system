"""
latent_mas/pipeline.py
====================
Orkestrator yang menyambung agent + DISTRIBUSI KV yang benar.

Inti modul ini bukan "memanggil agent berurutan", tapi **mengelola KV dengan
disiplin isolasi**. Aturan emas:

    Sebuah KV yang akan dibaca oleh > 1 konsumen HARUS di-`kv_deepcopy` dulu
    untuk tiap konsumen.

Kenapa wajib: `LocalLLMBackend.run()` memutasi `past_key_values` IN-PLACE
(latent_pass meng-extend objek DynamicCache yang dioper). Jadi kalau kv_consist
dioper apa adanya ke judger, objek kv_consist ikut ter-extend oleh prompt judger
— lalu saat feedback memakai kv_consist yang sama, ia melihat token judger.
Itu kontaminasi senyap. `kv_deepcopy` memutus rantai itu.

Aliran KV (sesuai desain):

  FRONT-END (sequential, in-place extension sah karena linear):
    seed → proposal(kv_only) → construct(kv_only) → consistency(kv_only)
                                                       └→ kv_consist (BASELINE)
  KONSUMEN kv_consist (masing-masing dapat CLONE sendiri):
    judger(kv_and_text)        ← deepcopy(kv_consist)   → teks final + parse
    repair(kv_and_text) ×N     ← deepcopy(kv_consist)   per attempt (baseline sama)
    feedback(kv_and_text)      ← deepcopy(kv_consist)   ← bukan kv_judger (anti-bias)

  EVOLUTION:
    mutation_reflection ← deepcopy(kv_feedback)         → diagnosis
    mutation_judger     ← (lanjut dari kv_reflect)      → mutated hypo+expr
    crossover_judger    ← kv_concat([deepcopy(p_i)...]) → recombined hypo+expr
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from llm.client import LocalLLMBackend, KVCache
from latent_mas import kv_ops
from latent_mas.agent import LatentAgent, AgentResult, load_all_agents
from latent_mas.parsers import HypothesisExpr, PASS_SENTINEL, MutationDiagnosis

# Type alias untuk gate: (expression) -> (ok, error_message)
QualityGate = Callable[[str], "tuple[bool, str]"]
# Type alias untuk backtest: (expression, hypothesis) -> dict hasil
Backtester = Callable[[str, str], dict]


def default_quality_gate(expression: str) -> "tuple[bool, str]":
    """Gate AST/arity deterministik. Lazy-import parser asli; fallback ke
    pengecekan dasar bila modul belum ada di branch ini."""
    if not expression or not expression.strip():
        return False, "empty expression"
    try:
        from factors.coder.expr_parser import parse_expression  # type: ignore
        parse_expression(expression)  # raises on invalid
        return True, ""
    except ImportError:
        # fallback ringan: cek kurung balance + variabel dikenal
        if expression.count("(") != expression.count(")"):
            return False, "unbalanced parentheses"
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrontEndOutput:
    hypothesis: str
    expression: str
    kv_consist: Optional[KVCache]      # baseline (pristine) untuk feedback/evolution
    kv_judger: Optional[KVCache]       # KV judger (untuk crossover antar-trajectory)
    judger_text: str
    repaired: bool = False
    repair_attempts: int = 0
    gate_error: str = ""


class FrontEndPipeline:
    """proposal → construct → consistency → judger → [gate → repair]."""

    def __init__(
        self,
        backend: LocalLLMBackend,
        *,
        runlog: Any = None,
        quality_gate: QualityGate = default_quality_gate,
        max_repair_attempts: int = 3,
        agents: Optional[dict] = None,
    ) -> None:
        self.backend = backend
        self.runlog = runlog
        self.gate = quality_gate
        self.max_repair_attempts = max_repair_attempts
        self.agents: dict = agents or load_all_agents(backend, runlog=runlog)

    def _a(self, name: str) -> LatentAgent:
        return self.agents[name]

    def run(
        self,
        *,
        direction: str,
        seed_kv: Optional[KVCache] = None,
        market_context: str = "",
        prior_feedback: str = "",
    ) -> FrontEndOutput:
        rl = self.runlog

        # ── sequential latent chain (in-place extension OK: linear) ──────────
        r_prop = self._a("proposal").run(
            past_kv=seed_kv, direction=direction,
            market_context=market_context, prior_feedback=prior_feedback,
        )
        r_con = self._a("construct").run(past_kv=r_prop.kv_cache)
        r_cons = self._a("consistency").run(past_kv=r_con.kv_cache)

        kv_consist = r_cons.kv_cache          # BASELINE — jaga tetap pristine

        # ── judger membaca CLONE dari baseline ───────────────────────────────
        r_judge = self._a("judger").run(past_kv=kv_ops.kv_deepcopy(kv_consist))
        he: Optional[HypothesisExpr] = r_judge.parsed
        if he is None:
            if rl: rl.warn("judger output unparseable", head=(r_judge.text or "")[:120])
            hypothesis, expression = "", ""
        else:
            hypothesis, expression = he.hypothesis, he.expression

        out = FrontEndOutput(
            hypothesis=hypothesis, expression=expression,
            kv_consist=kv_consist, kv_judger=r_judge.kv_cache,
            judger_text=r_judge.text or "",
        )

        # ── quality gate + repair (logika dipakai ulang oleh evolution) ──────
        out.expression, out.repaired, out.repair_attempts, out.gate_error = \
            self._gate_and_repair(expression, kv_consist)
        return out

    def _gate_and_repair(
        self,
        expression: str,
        kv_baseline: Optional[KVCache],
    ) -> "tuple[str, bool, int, str]":
        """Gate ekspresi; jika gagal jalankan repair (≤ max attempts), tiap attempt
        berangkat dari CLONE pristine kv_baseline. Dipakai jalur front-end
        (original) MAUPUN evolution (mutation/crossover) agar kualitas seragam.

        Returns: (final_expression, repaired, attempts, gate_error).
        Pada exhaustion, kembalikan ekspresi ASAL (perilaku sama dengan front-end lama).
        """
        rl = self.runlog
        if expression:
            ok, err = self.gate(expression)
        else:
            ok, err = False, "no expression"
        if ok:
            return expression, False, 0, ""

        gate_error = err
        tried = {expression.replace(" ", "").lower()} if expression else set()
        modes = ["minimal", "different", "bold"]
        former = expression

        for attempt in range(self.max_repair_attempts):
            mode = modes[min(attempt, len(modes) - 1)]
            r_rep = self._a("repair").run(
                past_kv=kv_ops.kv_deepcopy(kv_baseline),
                former_expression=former, error_log=err,
                value_feedback="", attempt_mode=mode,
            )
            parsed = r_rep.parsed
            if parsed is None:
                if rl: rl.warn(f"repair attempt {attempt+1} unparseable")
                continue
            if parsed == PASS_SENTINEL:
                if rl: rl.info("repair returned PASS; keeping expression")
                return former, True, attempt + 1, gate_error
            norm = parsed.replace(" ", "").lower()
            if norm in tried:
                if rl: rl.warn(f"repair attempt {attempt+1} repeated a tried expr")
                former = parsed
                continue
            ok2, err2 = self.gate(parsed)
            if ok2:
                return parsed, True, attempt + 1, gate_error
            tried.add(norm)
            former, err = parsed, err2

        if rl: rl.error("repair exhausted; keeping original expression", expr=expression)
        return expression, False, self.max_repair_attempts, gate_error


# ─────────────────────────────────────────────────────────────────────────────
# Evolution operators
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvolutionOutput:
    """Hasil mutation/crossover: hipotesis+ekspresi + KV judger (untuk feedback)."""
    hypothesis: str
    expression: str
    kv: Optional[KVCache]          # KV judger evolution → baseline feedback & next round
    raw_text: str = ""


class EvolutionOps:
    """Mutation (sequential 2-step) + Crossover (hierarchical KV concat)."""

    def __init__(self, backend: LocalLLMBackend, *, runlog: Any = None,
                 agents: Optional[dict] = None) -> None:
        self.backend = backend
        self.runlog = runlog
        self.agents: dict = agents or load_all_agents(backend, runlog=runlog)

    def _a(self, name: str) -> LatentAgent:
        return self.agents[name]

    # ── Mutation: reflection (diagnose) → judger (rewrite) ───────────────────
    def mutate(
        self,
        *,
        parent_kv_feedback: Optional[KVCache],
        parent_hypothesis: str,
        parent_expression: str,
        parent_feedback: str,
        backtest_summary: str,
    ) -> Optional[EvolutionOutput]:
        r_ref = self._a("mutation_reflection").run(
            past_kv=kv_ops.kv_deepcopy(parent_kv_feedback),
            parent_hypothesis=parent_hypothesis,
            parent_expression=parent_expression,
            parent_feedback=parent_feedback,
            backtest_summary=backtest_summary,
        )
        diag: MutationDiagnosis = r_ref.parsed or MutationDiagnosis("construct", "")
        # judger melanjutkan dari KV reflection (sequential) — sudah berisi
        # konteks parent + diagnosis.
        r_mut = self._a("mutation_judger").run(
            past_kv=r_ref.kv_cache,
            diagnosis_step=diag.failure_step,
            diagnosis_reason=diag.reason,
        )
        he: Optional[HypothesisExpr] = r_mut.parsed
        if he is None:
            return None
        return EvolutionOutput(he.hypothesis, he.expression, r_mut.kv_cache, r_mut.text or "")

    # ── Crossover: concat k parent KV (hierarchical) → judger ────────────────
    def crossover(
        self,
        *,
        parent_kvs: List[Optional[KVCache]],
    ) -> Optional[EvolutionOutput]:
        # clone tiap parent lalu concat layer-wise (LatentMAS hierarchical).
        clones = [kv_ops.kv_deepcopy(kv) for kv in parent_kvs]
        merged = kv_ops.kv_concat(clones)
        if self.runlog:
            self.runlog.info("crossover KV merged",
                             n_parents=len([k for k in clones if k is not None]),
                             merged=kv_ops.kv_describe(merged))
        r_cross = self._a("crossover_judger").run(
            past_kv=merged, n_parents=len([k for k in parent_kvs if k is not None]),
        )
        he: Optional[HypothesisExpr] = r_cross.parsed
        if he is None:
            return None
        return EvolutionOutput(he.hypothesis, he.expression, r_cross.kv_cache, r_cross.text or "")
