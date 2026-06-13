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

  EVOLUTION (GUIDANCE kv_only → re-entry front-end, bounded per-generasi):
    mutation   (kv_only) ← past_kv=None + TEKS 1 parent → guidance_kv (arah refine)
    crossover  (kv_only) ← past_kv=None + TEKS k parent → guidance_kv (arah fusi)
  guidance_kv lalu MENYEMAI front-end: run(seed_kv=guidance_kv) →
    proposal → construct → consistency → judger → gate/repair (lihat run_evolution).
  Materi parent masuk sebagai TEKS & KV TIDAK diwariskan antar-generasi (trajectory
  .kv_cache tetap None) → KV per ronde terbatas (~guidance + front-end ≈ 3k) → tak
  ada over-KV / collapse lintas-generasi (akar regresi 2026-06-02).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from llm.client import LocalLLMBackend, KVCache
from latent_mas import kv_ops
from latent_mas.agent import LatentAgent, AgentResult, load_all_agents
from latent_mas.parsers import (
    HypothesisExprs, PASS_SENTINEL,
    parse_repair_multi,
)

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
    expressions: List[str]             # SEMUA ekspresi lolos regulator (≥1) → LightGBM combined
    kv_consist: Optional[KVCache]      # baseline (pristine) — dipakai judger & repair
    kv_judger: Optional[KVCache]       # KV judger
    judger_text: str
    repaired: bool = False
    repair_attempts: int = 0
    gate_error: str = ""
    # KV yang BENAR-BENAR menghasilkan ekspresi final yang diterima:
    #   kv_repair bila repair berjalan & sukses, selain itu kv_judger.
    # Inilah seed feedback (trajectory-coherent) — lihat loop._run_latent_feedback.
    kv_final: Optional[KVCache] = None

    @property
    def expression(self) -> str:
        """Ekspresi pertama (untuk logging/penamaan; back-compat single-expr)."""
        return self.expressions[0] if self.expressions else ""


class FrontEndPipeline:
    """proposal → construct → consistency → judger → [regulator-gate → repair].

    Gate = FactorRegulator PENUH (parsable + complexity SL/PC/ER + redundansi
    alpha-zoo) bila tersedia, fallback ke `default_quality_gate` (sintaks AST).
    Judger boleh keluarkan N ekspresi; tiap ekspresi di-gate. Aturan repair:
    hanya bila SEMUA ekspresi gagal (repair pun hasilkan N ekspresi).
    """

    def __init__(
        self,
        backend: LocalLLMBackend,
        *,
        runlog: Any = None,
        quality_gate: Optional[QualityGate] = None,
        max_repair_attempts: int = 3,
        agents: Optional[dict] = None,
        use_regulator: bool = True,
    ) -> None:
        self.backend = backend
        self.runlog = runlog
        self.max_repair_attempts = max_repair_attempts
        self.agents: dict = agents or load_all_agents(backend, runlog=runlog)
        self._regulator = None
        if quality_gate is not None:
            self.gate = quality_gate
        elif use_regulator:
            self.gate, self._regulator = self._build_regulator_gate(runlog)
        else:
            self.gate = default_quality_gate

    def _a(self, name: str) -> LatentAgent:
        return self.agents[name]

    @staticmethod
    def _build_regulator_gate(runlog: Any = None) -> "tuple[QualityGate, Any]":
        """Bangun gate berbasis FactorRegulator PENUH (dari FACTOR_COSTEER_SETTINGS).
        Fallback ke default_quality_gate bila modul/zoo tak tersedia."""
        try:
            # Pre-import factor_ast lebih dulu: memutus circular import
            # (factor_regulator → coder/__init__ → evaluators → factor_regulator)
            # yang terjadi bila factor_regulator di-import COLD. Di jalur latent,
            # FrontEndPipeline dibangun SEBELUM coder di-instansiasi, jadi tanpa ini
            # gate diam-diam fallback ke sintaks (regulator tak aktif).
            import factors.coder.factor_ast  # noqa: F401
            from factors.regulator.factor_regulator import FactorRegulator
            from factors.coder.config import FACTOR_COSTEER_SETTINGS as S
            reg = FactorRegulator(
                factor_zoo_path=getattr(S, "factor_zoo_path", None),
                duplication_threshold=getattr(S, "duplication_threshold", 8),
                symbol_length_threshold=getattr(S, "symbol_length_threshold", 300),
                base_features_threshold=getattr(S, "base_features_threshold", 6),
            )
        except Exception as e:  # noqa: BLE001
            if runlog:
                runlog.warn("FactorRegulator unavailable; fallback to syntax gate",
                            err=repr(e))
            return default_quality_gate, None

        def gate(expr: str) -> "tuple[bool, str]":
            if not expr or not expr.strip():
                return False, "empty expression"
            try:
                if not reg.is_parsable(expr):
                    return False, "unparsable expression"
                ok, ev = reg.evaluate(expr)
                if not ok or ev is None:
                    return False, "regulator evaluate failed"
                if not reg.is_expression_acceptable(ev):
                    return False, (f"regulator reject: sl={ev.get('symbol_length')}, "
                                   f"base_feat={ev.get('num_base_features')}, "
                                   f"dup={ev.get('duplicated_subtree_size')}")
                return True, ""
            except Exception as e:  # noqa: BLE001
                return False, f"{type(e).__name__}: {e}"

        return gate, reg

    def _register_factors(self, exprs: List[str]) -> None:
        """Daftarkan ekspresi lolos ke alpha-zoo regulator (dedup intra-run)."""
        reg = self._regulator
        if reg is None or not exprs:
            return
        try:
            names = [f"latent_{i}" for i in range(len(exprs))]
            reg.add_factor(names, exprs)
        except Exception as e:  # noqa: BLE001
            if self.runlog:
                self.runlog.warn("regulator add_factor failed", err=repr(e))

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

        # ── judger membaca CLONE dari baseline; boleh keluarkan N ekspresi ────
        # Output judger kadang degenerate (sampling dari KV berisi virtual token
        # itu borderline-stabil). Retry sekali dari baseline pristine jauh lebih
        # murah (~10s) daripada membiarkan kandidat kosong jatuh ke repair.
        he: Optional[HypothesisExprs] = None
        for attempt in range(2):
            r_judge = self._a("judger").run(
                past_kv=kv_ops.kv_deepcopy(kv_consist), direction=direction,
            )
            he = r_judge.parsed
            if he is not None:
                break
            if rl: rl.warn(
                f"judger output unparseable (attempt {attempt + 1}/2)",
                head=(r_judge.text or "")[:120],
            )
        if he is None:
            hypothesis, candidates = "", []
        else:
            hypothesis, candidates = he.hypothesis, list(he.expressions)

        # ── regulator-gate semua ekspresi; repair hanya bila SEMUA gagal ──────
        passing, kv_final, repaired, attempts, gate_err = self._gate_and_repair_multi(
            candidates, kv_consist, r_judge.kv_cache,
        )
        return FrontEndOutput(
            hypothesis=hypothesis, expressions=passing,
            kv_consist=kv_consist, kv_judger=r_judge.kv_cache,
            judger_text=r_judge.text or "", kv_final=kv_final,
            repaired=repaired, repair_attempts=attempts, gate_error=gate_err,
        )

    def run_evolution(
        self,
        *,
        kind: str,                       # "mutation" | "crossover"
        parent_text: str,
        n_parents: int = 1,
        direction: str = "",
    ) -> FrontEndOutput:
        """Evolution GUIDANCE → re-entry front-end (bounded per-generasi).

          mutation  : arahkan refine SATU target (eksploitasi).
          crossover : arahkan fusi k parent (eksplorasi).

        Agent guidance (`mutation`/`crossover`, mode **kv_only**) di-seed dari
        **None** dan membaca materi parent sebagai **TEKS** (`parent_text`); ia
        bernalar laten untuk menetapkan ARAH (diagnosis + direction) TANPA menulis
        ekspresi. KV-nya (`guidance_kv`) lalu MENYEMAI front-end via
        `run(seed_kv=guidance_kv)` → proposal→construct→consistency→judger→
        gate/repair yang menyusun ekspresinya. Downstream (backtest/feedback) tak
        berubah; output tetap `FrontEndOutput`.

        Karena guidance di-seed None + parent=teks, dan `trajectory.kv_cache` tetap
        None (transfer antar-generasi via teks `_format_parents_text`), KV per ronde
        terbatas (~guidance + front-end ≈ 3k) — tak ada warisan/akumulasi KV parent,
        jadi tak ada over-KV maupun collapse lintas-generasi (akar regresi 2026-06-02).
        """
        rl = self.runlog
        if kind == "mutation":
            agent_name = "mutation"
            kw = dict(target_text=parent_text, direction=direction)
        elif kind == "crossover":
            agent_name = "crossover"
            kw = dict(parents_text=parent_text, n_parents=n_parents, direction=direction)
        else:
            raise ValueError(f"unknown evolution kind: {kind!r}")

        # ── 1. GUIDANCE (kv_only, seed=None): parent sebagai TEKS → arah laten ──
        # past_kv=None → tak warisi KV parent (RESET tiap generasi). Output tak
        # di-decode (kv_only); yang dipakai HANYA KV-nya sebagai seed proposal.
        r_guide = self._a(agent_name).run(past_kv=None, **kw)
        guidance_kv = r_guide.kv_cache
        if guidance_kv is None and rl:
            rl.warn(f"{kind} guidance produced no KV; front-end runs unseeded")

        # ── 1b. (opsional) decode guidance KV untuk inspeksi arah yang dipilih ──
        # deepcopy: introspect meng-extend KV in-place → jaga guidance_kv pristine
        # untuk konsumen sebenarnya (proposal).
        if guidance_kv is not None and os.environ.get("LATENTMAS_EVO_DEBUG"):
            try:
                probe = self._a("introspect").run(past_kv=kv_ops.kv_deepcopy(guidance_kv))
                if rl: rl.info(f"{kind} guidance (probe)", head=(probe.text or "")[:400])
            except Exception as e:  # noqa: BLE001
                if rl: rl.warn("evo guidance probe failed", err=repr(e))

        # ── 2. RE-ENTER front-end di-seed guidance_kv (1 konsumen → no deepcopy) ─
        # proposal(past_kv=guidance_kv) → construct → consistency → judger →
        # _gate_and_repair_multi. FrontEndOutput sama seperti jalur original.
        return self.run(direction=direction, seed_kv=guidance_kv)

    def _gate_and_repair_multi(
        self,
        candidates: List[str],
        kv_baseline: Optional[KVCache],
        kv_judger: Optional[KVCache],
    ) -> "tuple[List[str], Optional[KVCache], bool, int, str]":
        """Gate tiap ekspresi via regulator. Aturan (sesuai keputusan user):
          - ada ≥1 lolos → pakai yang lolos, TANPA repair. kv_final = kv_judger.
          - SEMUA gagal → repair (≤max attempts), repair pun hasilkan N ekspresi;
            gate ulang, pakai yang lolos. kv_final = kv_repair.
          - repair habis tetap gagal → kembalikan kandidat asal (pipeline tak
            dead-end; backtest yang akan menyaring). kv_final = kv_judger.

        Returns: (passing_exprs, kv_final, repaired, attempts, gate_error).
        """
        rl = self.runlog
        passing = [e for e in candidates if self.gate(e)[0]]
        if passing:
            self._register_factors(passing)
            return passing, kv_judger, False, 0, ""

        # Tanpa kandidat tak ada yang bisa diperbaiki: repair dengan
        # former_expression kosong terbukti menghasilkan sampah (run
        # 20260608_064005) dan membuang ~60s per ronde.
        if not candidates:
            if rl: rl.error("no expression from judger; skipping repair")
            return [], kv_judger, False, 0, "no expression from judger"

        gate_err = self.gate(candidates[0])[1]
        former = candidates
        err = gate_err
        for attempt in range(self.max_repair_attempts):
            mode = ["minimal", "different", "bold"][min(attempt, 2)]
            r_rep = self._a("repair").run(
                past_kv=kv_ops.kv_deepcopy(kv_baseline),
                former_expression="; ".join(former) if former else "",
                error_log=err, value_feedback="", attempt_mode=mode,
            )
            is_pass, rep_exprs = parse_repair_multi(r_rep.text or "")
            if is_pass:
                # PASS = model klaim valid; tapi gate kita deterministik → gate ulang.
                passing = [e for e in former if self.gate(e)[0]]
                if passing:
                    if rl: rl.info("repair PASS confirmed by gate")
                    self._register_factors(passing)
                    return passing, r_rep.kv_cache, True, attempt + 1, gate_err
                continue
            if not rep_exprs:
                if rl: rl.warn(f"repair attempt {attempt+1} produced no expression")
                continue
            passing = [e for e in rep_exprs if self.gate(e)[0]]
            if passing:
                self._register_factors(passing)
                return passing, r_rep.kv_cache, True, attempt + 1, gate_err
            former = rep_exprs
            err = self.gate(rep_exprs[0])[1]

        if rl: rl.error("multi-repair exhausted; keeping original candidates",
                        n=len(candidates))
        return candidates, kv_judger, False, self.max_repair_attempts, gate_err

    def _gate_and_repair(
        self,
        expression: str,
        kv_baseline: Optional[KVCache],
    ) -> "tuple[str, bool, int, str, Optional[KVCache]]":
        """Gate ekspresi; jika gagal jalankan repair (≤ max attempts), tiap attempt
        berangkat dari CLONE pristine kv_baseline.

        Returns: (final_expression, repaired, attempts, gate_error, kv_repair).
        `kv_repair` = KV dari attempt repair yang diterima (untuk dijadikan kv_final),
        atau None bila gate langsung lolos / repair gagal (caller pakai kv_judger).
        """
        rl = self.runlog
        if expression:
            ok, err = self.gate(expression)
        else:
            ok, err = False, "no expression"
        if ok:
            return expression, False, 0, "", None

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
                return former, True, attempt + 1, gate_error, r_rep.kv_cache
            norm = parsed.replace(" ", "").lower()
            if norm in tried:
                if rl: rl.warn(f"repair attempt {attempt+1} repeated a tried expr")
                former = parsed
                continue
            ok2, err2 = self.gate(parsed)
            if ok2:
                return parsed, True, attempt + 1, gate_error, r_rep.kv_cache
            tried.add(norm)
            former, err = parsed, err2

        if rl: rl.error("repair exhausted; keeping original expression", expr=expression)
        return expression, False, self.max_repair_attempts, gate_error, None
