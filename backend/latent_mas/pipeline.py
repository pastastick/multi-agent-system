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
from pathlib import Path
from typing import Any, Callable, List, Optional

from llm.client import LocalLLMBackend, KVCache
from latent_mas import kv_ops
from latent_mas.agent import LatentAgent, AgentResult, load_all_agents
from latent_mas.parsers import (
    HypothesisExpr, HypothesisExprs, PASS_SENTINEL, MutationDiagnosis,
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
        r_judge = self._a("judger").run(past_kv=kv_ops.kv_deepcopy(kv_consist))
        he: Optional[HypothesisExprs] = r_judge.parsed
        if he is None:
            if rl: rl.warn("judger output unparseable", head=(r_judge.text or "")[:120])
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

        gate_err = self.gate(candidates[0])[1] if candidates else "no expression from judger"
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


# ─────────────────────────────────────────────────────────────────────────────
# Evolution operators
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvolutionSeed:
    """Guidance KV dari evolution (kv_only) untuk MENYEMAI re-entry ORIGINAL.

    Tidak membawa (hypo, expr): di desain ini evolution hanya memberi *sinyal arah*
    di ruang laten; judger ORIGINAL (yang di-reentry dengan `kv` sebagai seed) yang
    menghasilkan factor — ini instantiasi "regenerate from node k" (paper Eq. 6/7).

      kv         : KV untuk dijadikan seed_kv FrontEndPipeline.run.
      debug_text : rekonstruksi teks isi KV (hanya terisi bila debug aktif).
    """
    kv: Optional[KVCache]
    debug_text: str = ""


class EvolutionOps:
    """Evolution = hasilkan GUIDANCE KV (kv_only), bukan factor.

    Mutation  : mutation_reflection(kv_only) mendiagnosa node gagal → guidance KV.
    Crossover : kv_concat(parent KV) → crossover_judger(kv_only) → guidance KV.
    Re-entry ORIGINAL dilakukan oleh caller (loop) via FrontEndPipeline.run(seed_kv=...).

    Debug: karena kedua agent kv_only (tak emit teks), set `debug=True` untuk men-
    decode guidance KV jadi teks (probe `introspect`) + menyimpannya ke `.pt`/`.txt`
    di `debug_dir`. Produksi (debug=False) tetap murni-laten & tanpa pass ekstra.
    """

    def __init__(self, backend: LocalLLMBackend, *, runlog: Any = None,
                 agents: Optional[dict] = None,
                 debug: bool = False, debug_dir: Optional[Path] = None) -> None:
        self.backend = backend
        self.runlog = runlog
        self.agents: dict = agents or load_all_agents(backend, runlog=runlog)
        self.debug = debug
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self._dbg_idx = 0

    def _a(self, name: str) -> LatentAgent:
        return self.agents[name]

    # ── Mutation step-1: reflection (kv_only) → guidance KV ───────────────────
    def reflect(
        self,
        *,
        parent_kv: Optional[KVCache],
        parent_hypothesis: str,
        parent_expression: str,
        parent_feedback: str,
        backtest_summary: str,
    ) -> EvolutionSeed:
        r_ref = self._a("mutation_reflection").run(
            past_kv=kv_ops.kv_deepcopy(parent_kv),
            parent_hypothesis=parent_hypothesis,
            parent_expression=parent_expression,
            parent_feedback=parent_feedback,
            backtest_summary=backtest_summary,
        )
        dbg = self._debug_decode(r_ref.kv_cache, "mutation_reflection")
        return EvolutionSeed(kv=r_ref.kv_cache, debug_text=dbg)

    # ── Crossover: concat k parent KV (hierarchical) → crossover_judger (kv_only)
    def synthesize(
        self,
        parent_kvs: List[Optional[KVCache]],
    ) -> EvolutionSeed:
        """`parent_kvs` diasumsikan SUDAH diurut (parent terbaik di posisi TERAKHIR
        agar mendapat bias recency saat di-concat)."""
        clones = [kv_ops.kv_deepcopy(kv) for kv in parent_kvs]
        merged = kv_ops.kv_concat(clones)
        if merged is None:
            return EvolutionSeed(kv=None)
        if self.runlog:
            self.runlog.info("crossover KV merged",
                             n_parents=len([k for k in clones if k is not None]),
                             merged=kv_ops.kv_describe(merged))
        r_cross = self._a("crossover_judger").run(
            past_kv=merged, n_parents=len([k for k in parent_kvs if k is not None]),
        )
        dbg = self._debug_decode(r_cross.kv_cache, "crossover_judger")
        return EvolutionSeed(kv=r_cross.kv_cache, debug_text=dbg)

    # ── Debug-only: decode guidance KV → teks (probe) + simpan .pt/.txt ───────
    def _debug_decode(self, kv: Optional[KVCache], label: str) -> str:
        if not self.debug or kv is None:
            return ""
        text = ""
        try:
            probe = self._a("introspect").run(past_kv=kv_ops.kv_deepcopy(kv))
            text = probe.text or ""
        except Exception as e:  # noqa: BLE001
            if self.runlog:
                self.runlog.warn(f"debug-decode {label} failed", err=repr(e))
        if self.debug_dir is not None:
            try:
                self._dbg_idx += 1
                stem = f"{self._dbg_idx:03d}_{label}"
                kv_ops.kv_save(kv, self.debug_dir / f"{stem}.pt",
                               metadata={"label": label, "decoded_text": text})
                (self.debug_dir / f"{stem}.txt").write_text(text)
                if self.runlog:
                    self.runlog.info("debug-decode saved",
                                     path=str(self.debug_dir / f"{stem}.pt"))
            except Exception as e:  # noqa: BLE001
                if self.runlog:
                    self.runlog.warn(f"debug-save {label} failed", err=repr(e))
        return text
