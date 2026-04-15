"""
insight.py — External Agent Orchestrator
==========================================

Penghubung antara folder eksternal (agent-agent pencari data)
dan pipeline internal (planning.py, loop.py, factor_mining.py).

Dua mode orkestrasi:

  SEQUENTIAL (default)
  ────────────────────
  Agent dijalankan berurutan. KV-cache di-chain: output agent N
  menjadi past_kv untuk agent N+1. Urutan default:
    makro → fundamental → news → technical → manager

  Setiap agent mendapat konteks dari agent sebelumnya melalui
  KV-cache, sehingga analisis semakin kaya secara inkremental.

  HIERARCHICAL
  ─────────────
  Semua domain agents berjalan independen (tidak saling tahu
  jawaban satu sama lain). Lalu semua insights dikumpulkan
  dan dikirim ke ManagerAgent untuk sintesis.

  Cocok untuk parallelisasi — setiap agent bisa di-thread/process.

Output tunggal: InsightResult
  .unified_insight → ExternalInsight dari ManagerAgent
  .agent_insights  → Dict[str, ExternalInsight] per domain agent
  .external_context→ formatted string untuk AlphaAgentLoop
  .kv_cache        → shortcut ke unified_insight.kv_cache
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from log import logger

# Lazy imports to avoid hard dependency
try:
    from eksternal.base import ExternalAgentBase, ExternalInsight
    from eksternal.manager import ManagerAgent
    _HAS_EXTERNAL = True
except ImportError:
    _HAS_EXTERNAL = False
    ExternalAgentBase = Any  # type: ignore[assignment,misc]
    ExternalInsight = Any    # type: ignore[assignment,misc]
    ManagerAgent = Any       # type: ignore[assignment,misc]

try:
    from llm.client import LocalLLMBackend, KVCache
    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False
    LocalLLMBackend = Any    # type: ignore[assignment,misc]
    KVCache = Any            # type: ignore[assignment,misc]


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InsightResult:
    """
    Output dari InsightOrchestrator.

    Routing:
        .unified_insight.kv_cache → planning.generate_parallel_directions()
        .external_context         → AlphaAgentLoop(external_context=...)
        .unified_insight          → [ExternalInsight] untuk planning external_insights
        .agent_insights           → per-agent results for debugging / selective use
    """

    unified_insight: Optional["ExternalInsight"] = None
    agent_insights: Dict[str, "ExternalInsight"] = field(default_factory=dict)
    mode: str = "sequential"  # "sequential" | "hierarchical"
    processing_time_s: float = 0.0

    @property
    def external_context(self) -> Optional[str]:
        """Formatted string untuk AlphaAgentLoop."""
        if self.unified_insight is None:
            return None
        return self.unified_insight.to_context_str()

    @property
    def kv_cache(self) -> Optional[Any]:
        """Shortcut ke unified KV-cache."""
        if self.unified_insight is None:
            return None
        return self.unified_insight.kv_cache

    @property
    def has_data(self) -> bool:
        return self.unified_insight is not None and bool(self.unified_insight.summary)

    def as_external_insights_list(self) -> List["ExternalInsight"]:
        """Return unified insight wrapped in a list, for planning API."""
        if self.unified_insight is None:
            return []
        return [self.unified_insight]


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

# Default execution order for sequential mode.
# makro first (broadest context), then narrowing to more specific domains.
_DEFAULT_AGENT_ORDER = ["makro", "fundamental", "news", "technical"]


class InsightOrchestrator:
    """
    Orchestrates external agents and ManagerAgent.

    Usage:
        orchestrator = InsightOrchestrator(
            agents=[makro_agent, news_agent, ...],
            manager=manager_agent,
        )
        result = orchestrator.run(mode="sequential")

        # Use in pipeline:
        directions = generate_parallel_directions(
            ...,
            external_insights=result.as_external_insights_list(),
            llm_backend=backend,
        )
        loop = AlphaAgentLoop(
            ...,
            external_context=result.external_context,
        )
    """

    def __init__(
        self,
        agents: List["ExternalAgentBase"],
        manager: "ManagerAgent",
        agent_order: Optional[List[str]] = None,
        max_parallel_workers: int = 4,
    ) -> None:
        """
        Args:
            agents            : domain agents (makro, news, fundamental, technical)
            manager           : ManagerAgent instance (synthesis, KV-cache only)
            agent_order       : custom execution order for sequential mode.
                                Agent names not in this list run last (alphabetical).
            max_parallel_workers: thread pool size for hierarchical mode.
        """
        self.agents = agents
        self.manager = manager
        self.max_parallel_workers = max_parallel_workers

        # Build agent lookup by name
        self._agent_map: Dict[str, "ExternalAgentBase"] = {}
        for a in agents:
            name = getattr(a, "agent_name", type(a).__name__)
            self._agent_map[name] = a

        # Resolve execution order
        order = agent_order or _DEFAULT_AGENT_ORDER
        ordered_names = [n for n in order if n in self._agent_map]
        remaining = sorted(n for n in self._agent_map if n not in ordered_names)
        self._ordered_names = ordered_names + remaining

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        mode: str = "sequential",
        past_kv: Optional[Any] = None,
    ) -> InsightResult:
        """
        Run all agents + manager and return InsightResult.

        Args:
            mode   : "sequential" or "hierarchical"
            past_kv: optional seed KV-cache
        """
        t_start = time.time()

        if mode == "hierarchical":
            result = self._run_hierarchical(past_kv)
        else:
            result = self._run_sequential(past_kv)

        result.mode = mode
        result.processing_time_s = time.time() - t_start

        logger.info(
            f"[InsightOrchestrator] Done: mode={mode}, "
            f"agents={len(result.agent_insights)}, "
            f"unified={'yes' if result.has_data else 'no'}, "
            f"time={result.processing_time_s:.1f}s"
        )

        return result

    def update_all_strategies(self, feedback: Dict[str, Any]) -> None:
        """
        Update strategies for all agents + manager after evolution completes.
        Called by factor_mining.py.
        """
        for agent in self.agents:
            try:
                agent.update_strategy(feedback)
                name = getattr(agent, "agent_name", type(agent).__name__)
                logger.info(f"[InsightOrchestrator] Strategy updated: {name}")
            except Exception as e:
                logger.warning(
                    f"[InsightOrchestrator] Strategy update failed for "
                    f"{type(agent).__name__}: {e}"
                )
        try:
            self.manager.update_strategy(feedback)
            logger.info("[InsightOrchestrator] Strategy updated: manager")
        except Exception as e:
            logger.warning(f"[InsightOrchestrator] Manager strategy update failed: {e}")

    # ── Sequential mode ──────────────────────────────────────────────────

    def _run_sequential(self, past_kv: Optional[Any]) -> InsightResult:
        """
        Run agents sequentially, chaining KV-cache.

        Order: makro → fundamental → news → technical → manager

        Each agent gets the KV-cache from the previous agent,
        building up latent context incrementally.
        """
        agent_insights: Dict[str, "ExternalInsight"] = {}
        current_kv = past_kv

        for name in self._ordered_names:
            agent = self._agent_map[name]
            logger.info(f"[Sequential] Running agent: {name}")

            try:
                insight = agent.run(past_kv=current_kv)
                agent_insights[name] = insight

                # Chain KV-cache to next agent
                if insight.has_kv:
                    current_kv = insight.kv_cache

                logger.info(
                    f"[Sequential] {name} done: "
                    f"summary_len={len(insight.summary)}, "
                    f"has_kv={insight.has_kv}, "
                    f"docs={insight.n_documents_collected}"
                )
            except Exception as e:
                logger.warning(f"[Sequential] Agent {name} failed: {e}")

        # ── Manager synthesis ─────────────────────────────────────────────
        # Manager receives all insights + chained KV-cache
        insights_list = list(agent_insights.values())

        if insights_list:
            logger.info(
                f"[Sequential] Running manager with {len(insights_list)} "
                f"insights, kv_len={'chained' if current_kv is not None else 'none'}"
            )
            try:
                unified = self.manager.run(
                    past_kv=current_kv,
                    insights=insights_list,
                )
                return InsightResult(
                    unified_insight=unified,
                    agent_insights=agent_insights,
                )
            except Exception as e:
                logger.warning(f"[Sequential] Manager failed: {e}")

        # Fallback: if manager fails, use best individual insight
        return self._fallback_result(agent_insights)

    # ── Hierarchical mode ─────────────────────────────────────────────────

    def _run_hierarchical(self, past_kv: Optional[Any]) -> InsightResult:
        """
        Run all domain agents independently (parallel), then feed
        all insights to ManagerAgent.

        Agents do NOT share KV-cache with each other.
        Each agent may receive the seed past_kv independently.
        """
        agent_insights: Dict[str, "ExternalInsight"] = {}

        if len(self._agent_map) <= 1:
            # Single agent — no need for thread pool
            for name, agent in self._agent_map.items():
                try:
                    insight = agent.run(past_kv=past_kv)
                    agent_insights[name] = insight
                except Exception as e:
                    logger.warning(f"[Hierarchical] Agent {name} failed: {e}")
        else:
            # Parallel execution via thread pool
            with ThreadPoolExecutor(
                max_workers=min(self.max_parallel_workers, len(self._agent_map))
            ) as pool:
                futures = {
                    pool.submit(self._run_single_agent, name, agent, past_kv): name
                    for name, agent in self._agent_map.items()
                }

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        insight = future.result()
                        if insight is not None:
                            agent_insights[name] = insight
                            logger.info(
                                f"[Hierarchical] {name} done: "
                                f"summary_len={len(insight.summary)}, "
                                f"has_kv={insight.has_kv}"
                            )
                    except Exception as e:
                        logger.warning(f"[Hierarchical] Agent {name} failed: {e}")

        # ── Manager synthesis (no chained KV — fresh start) ───────────────
        insights_list = list(agent_insights.values())

        if insights_list:
            logger.info(
                f"[Hierarchical] Running manager with {len(insights_list)} "
                f"independent insights"
            )
            try:
                # Manager starts with no prior KV (hierarchical = independent)
                unified = self.manager.run(
                    past_kv=None,
                    insights=insights_list,
                )
                return InsightResult(
                    unified_insight=unified,
                    agent_insights=agent_insights,
                )
            except Exception as e:
                logger.warning(f"[Hierarchical] Manager failed: {e}")

        return self._fallback_result(agent_insights)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _run_single_agent(
        name: str,
        agent: "ExternalAgentBase",
        past_kv: Optional[Any],
    ) -> Optional["ExternalInsight"]:
        """Run a single agent, catching exceptions."""
        try:
            logger.info(f"[Hierarchical] Starting agent: {name}")
            return agent.run(past_kv=past_kv)
        except Exception as e:
            logger.warning(f"[Hierarchical] Agent {name} error: {e}")
            return None

    @staticmethod
    def _fallback_result(
        agent_insights: Dict[str, "ExternalInsight"],
    ) -> InsightResult:
        """
        Fallback when manager fails: pick the best individual insight
        (longest summary with KV-cache) as the unified insight.
        """
        if not agent_insights:
            return InsightResult()

        # Prefer insight with KV-cache; among those, pick longest summary
        candidates = sorted(
            agent_insights.values(),
            key=lambda i: (i.has_kv, len(i.summary)),
            reverse=True,
        )
        return InsightResult(
            unified_insight=candidates[0],
            agent_insights=agent_insights,
        )


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY — convenience for factor_mining.py
# ═══════════════════════════════════════════════════════════════════════════

def build_orchestrator(
    agents: List["ExternalAgentBase"],
    llm_backend: "LocalLLMBackend",
    manager_config: Optional[Dict[str, Any]] = None,
    agent_order: Optional[List[str]] = None,
) -> InsightOrchestrator:
    """
    Build InsightOrchestrator with a ManagerAgent created from llm_backend.

    Args:
        agents         : list of domain agents (makro, news, etc.)
        llm_backend    : LocalLLMBackend for ManagerAgent
        manager_config : kwargs for ManagerConfig
        agent_order    : custom execution order for sequential mode
    """
    from eksternal.manager import ManagerAgent as _ManagerAgent, ManagerConfig
    cfg = ManagerConfig(**(manager_config or {}))
    manager = _ManagerAgent(llm_backend=llm_backend, config=cfg)
    return InsightOrchestrator(
        agents=agents,
        manager=manager,
        agent_order=agent_order,
    )
