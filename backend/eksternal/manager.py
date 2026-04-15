"""
manager.py — External Insight Manager Agent
=============================================

Menerima insights dari semua external agents (makro, news, fundamental, technical)
dan menghasilkan satu ExternalInsight terpadu untuk pipeline internal.

Berbeda dari agent lain:
  - TIDAK punya tool calling (tidak ada Phase 1 / search graph)
  - Hanya 1 LLM (LocalLLMBackend) untuk analisis + KV-cache
  - Menerima insights sebagai input, bukan query

Flow:
    [makro, news, fundamental, technical]
          ↓ List[ExternalInsight]
    ManagerAgent.run()
          ↓ latent passes (incremental KV-cache)
    ExternalInsight (unified)
          ↓
    planning.py (kv_cache)  +  loop.py (summary as external_context)

KV-cache chaining:
  - ManagerAgent menerima optional past_kv (bisa dari agent sebelumnya)
  - Setiap insight di-encode sebagai chunk via latent pass (kv_only)
  - Chunk terakhir: kv_and_text (generate unified summary)
  - KV-cache akhir mengandung seluruh konteks multi-agent

Usage:
    from eksternal.manager import ManagerAgent, ManagerConfig

    manager = ManagerAgent(llm_backend=backend)
    unified = manager.run(
        insights=[makro_insight, news_insight, ...],
        past_kv=None,
    )

    # unified.kv_cache      → planning.generate_parallel_directions()
    # unified.summary       → AlphaAgentLoop.external_context
    # unified.direction_hint→ direction seeding
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from eksternal.base import ExternalAgentBase, ExternalInsight, SearchStrategy
from llm.client import LocalLLMBackend, LLMResult, KVCache, _past_length
from debug.monitor import Monitor, MonitorConfig


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ManagerConfig:
    """Konfigurasi ManagerAgent."""

    # ── Analysis (LocalLLMBackend) ─────────────────────────────────────────
    analysis_mode: str   = "kv_and_text"  # final chunk mode
    max_new_tokens: int  = 3072           # longer — manager synthesizes all
    temperature: float   = 0.5            # lower — more focused synthesis
    top_p: float         = 0.9
    chunk_size: int      = 2             # insights per latent pass

    # ── Monitoring ─────────────────────────────────────────────────────────
    enable_monitor: bool = True
    log_dir: str         = "./logs/manager"

    # ── Strategy persistence ───────────────────────────────────────────────
    strategy_path: str = "./.strategy/manager_strategy.json"


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

_MANAGER_SYSTEM_PROMPT = """\
You are a senior investment strategist synthesizing multiple research streams
for a quantitative alpha factor mining system.

You receive analyses from specialized agents:
- MAKRO: macroeconomic outlook, central bank policy, rates, GDP, inflation
- NEWS:  breaking news, market sentiment, event-driven signals
- FUNDAMENTAL: company fundamentals, earnings, valuations, sector rotations
- TECHNICAL: price action, momentum, mean reversion, volatility regimes

Your task:
1. CROSS-REFERENCE: Identify agreement and disagreement across agents
2. PRIORITIZE: Rank themes by impact likelihood and factor mining relevance
3. SYNTHESIZE: Produce ONE coherent market view with concrete implications
4. DIRECTION HINTS: Generate 2-3 specific alpha factor research directions
   that exploit the identified themes

Format your output:
- Start with a brief executive summary (2-3 sentences)
- Then detail the cross-agent synthesis
- End with:

DIRECTION HINT: <primary actionable direction for alpha factor research>
DIRECTION HINT 2: <secondary direction>
DIRECTION HINT 3: <tertiary direction (optional)>
"""


# ═══════════════════════════════════════════════════════════════════════════
# ManagerAgent
# ═══════════════════════════════════════════════════════════════════════════

class ManagerAgent(ExternalAgentBase):
    """
    Manager agent — synthesizes insights from all external agents.

    No tool calling. Single LLM for incremental KV-cache analysis.

    Accepts List[ExternalInsight] instead of searching for data.
    """

    agent_name = "manager"

    def __init__(
        self,
        llm_backend: LocalLLMBackend,
        config: Optional[ManagerConfig] = None,
        monitor: Optional[Monitor] = None,
        strategy_path: Optional[Path] = None,
    ) -> None:
        cfg = config or ManagerConfig()
        super().__init__(
            strategy_path=strategy_path or Path(cfg.strategy_path)
        )
        self.config = cfg
        self.llm_backend = llm_backend

        # Monitor
        if monitor is not None:
            self.monitor = monitor
        elif cfg.enable_monitor:
            self.monitor = Monitor(
                MonitorConfig(
                    log_dir=cfg.log_dir,
                    log_file_prefix="manager_agent",
                )
            )
        else:
            self.monitor = None

    # ── Main entry ────────────────────────────────────────────────────────

    def run(
        self,
        past_kv: Optional[KVCache] = None,
        *,
        insights: Optional[List[ExternalInsight]] = None,
        **kwargs,
    ) -> ExternalInsight:
        """
        Synthesize multiple ExternalInsights into one unified insight.

        Args:
            past_kv  : optional KV-cache seed (e.g. from sequential chaining)
            insights : list of ExternalInsight from domain agents
        """
        t_start = time.time()
        conv_id = f"manager_{uuid.uuid4().hex[:8]}"
        agent_insights = insights or []

        if self.monitor:
            self.monitor.log_separator("═", 70)
            self.monitor.log(
                f"ManagerAgent.run()  conv_id={conv_id}  "
                f"n_insights={len(agent_insights)}"
            )

        if not agent_insights:
            return ExternalInsight(
                source_agent="manager",
                summary="No agent insights available for synthesis.",
                direction_hint="",
                conv_id=conv_id,
                processing_time_s=time.time() - t_start,
            )

        # ── Incremental KV-cache analysis ─────────────────────────────────
        summary, kv_cache, hidden_state = self._analyze_insights(
            agent_insights, past_kv, conv_id
        )

        if self.monitor:
            self.monitor.log(
                f"Analysis complete: summary_len={len(summary)}, "
                f"has_kv={kv_cache is not None}"
            )

        # ── Build unified ExternalInsight ─────────────────────────────────
        direction_hint = self._extract_direction_hint(summary)

        # Merge all search queries from sub-agents
        all_queries: List[str] = []
        total_docs = 0
        for ins in agent_insights:
            all_queries.extend(ins.search_queries)
            total_docs += ins.n_documents_collected

        unified = ExternalInsight(
            source_agent="manager",
            summary=summary,
            direction_hint=direction_hint,
            kv_cache=kv_cache,
            hidden_state=hidden_state,
            search_queries=list(dict.fromkeys(all_queries)),  # dedup
            n_documents_collected=total_docs,
            conv_id=conv_id,
            processing_time_s=time.time() - t_start,
            metadata={
                "n_source_agents": len(agent_insights),
                "source_agents": [ins.source_agent for ins in agent_insights],
                "kv_seq_len": _past_length(kv_cache) if kv_cache else 0,
            },
        )

        if self.monitor:
            self.monitor.log_variable("unified_insight", unified.to_dict())

        return unified

    # ── Incremental analysis ──────────────────────────────────────────────

    def _analyze_insights(
        self,
        insights: List[ExternalInsight],
        past_kv: Optional[KVCache],
        conv_id: str,
    ) -> Tuple[str, Optional[KVCache], Optional[torch.Tensor]]:
        """
        Encode insights incrementally into KV-cache, then generate summary.

        Each chunk of insights → latent pass (kv_only).
        Final chunk → kv_and_text (generate synthesis text + final KV-cache).
        """
        chunk_size = self.config.chunk_size
        chunks = [
            insights[i: i + chunk_size]
            for i in range(0, len(insights), chunk_size)
        ]
        current_kv = past_kv

        # ── Latent passes for all chunks except last ─────────────────────
        for step_idx, chunk in enumerate(chunks[:-1]):
            messages = self._build_messages(chunk)
            result: LLMResult = self.llm_backend.run(
                messages=messages,
                mode="kv_only",
                past_key_values=current_kv,
                conv_id=conv_id,
                step=step_idx,
                role="manager",
            )
            current_kv = result.kv_cache

            if self.monitor:
                self.monitor.log(
                    f"Latent pass {step_idx}: "
                    f"agents=[{', '.join(i.source_agent for i in chunk)}], "
                    f"kv_len={_past_length(current_kv)}"
                )

        # ── Final chunk: generate text + KV-cache ────────────────────────
        final_messages = self._build_messages(chunks[-1])
        final_result: LLMResult = self.llm_backend.run(
            messages=final_messages,
            mode=self.config.analysis_mode,
            past_key_values=current_kv,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            conv_id=conv_id,
            step=len(chunks) - 1,
            role="manager",
        )

        return (
            final_result.text or "",
            final_result.kv_cache,
            final_result.hidden_last,
        )

    def _build_messages(
        self,
        insights: List[ExternalInsight],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages from a chunk of ExternalInsights."""
        context_parts = []
        for i, ins in enumerate(insights, 1):
            context_parts.append(
                f"=== AGENT: {ins.source_agent.upper()} "
                f"[{ins.timestamp.strftime('%Y-%m-%d %H:%M')}] ===\n"
                f"Direction Hint: {ins.direction_hint}\n\n"
                f"{ins.summary}\n"
                f"{'─' * 50}"
            )

        context = "\n\n".join(context_parts)
        user_prompt = (
            f"Synthesize the following {len(insights)} agent report(s) "
            f"into a unified market view and alpha factor directions:\n\n"
            f"{context}"
        )

        return self.llm_backend.build_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt or _MANAGER_SYSTEM_PROMPT,
        )

    # ── Direction hint extraction ─────────────────────────────────────────

    @staticmethod
    def _extract_direction_hint(summary: str) -> str:
        """
        Extract primary DIRECTION HINT from manager summary.
        Falls back to first meaningful sentence.
        """
        match = re.search(
            r"DIRECTION HINT:\s*(.+?)(?:\n|$)", summary, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        sentences = re.split(r"(?<=[.!?])\s+", summary)
        for s in sentences:
            s = s.strip()
            if 20 < len(s) < 150:
                return s

        return summary[:120].strip()

    @staticmethod
    def _extract_all_direction_hints(summary: str) -> List[str]:
        """Extract all DIRECTION HINT lines from summary."""
        hints = re.findall(
            r"DIRECTION HINT(?:\s*\d*)?:\s*(.+?)(?:\n|$)",
            summary, re.IGNORECASE,
        )
        return [h.strip() for h in hints if h.strip()]

    # ── Strategy update (3.B) ─────────────────────────────────────────────

    def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """
        Update strategy based on evolution feedback.

        Manager's strategy doesn't control search (no tool calling),
        but it tracks which synthesis themes correlate with best performance.
        This can influence how future synthesis weighs different agents.
        """
        topic_perf: Dict[str, float] = feedback.get("topic_performance", {})

        baseline = 0.05
        scale = 2.0
        for topic, score in topic_perf.items():
            delta = (score - baseline) * scale
            self.strategy.update_topic_weight(topic, delta)

        self.save_strategy()

        if self.monitor:
            self.monitor.log(
                f"Strategy updated (count={self.strategy.update_count}). "
                f"Top topics: {self.strategy.top_topics(3)}"
            )

    # ── Info & cleanup ────────────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "strategy_update_count": self.strategy.update_count,
            "config": {
                "analysis_mode": self.config.analysis_mode,
                "chunk_size": self.config.chunk_size,
                "max_new_tokens": self.config.max_new_tokens,
            },
        }

    def close(self) -> None:
        if self.monitor:
            self.monitor.close()


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def create_manager_agent(
    llm_backend: LocalLLMBackend,
    **config_kwargs: Any,
) -> ManagerAgent:
    """
    Factory untuk ManagerAgent.

    Examples
    --------
    manager = create_manager_agent(llm_backend=backend)
    manager = create_manager_agent(
        llm_backend=backend,
        max_new_tokens=4096,
        chunk_size=1,   # one insight per latent pass
    )
    """
    config = ManagerConfig(**config_kwargs)
    return ManagerAgent(llm_backend=llm_backend, config=config)
