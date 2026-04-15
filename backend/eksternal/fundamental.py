"""
fundamental.py — Fundamental Analysis Agent
=============================================

Searches for company fundamentals, earnings, financial ratios, and valuation
data.  Follows the same two-phase architecture as MacroExternalAgent:

  Phase 1 — Search: web search for fundamental / earnings data
  Phase 2 — Analysis: LocalLLMBackend latent reasoning → ExternalInsight

Usage:
    from eksternal.fundamental import FundamentalExternalAgent, FundamentalConfig

    agent = FundamentalExternalAgent(llm_backend=local_backend)
    insight = agent.run()
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from eksternal.base import ExternalAgentBase, ExternalInsight, SearchStrategy

try:
    from llm.client import LocalLLMBackend, LLMResult, KVCache, _past_length
except ImportError:
    LocalLLMBackend = Any  # type: ignore[assignment,misc]
    LLMResult = Any        # type: ignore[assignment,misc]
    KVCache = Any          # type: ignore[assignment,misc]
    _past_length = lambda x: 0  # type: ignore[assignment]

from log import logger


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FundamentalConfig:
    """Konfigurasi FundamentalExternalAgent."""

    # ── Phase 1: Search ────────────────────────────────────────────────────
    max_search_iterations: int = 6
    max_items_per_query: int   = 5
    search_provider: str       = "duckduckgo"
    search_api_key: Optional[str] = None

    # ── Phase 2: Analysis ──────────────────────────────────────────────────
    analysis_mode: str   = "kv_and_text"
    max_new_tokens: int  = 2048
    temperature: float   = 0.6
    top_p: float         = 0.95
    chunk_size: int      = 5

    # ── Strategy persistence ───────────────────────────────────────────────
    strategy_path: str = "./.strategy/fundamental_strategy.json"

    # ── Focus ──────────────────────────────────────────────────────────────
    default_topics: List[str] = field(default_factory=lambda: [
        "earnings surprise",
        "revenue growth",
        "PE ratio sector",
        "balance sheet strength",
        "dividend yield",
        "insider trading activity",
        "analyst upgrades downgrades",
        "free cash flow",
    ])
    focus_sectors: List[str] = field(default_factory=lambda: [
        "Technology", "Financials", "Healthcare", "Energy", "Consumer",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Data structure
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FundamentalDataItem:
    """Satu potong data fundamental yang dikumpulkan di Phase 1."""
    title: str
    content: str
    source: str = "web_search"
    timestamp: datetime = field(default_factory=datetime.now)
    category: str = "fundamental"  # earnings | valuation | balance_sheet | etc
    url: str = ""
    ticker: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# FundamentalExternalAgent
# ═══════════════════════════════════════════════════════════════════════════

class FundamentalExternalAgent(ExternalAgentBase):
    """
    Fundamental Analysis Intelligence Agent.

    Phase 1: Search for earnings, valuations, financial ratios
    Phase 2: LocalLLMBackend analysis with KV-cache → ExternalInsight
    """

    agent_name = "fundamental"

    def __init__(
        self,
        llm_backend: "LocalLLMBackend",
        config: Optional[FundamentalConfig] = None,
        strategy_path: Optional[Path] = None,
    ) -> None:
        cfg = config or FundamentalConfig()
        super().__init__(
            strategy_path=strategy_path or Path(cfg.strategy_path)
        )
        self.config = cfg
        self.llm_backend = llm_backend

    # ── Main entry ────────────────────────────────────────────────────────

    def run(
        self,
        past_kv: Optional[Any] = None,
        extra_queries: Optional[List[str]] = None,
        **kwargs,
    ) -> ExternalInsight:
        t_start = time.time()
        conv_id = f"fundamental_{uuid.uuid4().hex[:8]}"

        # Phase 1: Search
        items, queries_issued = self._phase1_search(extra_queries or [])
        logger.info(f"[Fundamental] Phase 1: {len(items)} items, {len(queries_issued)} queries")

        # Phase 2: Analyze
        summary, kv_cache, hidden_state = self._phase2_analyze(items, past_kv, conv_id)
        logger.info(f"[Fundamental] Phase 2: summary_len={len(summary)}, has_kv={kv_cache is not None}")

        direction_hint = self._extract_direction_hint(summary)

        return ExternalInsight(
            source_agent="fundamental",
            summary=summary,
            direction_hint=direction_hint,
            kv_cache=kv_cache,
            hidden_state=hidden_state,
            search_queries=queries_issued,
            n_documents_collected=len(items),
            conv_id=conv_id,
            processing_time_s=time.time() - t_start,
        )

    # ── Phase 1: Search ──────────────────────────────────────────────────

    def _phase1_search(
        self, extra_queries: List[str]
    ) -> Tuple[List[FundamentalDataItem], List[str]]:
        queries = self._build_queries() + extra_queries
        return self._fallback_search(queries), queries[:6]

    def _build_queries(self) -> List[str]:
        year = datetime.now().year
        queries = []
        for topic in self.config.default_topics[:5]:
            queries.append(f"{topic} {year}")
        for sector in self.config.focus_sectors[:3]:
            queries.append(f"{sector} sector fundamentals outlook {year}")
        queries += self.strategy.additional_queries[:3]
        return queries

    def _fallback_search(self, queries: List[str]) -> List[FundamentalDataItem]:
        from eksternal.tools.web_tools import make_web_search_tool
        search_fn = make_web_search_tool(
            provider=self.config.search_provider,
            api_key=self.config.search_api_key,
        )
        items: List[FundamentalDataItem] = []
        for q in queries[:self.config.max_search_iterations]:
            try:
                result_json = search_fn.invoke({"query": q, "max_results": self.config.max_items_per_query})
                results = json.loads(result_json)
                if isinstance(results, list):
                    for r in results:
                        items.append(FundamentalDataItem(
                            title=r.get("title", ""),
                            content=r.get("snippet", ""),
                            source="web_search",
                            url=r.get("url", ""),
                        ))
            except Exception:
                continue
        return items

    # ── Phase 2: Analyze ─────────────────────────────────────────────────

    def _phase2_analyze(
        self,
        items: List[FundamentalDataItem],
        past_kv: Optional[Any],
        conv_id: str,
    ) -> Tuple[str, Optional[Any], Optional[torch.Tensor]]:
        if not items:
            return "Insufficient data for fundamental analysis.", None, None

        # Build analysis prompt from collected data
        data_text = "\n\n".join(
            f"[{i+1}] {item.title}\n{item.content}" for i, item in enumerate(items[:20])
        )

        system_prompt = (
            "You are a fundamental analysis expert. Analyze the following financial data "
            "and provide:\n"
            "1. Key earnings and valuation trends\n"
            "2. Sectors showing strength or weakness\n"
            "3. Notable fundamental signals for factor construction\n"
            "4. DIRECTION HINT: one sentence suggesting a factor strategy direction\n"
            "Be concise and quantitative."
        )
        user_prompt = f"Analyze these fundamental data points:\n\n{data_text}"

        result = self.llm_backend.build_messages_and_run(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            past_key_values=past_kv,
            mode=self.config.analysis_mode,
            role="fundamental_analyst",
            conv_id=conv_id,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        return (
            result.text or "",
            result.kv_cache,
            result.hidden_last,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_direction_hint(summary: str) -> str:
        match = re.search(r"DIRECTION HINT:\s*(.+?)(?:\n|$)", summary, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        for s in sentences:
            s = s.strip()
            if 20 < len(s) < 150:
                return s
        return summary[:120].strip()

    # ── Strategy update (3.B) ────────────────────────────────────────────

    def update_strategy(self, feedback: Dict[str, Any]) -> None:
        topic_perf = feedback.get("topic_performance", {})
        for topic, avg_metric in topic_perf.items():
            delta = 0.2 if avg_metric > 0.03 else -0.1
            self.strategy.update_topic_weight(topic, delta)

        successful = feedback.get("successful_queries", [])
        for q in successful:
            if q not in self.strategy.additional_queries:
                self.strategy.additional_queries.append(q)
        self.strategy.additional_queries = self.strategy.additional_queries[-10:]

        self.save_strategy()
        logger.info("[Fundamental] Strategy updated")
