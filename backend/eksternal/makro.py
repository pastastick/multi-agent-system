"""
makro.py — Macro-Economic Intelligence Agent
=============================================

Two-phase architecture:

  Phase 1 ─ MacroSearchGraph (LangGraph + API model)
  ─────────────────────────────────────────────────────
  Iterative ReAct loop: LLM decides what to search → calls tools →
  observes results → decides: enough data? → stop atau search lagi.

  Tools: web_search, scrape_url, get_economic_data, mcp_call, mark_done
  Model: API model (Claude / OpenAI / any LangChain BaseChatModel)

  Phase 2 ─ MacroAnalyst (LocalLLMBackend, Qwen)
  ─────────────────────────────────────────────────────
  Raw data dari Phase 1 diproses via analyze_incremental():
    chunk → latent pass (KV-cache) → chunk → ... → final generate + KV

  Output: ExternalInsight
    .summary       → AlphaAgentLoop.factor_propose (external_context)
    .kv_cache      → planning.generate_parallel_directions (past_key_values)
    .direction_hint→ direction seeding

Update Strategy (3.B):
  MacroExternalAgent.update_strategy(feedback) dipanggil setelah semua
  iterasi evolusi selesai. Feedback dari EvolutionController (performance
  tiap direction) dipakai untuk menyesuaikan bobot topic dan query template.

Usage:
    from eksternal.makro import MacroExternalAgent, MacroConfig

    agent = MacroExternalAgent(
        llm_backend=local_backend,          # LocalLLMBackend untuk Phase 2
        api_model=chat_anthropic_instance,  # API model untuk Phase 1
    )
    insight = agent.run()

    # Pass ke planning
    directions = generate_parallel_directions(
        initial_direction="...",
        n=5,
        prompt_file=path,
        external_insights=[insight],
        llm_backend=local_backend,
    )
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# ── Internal imports ──────────────────────────────────────────────────────
from eksternal.base import ExternalAgentBase, ExternalInsight, SearchStrategy
from eksternal.tools.web_tools import get_default_tools
from llm.client import LocalLLMBackend, LLMResult, KVCache, _past_length
from debug.monitor import Monitor, MonitorConfig

# ── LangGraph / LangChain (optional) ─────────────────────────────────────
try:
    from typing import Annotated
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import (
        BaseMessage,
        HumanMessage,
        SystemMessage,
        AIMessage,
        ToolMessage,
    )
    from langchain_core.language_models import BaseChatModel
    try:
        from langgraph.graph.message import add_messages
    except ImportError:
        from operator import add as add_messages  # fallback
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False
    BaseChatModel = Any  # type: ignore[assignment,misc]


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 1 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MacroConfig:
    """Konfigurasi MacroExternalAgent."""

    # ── Phase 1: Search (LangGraph + API model) ───────────────────────────
    max_search_iterations: int = 6       # max iterasi ReAct loop
    max_items_per_query: int   = 5       # max hasil per web_search call
    search_provider: str       = "duckduckgo"  # "duckduckgo" | "tavily" | "serpapi"
    search_api_key: Optional[str] = None # untuk Tavily / SerpAPI

    # ── Phase 2: Analysis (LocalLLMBackend) ───────────────────────────────
    analysis_mode: str   = "kv_and_text"   # "kv_only" | "text_only" | "kv_and_text"
    max_new_tokens: int  = 2048
    temperature: float   = 0.6
    top_p: float         = 0.95
    chunk_size: int      = 5              # items per latent pass

    # ── Monitoring ────────────────────────────────────────────────────────
    enable_monitor: bool = True
    log_dir: str         = "./logs/makro"

    # ── Strategy persistence ──────────────────────────────────────────────
    strategy_path: str = "./.strategy/makro_strategy.json"


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 2 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MacroDataItem:
    """Satu potong data / berita makro ekonomi yang dikumpulkan di Phase 1."""

    title: str
    content: str
    source: str
    timestamp: datetime
    category: str = "general"
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_context_str(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M")
        parts = [f"[{ts}] [{self.category.upper()}] {self.title}"]
        if self.content:
            parts.append(self.content[:500])
        if self.url:
            parts.append(f"Source: {self.url}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 3 — PHASE 1: MacroSearchGraph (LangGraph)
# ═══════════════════════════════════════════════════════════════════════════

_SEARCH_SYSTEM_PROMPT = """\
You are a macroeconomic research agent. Your task is to gather comprehensive,
up-to-date information about the global macroeconomic situation.

Focus on:
- Central bank policies (Fed, ECB, PBoC, BoJ) and interest rate decisions
- Inflation trends (CPI, PCE, producer prices)
- GDP growth and recession risks
- Employment data and labor market conditions
- Geopolitical risks affecting global markets
- Currency dynamics (USD strength, EM currencies)
- Commodity markets (oil, gold, industrial metals)
- Credit conditions and spread dynamics

Strategy:
1. Start with broad searches to identify key themes
2. Drill down into the most market-relevant developments
3. Gather quantitative data where possible (rates, growth figures)
4. Call mark_done when you have enough to produce a thorough analysis
   (typically after 3-5 meaningful searches)
"""


class MacroSearchGraph:
    """
    Phase 1: iterative data collection via LangGraph ReAct pattern.

    Graph:
        START → agent_node → (tool_calls?) → tool_node → agent_node → ...
                           → (no calls / mark_done / max_iter) → END
    """

    def __init__(
        self,
        api_model: "BaseChatModel",
        config: MacroConfig,
        tools_list: Optional[List] = None,
        mcp_caller: Optional[Callable] = None,
    ) -> None:
        if not _HAS_LANGGRAPH:
            raise ImportError(
                "LangGraph required for MacroSearchGraph.\n"
                "pip install langgraph langchain-core"
            )

        self.config = config
        self.tools = tools_list or get_default_tools(
            search_provider=config.search_provider,
            search_api_key=config.search_api_key,
            mcp_caller=mcp_caller,
        )

        self.tool_map = {t.name: t for t in self.tools}
        self.llm = api_model.bind_tools(self.tools)
        self._graph = self._build_graph()

    # ── Graph construction ─────────────────────────────────────────────────

    def _build_graph(self):
        from typing import TypedDict

        # ── State ─────────────────────────────────────────────────────────
        class SearchState(dict):
            pass

        builder = StateGraph(dict)
        tool_node = ToolNode(self.tools)

        def agent_node(state: dict) -> dict:
            messages = state.get("messages", [])
            response = self.llm.invoke(messages)
            new_messages = messages + [response]

            # Check if mark_done was called
            done = state.get("done", False)
            if hasattr(response, "tool_calls"):
                for tc in (response.tool_calls or []):
                    if tc.get("name") == "mark_done":
                        done = True

            return {
                **state,
                "messages": new_messages,
                "iteration": state.get("iteration", 0) + 1,
                "done": done,
            }

        def tools_node(state: dict) -> dict:
            result = tool_node.invoke(state)
            return {**state, **result}

        def should_continue(state: dict) -> str:
            if state.get("done", False):
                return END
            if state.get("iteration", 0) >= self.config.max_search_iterations:
                return END
            msgs = state.get("messages", [])
            if msgs:
                last = msgs[-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    return "tools"
            return END

        builder.add_node("agent", agent_node)
        builder.add_node("tools", tools_node)
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")

        return builder.compile()

    # ── Public interface ───────────────────────────────────────────────────

    def search(
        self,
        queries: List[str],
    ) -> Tuple[List[MacroDataItem], List[str]]:
        """
        Run the search graph.

        Args:
            queries: Initial search queries (dari SearchStrategy)

        Returns:
            (collected_items, queries_actually_issued)
        """
        initial_msg = HumanMessage(
            content=(
                "Please research the current global macroeconomic situation. "
                f"Start with these initial queries: {queries[:3]}. "
                "Gather comprehensive data, then call mark_done when finished."
            )
        )

        initial_state = {
            "messages": [SystemMessage(content=_SEARCH_SYSTEM_PROMPT), initial_msg],
            "iteration": 0,
            "done": False,
        }

        final_state = self._graph.invoke(initial_state)
        return self._extract_items(final_state), self._extract_queries(final_state)

    # ── Extraction helpers ─────────────────────────────────────────────────

    def _extract_items(self, state: dict) -> List[MacroDataItem]:
        """Parse tool messages → MacroDataItem list."""
        items: List[MacroDataItem] = []
        messages = state.get("messages", [])
        queries_seen: List[str] = []  # track to map source

        for i, msg in enumerate(messages):
            if not isinstance(msg, ToolMessage):
                # Track queries from AIMessage tool_calls
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                    for tc in (msg.tool_calls or []):
                        if tc.get("name") == "web_search":
                            args = tc.get("args", {})
                            queries_seen.append(args.get("query", ""))
                continue

            tool_name = getattr(msg, "name", "unknown")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            if tool_name == "web_search":
                items.extend(self._parse_web_search(content))
            elif tool_name == "scrape_url":
                items.extend(self._parse_scrape_url(content, msg))
            elif tool_name == "get_economic_data":
                items.extend(self._parse_econ_data(content))
            elif tool_name == "mcp_call":
                items.extend(self._parse_mcp_result(content))

        return items

    def _extract_queries(self, state: dict) -> List[str]:
        queries: List[str] = []
        for msg in state.get("messages", []):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                for tc in (msg.tool_calls or []):
                    if tc.get("name") == "web_search":
                        q = tc.get("args", {}).get("query", "")
                        if q:
                            queries.append(q)
        return queries

    def _parse_web_search(self, content: str) -> List[MacroDataItem]:
        try:
            results = json.loads(content)
            if not isinstance(results, list):
                return []
            items = []
            for r in results:
                if isinstance(r, dict):
                    items.append(MacroDataItem(
                        title=r.get("title", ""),
                        content=r.get("snippet", ""),
                        source="web_search",
                        timestamp=datetime.now(),
                        category="news",
                        url=r.get("url", ""),
                    ))
            return items
        except Exception:
            return []

    def _parse_scrape_url(self, content: str, msg: Any) -> List[MacroDataItem]:
        if content.startswith("Error"):
            return []
        return [MacroDataItem(
            title="Scraped Content",
            content=content[:800],
            source="scrape_url",
            timestamp=datetime.now(),
            category="web",
        )]

    def _parse_econ_data(self, content: str) -> List[MacroDataItem]:
        try:
            data = json.loads(content)
            if "error" in data:
                return []
            indicator = data.get("indicator", "")
            country = data.get("country", "")
            points = data.get("data", [])
            summary = f"{indicator} ({country}): " + "; ".join(
                f"{p['year']}={p['value']}" for p in points if p.get("value")
            )
            return [MacroDataItem(
                title=f"Economic Data: {indicator} ({country})",
                content=summary,
                source="world_bank_api",
                timestamp=datetime.now(),
                category="economic_data",
            )]
        except Exception:
            return []

    def _parse_mcp_result(self, content: str) -> List[MacroDataItem]:
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "error" not in data:
                return [MacroDataItem(
                    title=data.get("title", "MCP Result"),
                    content=str(data.get("content", data))[:800],
                    source="mcp",
                    timestamp=datetime.now(),
                    category="mcp",
                    url=data.get("url"),
                )]
        except Exception:
            pass
        return []


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 4 — PHASE 2: MacroAnalyst (LocalLLMBackend + KV-cache)
# ═══════════════════════════════════════════════════════════════════════════

_ANALYSIS_SYSTEM_PROMPT = """\
You are a senior macroeconomic analyst for a quantitative trading fund.
Analyze the provided data to identify:

1. Key macroeconomic trends and their market implications
2. Central bank policy trajectory and rate outlook
3. Risk factors (geopolitical, financial stability, currency)
4. Cross-asset implications (equities, bonds, FX, commodities)
5. DIRECTION HINT: One actionable trading direction (single sentence)

Format your DIRECTION HINT as:
DIRECTION HINT: <one concise actionable direction for alpha factor research>
"""


class MacroAnalyst:
    """
    Phase 2: analisis latent via LocalLLMBackend.

    Menggunakan analyze_incremental():
    - Data di-chunk (chunk_size items per pass)
    - Setiap chunk di-encode sebagai latent pass (kv_only)
    - Chunk terakhir: kv_and_text (generate summary + simpan KV-cache)
    - Seluruh KV-cache menjadi "macro context" untuk planning.py
    """

    def __init__(self, llm: LocalLLMBackend, config: MacroConfig) -> None:
        self.llm = llm
        self.config = config

    def analyze(
        self,
        items: List[MacroDataItem],
        past_kv: Optional[KVCache] = None,
        conv_id: Optional[str] = None,
    ) -> Tuple[str, Optional[KVCache], Optional[torch.Tensor]]:
        """
        Analyze collected items via incremental latent reasoning.

        Returns:
            (summary_text, kv_cache, hidden_state)
        """
        if not items:
            return "No data available for analysis.", None, None

        _cid = conv_id or f"makro_{uuid.uuid4().hex[:8]}"
        chunk_size = self.config.chunk_size
        chunks = [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]
        current_kv = past_kv

        # ── Latent passes (all chunks except last) ────────────────────────
        for step_idx, chunk in enumerate(chunks[:-1]):
            messages = self._build_messages(chunk)
            result: LLMResult = self.llm.run(
                messages=messages,
                mode="kv_only",
                past_key_values=current_kv,
                conv_id=_cid,
                step=step_idx,
                role="makro_analyst",
            )
            current_kv = result.kv_cache

        # ── Final chunk: generate text + KV-cache ─────────────────────────
        final_messages = self._build_messages(chunks[-1])
        final_result: LLMResult = self.llm.run(
            messages=final_messages,
            mode=self.config.analysis_mode,
            past_key_values=current_kv,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            conv_id=_cid,
            step=len(chunks) - 1,
            role="makro_analyst",
        )

        return (
            final_result.text or "",
            final_result.kv_cache,
            final_result.hidden_last,
        )

    def _build_messages(
        self,
        items: List[MacroDataItem],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        context = "\n\n".join(
            f"--- Item {i} ---\n{item.to_context_str()}"
            for i, item in enumerate(items, 1)
        )
        user_prompt = (
            f"Analyze the following {len(items)} macro-economic data points:\n\n{context}"
        )
        return self.llm.build_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt or _ANALYSIS_SYSTEM_PROMPT,
        )


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 5 — MacroExternalAgent (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

class MacroExternalAgent(ExternalAgentBase):
    """
    Macro-Economic Intelligence Agent.

    Combines Phase 1 (LangGraph search) and Phase 2 (KV-cache analysis).

    Flow:
        strategy.build_queries()
            ↓
        MacroSearchGraph.search()    ← Phase 1: API model + tools
            ↓ List[MacroDataItem]
        MacroAnalyst.analyze()       ← Phase 2: local Qwen + KV-cache
            ↓
        ExternalInsight
            ↓
        planning.py  &  AlphaAgentLoop
    """

    agent_name = "makro"

    def __init__(
        self,
        llm_backend: LocalLLMBackend,
        api_model: Optional["BaseChatModel"] = None,
        config: Optional[MacroConfig] = None,
        tools_list: Optional[List] = None,
        mcp_caller: Optional[Callable] = None,
        monitor: Optional[Monitor] = None,
        strategy_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            llm_backend : LocalLLMBackend instance untuk Phase 2 (KV-cache analysis)
            api_model   : LangChain BaseChatModel untuk Phase 1 (tool calling).
                          Jika None, Phase 1 dilewati dan hanya search manual yang digunakan.
            config      : MacroConfig
            tools_list  : Override tool list untuk MacroSearchGraph
            mcp_caller  : callable(tool_name, params) → untuk MCP tool
            monitor     : Monitor instance (dibuat otomatis jika None)
            strategy_path: Path ke file strategi JSON
        """
        cfg = config or MacroConfig()
        super().__init__(
            strategy_path=strategy_path or Path(cfg.strategy_path)
        )
        self.config = cfg
        self.llm_backend = llm_backend
        self.analyst = MacroAnalyst(llm_backend, cfg)

        # Phase 1: optional
        self.search_graph: Optional[MacroSearchGraph] = None
        if api_model is not None and _HAS_LANGGRAPH:
            self.search_graph = MacroSearchGraph(
                api_model=api_model,
                config=cfg,
                tools_list=tools_list,
                mcp_caller=mcp_caller,
            )

        # Monitor
        if monitor is not None:
            self.monitor = monitor
        elif cfg.enable_monitor:
            self.monitor = Monitor(
                MonitorConfig(
                    log_dir=cfg.log_dir,
                    log_file_prefix="makro_agent",
                )
            )
        else:
            self.monitor = None

    # ── Main entry ────────────────────────────────────────────────────────

    def run(
        self,
        past_kv: Optional[KVCache] = None,
        extra_queries: Optional[List[str]] = None,
        **kwargs,
    ) -> ExternalInsight:
        """
        Full pipeline: Phase 1 (search) + Phase 2 (analyze).

        Args:
            past_kv      : KV-cache dari agent lain (latent chaining)
            extra_queries: Tambahan queries di luar SearchStrategy
        """
        t_start = time.time()
        conv_id = f"makro_{uuid.uuid4().hex[:8]}"

        if self.monitor:
            self.monitor.log_separator("═", 70)
            self.monitor.log(f"MacroExternalAgent.run()  conv_id={conv_id}")

        # ── Phase 1: Search ───────────────────────────────────────────────
        items, queries_issued = self._phase1_search(extra_queries or [])

        if self.monitor:
            self.monitor.log(f"Phase 1 complete: {len(items)} items, {len(queries_issued)} queries")

        # ── Phase 2: Analyze ──────────────────────────────────────────────
        summary, kv_cache, hidden_state = self._phase2_analyze(items, past_kv, conv_id)

        if self.monitor:
            self.monitor.log(f"Phase 2 complete: summary_len={len(summary)}, has_kv={kv_cache is not None}")

        # ── Build ExternalInsight ─────────────────────────────────────────
        direction_hint = self._extract_direction_hint(summary)

        insight = ExternalInsight(
            source_agent="makro",
            summary=summary,
            direction_hint=direction_hint,
            kv_cache=kv_cache,
            hidden_state=hidden_state,
            search_queries=queries_issued,
            n_documents_collected=len(items),
            conv_id=conv_id,
            processing_time_s=time.time() - t_start,
            metadata={
                "n_chunks": max(1, len(items) // self.config.chunk_size),
                "kv_seq_len": _past_length(kv_cache) if kv_cache else 0,
            },
        )

        if self.monitor:
            self.monitor.log_variable("insight_dict", insight.to_dict())

        return insight

    # ── Phase implementations ─────────────────────────────────────────────

    def _phase1_search(
        self, extra_queries: List[str]
    ) -> Tuple[List[MacroDataItem], List[str]]:
        """Phase 1: gather data via MacroSearchGraph OR fallback."""
        # Build queries from strategy
        queries = self.strategy.build_queries() + extra_queries

        if self.search_graph is not None:
            # LangGraph path (API model)
            try:
                items, queries_issued = self.search_graph.search(queries)
                return items, queries_issued
            except Exception as e:
                if self.monitor:
                    self.monitor.log(f"MacroSearchGraph failed, using fallback: {e}")
                return self._fallback_search(queries), queries[:3]
        else:
            # Fallback: manual requests (no API model)
            return self._fallback_search(queries), queries[:3]

    def _fallback_search(self, queries: List[str]) -> List[MacroDataItem]:
        """Fallback ketika LangGraph tidak tersedia: pakai web_search langsung."""
        from eksternal.tools.web_tools import make_web_search_tool
        search_fn = make_web_search_tool(
            provider=self.config.search_provider,
            api_key=self.config.search_api_key,
        )
        items: List[MacroDataItem] = []
        for q in queries[:self.config.max_search_iterations]:
            try:
                result_json = search_fn.invoke({"query": q, "max_results": self.config.max_items_per_query})
                results = json.loads(result_json)
                if isinstance(results, list):
                    for r in results:
                        items.append(MacroDataItem(
                            title=r.get("title", ""),
                            content=r.get("snippet", ""),
                            source="fallback_search",
                            timestamp=datetime.now(),
                            category="news",
                            url=r.get("url", ""),
                        ))
            except Exception:
                continue
        return items

    def _phase2_analyze(
        self,
        items: List[MacroDataItem],
        past_kv: Optional[KVCache],
        conv_id: str,
    ) -> Tuple[str, Optional[KVCache], Optional[torch.Tensor]]:
        """Phase 2: latent reasoning via MacroAnalyst."""
        if not items:
            return "Insufficient data for macro analysis.", None, None
        return self.analyst.analyze(items, past_kv, conv_id)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_direction_hint(summary: str) -> str:
        """
        Extract DIRECTION HINT dari summary.
        Falls back ke first meaningful sentence jika tidak ditemukan.
        """
        # Try explicit label
        match = re.search(
            r"DIRECTION HINT:\s*(.+?)(?:\n|$)", summary, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        # Fallback: first sentence under 150 chars
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        for s in sentences:
            s = s.strip()
            if 20 < len(s) < 150:
                return s

        return summary[:120].strip()

    # ── Strategy update (3.B) ─────────────────────────────────────────────

    def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """
        Update search strategy setelah semua iterasi evolusi selesai.

        Dipanggil oleh factor_mining.py setelah EvolutionController selesai.

        Args:
            feedback: {
                "topic_performance":  Dict[str, float],  # topic → avg IC/Sharpe
                "successful_queries": List[str],          # queries yang berguna
                "top_directions":     List[str],          # directions terbaik
            }
        """
        topic_perf: Dict[str, float] = feedback.get("topic_performance", {})
        successful_queries: List[str] = feedback.get("successful_queries", [])

        # Update topic weights berdasarkan performance
        # delta = (score - baseline) * scale
        baseline = 0.05  # IC baseline
        scale = 3.0
        for topic, score in topic_perf.items():
            delta = (score - baseline) * scale
            self.strategy.update_topic_weight(topic, delta)

        # Tambah successful queries ke additional_queries
        for q in successful_queries:
            if q not in self.strategy.additional_queries:
                self.strategy.additional_queries.append(q)
        # Keep latest 10
        self.strategy.additional_queries = self.strategy.additional_queries[-10:]

        self.save_strategy()

        if self.monitor:
            self.monitor.log(
                f"Strategy updated (count={self.strategy.update_count}). "
                f"Top topics: {self.strategy.top_topics(3)}"
            )

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "has_search_graph": self.search_graph is not None,
            "strategy_update_count": self.strategy.update_count,
            "strategy_top_topics": self.strategy.top_topics(5),
            "config": {
                "max_search_iterations": self.config.max_search_iterations,
                "analysis_mode": self.config.analysis_mode,
                "chunk_size": self.config.chunk_size,
            },
        }

    def close(self) -> None:
        if self.monitor:
            self.monitor.close()


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 6 — FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def create_macro_agent(
    llm_backend: LocalLLMBackend,
    api_model: Optional["BaseChatModel"] = None,
    search_provider: str = "duckduckgo",
    search_api_key: Optional[str] = None,
    max_search_iterations: int = 6,
    mcp_caller: Optional[Callable] = None,
    **config_kwargs: Any,
) -> MacroExternalAgent:
    """
    Factory untuk MacroExternalAgent.

    Examples
    --------
    # Minimal (only Phase 2, no LangGraph):
    agent = create_macro_agent(llm_backend=backend)

    # With Claude as Phase 1 model:
    from langchain_anthropic import ChatAnthropic
    model = ChatAnthropic(model="claude-3-5-haiku-20241022")
    agent = create_macro_agent(llm_backend=backend, api_model=model)

    # With OpenAI + Tavily:
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model="gpt-4o-mini")
    agent = create_macro_agent(
        llm_backend=backend,
        api_model=model,
        search_provider="tavily",
        search_api_key="tvly-xxx",
    )
    """
    config = MacroConfig(
        search_provider=search_provider,
        search_api_key=search_api_key,
        max_search_iterations=max_search_iterations,
        **config_kwargs,
    )
    return MacroExternalAgent(
        llm_backend=llm_backend,
        api_model=api_model,
        config=config,
        mcp_caller=mcp_caller,
    )
