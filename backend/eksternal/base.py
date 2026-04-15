"""
base.py — External Agent Base Classes
======================================

Interface antara external agents (makro, news, fundamental, technical)
dan pipeline internal (planning.py + AlphaAgentLoop).

Routing insight:
    ExternalInsight.kv_cache       → planning.generate_parallel_directions()
                                     sebagai past_key_values (latent context)
    ExternalInsight.summary        → AlphaAgentLoop via external_context
                                     (context tambahan di factor_propose)
    ExternalInsight.direction_hint → seed dalam generate_parallel_directions()
    ExternalInsight.search_queries → SearchStrategy.update_from_feedback()
                                     untuk trajectory improvement (3.B)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# KVCache type alias — kompatibel dengan new_client.py
# Tuple[Tuple[torch.Tensor, torch.Tensor], ...]  per-layer (key, value)
try:
    from llm.client import KVCache
except ImportError:
    KVCache = Any  # graceful fallback sebelum dependency tersedia


# ═══════════════════════════════════════════════════════════════════════════
# ExternalInsight — output tunggal dari setiap external agent
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExternalInsight:
    """
    Output dari external agent.

    ExternalInsight
    .kv_cache       ──────────────► planning.py
                                   past_key_values
    .summary        ──────────────► AlphaAgentLoop
                                    factor_propose context
    .direction_hint ──────────────► generate_parallel_    
                                   directions seeding
    .search_queries ──────────────► update_strategy()
                                    (setelah iterasi)
    """

    source_agent: str                             # "makro" | "news" | "fundamental" | "technical"
    summary: str                                  # full analysis → factor_propose context
    direction_hint: str                           # satu kalimat → seed direction generation

    # ── Latent payloads (Phase 2 output) ──────────────────────────────────
    kv_cache: Optional[Any] = None               # KVCache → planning past_key_values
    hidden_state: Optional[torch.Tensor] = None  # [B, d]  → embedding communication
    latent_vecs: Optional[torch.Tensor] = None   # [steps, d]

    # ── Search trace (untuk strategy update setelah evolusi) ──────────────
    search_queries: List[str] = field(default_factory=list)
    n_documents_collected: int = 0

    # ── Metadata ──────────────────────────────────────────────────────────
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    conv_id: str = ""
    processing_time_s: float = 0.0

    # -- properties --------------------------------------------------------

    @property
    def has_kv(self) -> bool:
        return self.kv_cache is not None

    @property
    def has_hidden(self) -> bool:
        return self.hidden_state is not None

    # -- helpers -----------------------------------------------------------

    def to_context_str(self) -> str:
        """Format untuk disertakan dalam LLM prompt (planning / factor_propose)."""
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"=== {self.source_agent.upper()} EXTERNAL CONTEXT [{ts}] ===\n"
            f"{self.summary}\n"
            f"{'=' * 40}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialisasi ringan (tanpa tensor) untuk logging."""
        return {
            "source_agent": self.source_agent,
            "summary": self.summary[:500],
            "direction_hint": self.direction_hint,
            "has_kv_cache": self.has_kv,
            "search_queries": self.search_queries,
            "n_documents": self.n_documents_collected,
            "timestamp": self.timestamp.isoformat(),
            "conv_id": self.conv_id,
            "processing_time_s": round(self.processing_time_s, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SearchStrategy — state pencarian yang persisten lintas run
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SearchStrategy:
    """
    Strategi pencarian yang di-update setelah iterasi evolusi selesai.
    Disimpan sebagai JSON untuk persistensi lintas run.

    update_from_feedback() dipanggil oleh ExternalAgentBase.update_strategy()
    yang dipanggil oleh factor_mining.py SETELAH semua iterasi selesai (3.B).
    """

    topic_weights: Dict[str, float] = field(default_factory=lambda: {
        "monetary_policy":  1.0,
        "inflation":        1.0,
        "gdp_growth":       1.0,
        "employment":       0.8,
        "geopolitical_risk": 0.8,
        "commodity_prices": 0.9,
        "currency_markets": 0.9,
        "credit_spreads":   0.7,
        "central_bank":     1.0,
        "trade_balance":    0.7,
    })
    additional_queries: List[str] = field(default_factory=list)
    #* query tambahan yang di inject dari feedback
    
    focus_regions: List[str] = field(
        default_factory=lambda: ["US", "EU", "China", "Global"]
    )
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    update_count: int = 0

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "topic_weights": self.topic_weights,
                    "additional_queries": self.additional_queries,
                    "focus_regions": self.focus_regions,
                    "updated_at": self.updated_at,
                    "update_count": self.update_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )#* simpan semua field ke JSON

    @classmethod
    def load(cls, path: Path) -> "SearchStrategy":
        if not path.exists():
            return cls() #* file belum ada -> pakai default
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        return cls(
            topic_weights=data.get("topic_weights", {}),
            additional_queries=data.get("additional_queries", []),
            focus_regions=data.get("focus_regions", ["US", "EU", "China", "Global"]),
            updated_at=data.get("updated_at", ""),
            update_count=data.get("update_count", 0),
        )

    # -- query helpers -----------------------------------------------------
    def top_topics(self, n: int = 5) -> List[str]:
        """Return top-N topics by weight."""
        return sorted(
            self.topic_weights, key=lambda t: self.topic_weights[t], reverse=True
        )[:n] #* return N topik dengan bobot tertinggi

    def build_queries(self, base_topic: str = "macro economy") -> List[str]:
        """Build search queries from current strategy."""
        top = self.top_topics(5)
        queries = [f"global {t} outlook {datetime.now().year}" for t in top]
        queries += [
            f"{region} {base_topic} news" for region in self.focus_regions[:2]
        ]
        queries += self.additional_queries[:3]
        return queries #* tambah query tambahan dari feedback

    def update_topic_weight(self, topic: str, delta: float) -> None:
        """Adjust topic weight — clamped to [0.1, 3.0]."""
        current = self.topic_weights.get(topic, 1.0)
        self.topic_weights[topic] = max(0.1, min(3.0, current + delta))


# ═══════════════════════════════════════════════════════════════════════════
# ExternalAgentBase — ABC untuk semua external agents
# ═══════════════════════════════════════════════════════════════════════════

class ExternalAgentBase(ABC):
    """
    Base class untuk: MacroExternalAgent, NewsExternalAgent,
    FundamentalExternalAgent, TechnicalExternalAgent.

    Lifecycle:
        1. run()             — gather data + analyze → ExternalInsight
        2. (evolution loop runs)
        3. update_strategy() — di-panggil oleh factor_mining.py SETELAH
                               semua iterasi selesai (jawaban 3.B)
        4. save_strategy()   — otomatis di dalam update_strategy()
    """

    agent_name: str = "base"

    def __init__(self, strategy_path: Optional[Path] = None) -> None:
        #* path file JSON untuk menyimpan strategi
        self.strategy_path = (
            strategy_path
            or Path(f"./.strategy/{self.agent_name}_strategy.json")
        )
        self.strategy: SearchStrategy = SearchStrategy.load(self.strategy_path)
        #* load strategi dari dist

    @abstractmethod
    def run(self, past_kv: Optional[Any] = None, **kwargs) -> ExternalInsight:
        """
        Run the agent. Return ExternalInsight.

        Args:
            past_kv: KV-cache dari agent sebelumnya (optional latent chaining).
        """
        ...

    @abstractmethod
    def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """
        Update search strategy berdasarkan feedback
        (setelah semua iterasi evolusi selesai (3.B)).

        Args:
            feedback: {
                "top_directions":     List[str],    # directions terbaik
                "topic_performance":  Dict[str, float],  # topic → avg IC/Sharpe
                "successful_queries": List[str],    # queries yang berkontribusi
            }
        """
        ...

    def save_strategy(self) -> None:
        """Persist strategy ke disk."""
        self.strategy.updated_at = datetime.now().isoformat()
        self.strategy.update_count += 1
        self.strategy.save(self.strategy_path)
