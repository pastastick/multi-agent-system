"""prod/config.py — konfigurasi terpusat pipeline produksi. Lihat DESIGN.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROD_DIR = Path(__file__).resolve().parent
RESULTS = PROD_DIR / "results"
PROMPTS_YAML = PROD_DIR / "prompts.yaml"

MARKET_CONTEXT = "Liquid equities, daily bars, 2018-2021 train segment."

# Seed/command gen-0 (root KV belum punya direction → diberi sebagai teks).
DEFAULT_SEED_DIRECTION = (
    "Open exploration: propose one original, conditional cross-sectional return "
    "mechanism grounded in a clear price/volume driver. Prefer a mechanism with a "
    "named regime gate over a bare textbook label."
)


@dataclass
class RunConfig:
    """Parameter satu run loop evolusi (segmented KV; restart di feedback)."""
    generations: int = 3                 # total ronde construct (gen0 seed + sisanya evolusi)
    latent_steps: int = 10               # 0 = tanpa realignment (dipakai mode 'text')
    decode_temperature: float = 0.7
    max_new_tokens: int = 10000
    # 'kv'   = NO-CROP latent KV (produksi). 'text' = full-text handoff, kv none,
    #          latent_steps dipaksa 0 (baseline A/B, DESIGN.md §7).
    transfer_mode: str = "kv"
    director: str = "mutation"           # 'mutation' (1 parent) | 'crossover' (>=2 parent)
    backtest_mode: str = "mock"          # 'mock' (no Qlib) | 'real' (Qlib adapter, F3+)
    verbose: bool = True                 # log terminal (durasi, token, gate reason)
    market_context: str = MARKET_CONTEXT
    seed_direction: str = DEFAULT_SEED_DIRECTION
    out_dir: Path = field(default=RESULTS)

    @property
    def effective_latent_steps(self) -> int:
        return 0 if self.transfer_mode == "text" else self.latent_steps

    @property
    def handoff(self) -> str:
        """Cabang Jinja default untuk node non-root."""
        return "text" if self.transfer_mode == "text" else "kv"

    def __post_init__(self) -> None:
        if self.transfer_mode not in ("kv", "text"):
            raise ValueError(f"transfer_mode harus 'kv' atau 'text', dapat {self.transfer_mode!r}")
        if self.director not in ("mutation", "crossover"):
            raise ValueError(f"director harus 'mutation' atau 'crossover', dapat {self.director!r}")
        if self.backtest_mode not in ("mock", "real"):
            raise ValueError(f"backtest_mode harus 'mock' atau 'real', dapat {self.backtest_mode!r}")
        if self.generations < 1:
            raise ValueError("generations >= 1")
