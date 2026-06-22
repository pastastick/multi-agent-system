"""prod/config.py — konfigurasi terpusat pipeline produksi. Lihat DESIGN.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROD_DIR = Path(__file__).resolve().parent
RESULTS = PROD_DIR / "results"
PROMPTS_YAML = PROD_DIR / "prompts.yaml"

MARKET_CONTEXT = "Liquid equities, daily bars, 2018-2021 train segment."

# DSL library untuk agent repair (disuntik sbg {{ function_lib }}). Sumber tunggal
# nama+arity fungsi agar repair bisa mengganti fungsi/argumen yang tak sesuai.
FUNCTION_LIB = """\
VARIABLES (the only data leaves, case-insensitive): $open $high $low $close $volume $return
ARITY IS STRICT — each function takes exactly the arguments shown. Cross-sectional
functions take exactly ONE argument with no window; for a rolling rank use TS_RANK(A,n)
not RANK. A,B = sub-expression; C = condition; n,p = whole-number day windows; q = 0..1.

Cross-sectional (1 arg, no window): RANK(A) ZSCORE(A) MEAN(A) STD(A) SKEW(A) KURT(A)
  MAX(A) MIN(A) MEDIAN(A)
Time-series (series + window): DELTA(A,n) DELAY(A,n) TS_MEAN(A,n) TS_SUM(A,n)
  TS_RANK(A,n) TS_ZSCORE(A,n) TS_MEDIAN(A,n) TS_PCTCHANGE(A,p) TS_MIN(A,n) TS_MAX(A,n)
  TS_ARGMAX(A,n) TS_ARGMIN(A,n) TS_QUANTILE(A,p,q) TS_STD(A,n) TS_VAR(A,p)
  TS_CORR(A,B,n) TS_COVARIANCE(A,B,n) TS_MAD(A,n) PERCENTILE(A,q,p) HIGHDAY(A,n)
  LOWDAY(A,n) SUMAC(A,n)
Moving-average/smoothing: SMA(A,n,m) WMA(A,n) EMA(A,n) DECAYLINEAR(A,d)
Math (1 arg unless noted): PROD(A,n) LOG(A) SQRT(A) POW(A,n) SIGN(A) EXP(A) ABS(A)
  MAX(A,B) MIN(A,B) INV(A) FLOOR(A)
Conditional/logical: (C) ? (A) : (B) ; (C1) && (C2) ; (C1) || (C2) ; COUNT(C,n)
  SUMIF(A,n,C) FILTER(A,C)
Regression/residual: SEQUENCE(n) [only inside REGBETA/REGRESI as B] ; REGBETA(A,B,n)
  REGRESI(A,B,n)
Technical: RSI(A,n) MACD(A,short,long) BB_MIDDLE(A,n) BB_UPPER(A,n) BB_LOWER(A,n)

LEGAL EXPRESSION: only the six variables + exact function names above; mind TS_ prefix
(TS_STD rolling vs STD cross-sectional); balanced brackets; at least one variable.
Arithmetic ONLY inside expressions: + - * / (division only '/')."""

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
