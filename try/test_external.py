"""
Test LLM calls untuk external agents + manager synthesizer.

Semua prompt di sini INLINE di kode Python (bukan di YAML):
  - makro_analyze       eksternal/makro.py: _ANALYSIS_SYSTEM_PROMPT + dynamic user
  - news_analyze        eksternal/news.py:  inline system + inline user
  - fundamental_analyze eksternal/fundamental.py: inline system + inline user
  - technical_analyze   eksternal/technical.py: inline system + inline user
  - manager_synthesize  eksternal/manager.py: _MANAGER_SYSTEM_PROMPT + summary gabungan

Prompt di-copy verbatim dari source. User prompt dibangun dari data mock di fixtures.
"""

from __future__ import annotations

from .common import run_case
from . import fixtures as fx


# ═════════════════════════════════════════════════════════════════════════
# System prompts — COPY VERBATIM dari backend source code.
# Kalau di backend berubah, update juga di sini agar test relevan.
# ═════════════════════════════════════════════════════════════════════════

MAKRO_SYSTEM_PROMPT = """\
You are a macro-economic analyst synthesizing global macro data for a
quantitative factor mining system. Analyze the provided macro data points
and produce a concise, structured summary.

Your output MUST contain:
1. Current macro regime (one of: risk-on / risk-off / neutral / transition).
2. Key drivers (2–4 bullet points with evidence from the data).
3. Implication for equity factors (one paragraph; mention: momentum vs
   value vs quality; beta tilt; sector rotation; volatility regime).
4. DIRECTION HINT: one sentence proposing a factor strategy direction.

Be concise. Avoid generic disclaimers."""

NEWS_SYSTEM_PROMPT = """\
You are a financial news analyst. Analyze the following news items and provide:
1. Key market-moving events and their potential impact
2. Dominant market sentiment (bullish/bearish/neutral) with evidence
3. Event-driven trading signals for quantitative factor construction
4. DIRECTION HINT: one sentence suggesting a factor strategy direction based on current news flow
Be concise. Focus on actionable signals."""

FUNDAMENTAL_SYSTEM_PROMPT = """\
You are a fundamental analysis expert. Analyze the following financial data and provide:
1. Key earnings and valuation trends
2. Sectors showing strength or weakness
3. Notable fundamental signals for factor construction
4. DIRECTION HINT: one sentence suggesting a factor strategy direction
Be concise and quantitative."""

TECHNICAL_SYSTEM_PROMPT = """\
You are a technical analysis expert. Analyze the following market data and provide:
1. Key technical signals (momentum, trend, volume, volatility)
2. Notable chart patterns or indicator divergences
3. Cross-asset correlation signals
4. Actionable quantitative factor ideas based on technical analysis
5. DIRECTION HINT: one sentence suggesting a factor strategy direction based on current technical signals
Be concise. Focus on quantifiable signals."""

MANAGER_SYSTEM_PROMPT = """\
You are a senior investment strategist synthesizing multiple research streams
for a quantitative alpha factor mining system.

You receive analyses from specialized agents:
- MAKRO: macroeconomic outlook, central bank policy, rates, GDP, inflation
- NEWS:  breaking news, market sentiment, event-driven signals
- FUNDAMENTAL: company fundamentals, earnings, valuations, sector rotations
- TECHNICAL: price action, momentum, mean reversion, volatility regimes

Your task:
1. Synthesize all four analyses into a single unified market view.
2. Identify the DOMINANT theme across all streams.
3. Highlight any contradictions between streams.
4. Propose ONE specific factor-mining direction that leverages the
   combined insights (e.g., "short-term reversal in high-beta names during
   risk-off regime confirmed by negative news flow").

Respond in concise bullet points, then close with a single-sentence
DIRECTION HINT."""


# ═════════════════════════════════════════════════════════════════════════
# Test functions
# ═════════════════════════════════════════════════════════════════════════

def test_makro_analyze():
    user_prompt = f"Analyze the following 5 macro-economic data points:\n\n{fx.MAKRO_DATA_TEXT}"
    return run_case(
        group="external", case="makro_analyze",
        system_prompt=MAKRO_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_mode=False,
    )


def test_news_analyze():
    user_prompt = f"Analyze these financial news items:\n\n{fx.NEWS_DATA_TEXT}"
    return run_case(
        group="external", case="news_analyze",
        system_prompt=NEWS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_mode=False,
    )


def test_fundamental_analyze():
    user_prompt = f"Analyze these fundamental data points:\n\n{fx.FUNDAMENTAL_DATA_TEXT}"
    return run_case(
        group="external", case="fundamental_analyze",
        system_prompt=FUNDAMENTAL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_mode=False,
    )


def test_technical_analyze():
    user_prompt = f"Analyze these technical analysis data points:\n\n{fx.TECHNICAL_DATA_TEXT}"
    return run_case(
        group="external", case="technical_analyze",
        system_prompt=TECHNICAL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_mode=False,
    )


def test_manager_synthesize():
    """Manager merangkum output 4 agent di atas. Kita mock outputnya di fx.AGENT_SUMMARIES."""
    blocks = "\n\n".join(f"[{k}]\n{v}" for k, v in fx.AGENT_SUMMARIES.items())
    user_prompt = (
        "Synthesize the following agent analyses into a unified market view "
        "and propose a factor-mining direction:\n\n" + blocks
    )
    return run_case(
        group="external", case="manager_synthesize",
        system_prompt=MANAGER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_mode=False,
    )


CASES = {
    "makro_analyze": test_makro_analyze,
    "news_analyze": test_news_analyze,
    "fundamental_analyze": test_fundamental_analyze,
    "technical_analyze": test_technical_analyze,
    "manager_synthesize": test_manager_synthesize,
}
