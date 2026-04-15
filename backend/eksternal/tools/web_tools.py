"""
web_tools.py — LangGraph-compatible tools for MacroSearchGraph
===============================================================

Tools yang digunakan oleh MacroSearchGraph (Phase 1 — tool calling dengan
API model). Setiap tool adalah LangChain @tool yang bisa di-bind ke model.

Tools:
    web_search         — search web (DuckDuckGo default / Tavily / SerpAPI)
    scrape_url         — scrape content dari URL
    get_economic_data  — ambil data indikator ekonomi (World Bank API, free)
    mcp_call           — bridge ke MCP tool (scraping / web control)
    mark_done          — sinyal bahwa data sudah cukup terkumpul

Usage:
    tools = get_default_tools(search_provider="duckduckgo")
    # atau dengan MCP:
    tools = get_default_tools(mcp_caller=mcp_client.call_tool)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

# ── Optional dependencies ─────────────────────────────────────────────────
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _requests = None  # type: ignore[assignment]
    _HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment,misc]
    _HAS_BS4 = False

try:
    from duckduckgo_search import DDGS
    _HAS_DDG = True
except ImportError:
    DDGS = None  # type: ignore[assignment,misc]
    _HAS_DDG = False

try:
    from langchain_core.tools import tool
    _HAS_LANGCHAIN = True
except ImportError:
    # Fallback decorator jika LangChain belum terpasang
    def tool(func):  # type: ignore[misc]
        return func
    _HAS_LANGCHAIN = False


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 1 — web_search
# ═══════════════════════════════════════════════════════════════════════════

def make_web_search_tool(
    provider: str = "duckduckgo",
    api_key: Optional[str] = None,
    max_results_cap: int = 10,
):
    """
    Factory untuk web_search tool.

    provider: "duckduckgo" (default, free) | "tavily" | "serpapi"
    api_key : diperlukan untuk tavily / serpapi
    """

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """
        Search the web for macroeconomic news and data.
        Returns a JSON list of search results with title, url, and snippet.

        Args:
            query      : Search query string (be specific, e.g. 'Fed rate decision impact 2025')
            max_results: Number of results (max 10)
        """
        n = min(max_results, max_results_cap)

        # ── DuckDuckGo (free, no key) ─────────────────────────────────────
        if provider == "duckduckgo":
            if not _HAS_DDG:
                return json.dumps({
                    "error": "duckduckgo-search not installed. pip install duckduckgo-search"
                })
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=n))
                results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:400],
                    }
                    for r in raw
                ]
                return json.dumps(results, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)})

        # ── Tavily ────────────────────────────────────────────────────────
        if provider == "tavily":
            if not api_key:
                return json.dumps({"error": "api_key required for Tavily"})
            try:
                from langchain_community.tools.tavily_search import TavilySearchResults
                tavily = TavilySearchResults(api_key=api_key, max_results=n)
                results = tavily.invoke({"query": query})
                return json.dumps(results, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)})

        # ── SerpAPI ───────────────────────────────────────────────────────
        if provider == "serpapi":
            if not _HAS_REQUESTS or not api_key:
                return json.dumps({"error": "requests + api_key required for SerpAPI"})
            try:
                resp = _requests.get(
                    "https://serpapi.com/search",
                    params={"q": query, "api_key": api_key, "num": n},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "snippet": r.get("snippet", "")[:400],
                    }
                    for r in data.get("organic_results", [])[:n]
                ]
                return json.dumps(results, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)})

        return json.dumps({"error": f"Unknown provider: {provider}"})

    return web_search


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 2 — scrape_url
# ═══════════════════════════════════════════════════════════════════════════

@tool
def scrape_url(url: str, max_chars: int = 3000) -> str:
    """
    Scrape and extract text content from a web page URL.
    Returns extracted text (truncated to max_chars).

    Args:
        url      : The full URL to scrape
        max_chars: Maximum characters to return (default 3000)
    """
    if not _HAS_REQUESTS:
        return "Error: requests not installed. pip install requests"
    if not _HAS_BS4:
        return "Error: beautifulsoup4 not installed. pip install beautifulsoup4"

    try:
        resp = _requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MacroAgent/1.0)"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines)

        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    except Exception as e:
        return f"Error scraping {url}: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 3 — get_economic_data
# ═══════════════════════════════════════════════════════════════════════════

# World Bank indicator codes (free API, no key required)
_WB_INDICATORS: Dict[str, str] = {
    "gdp":              "NY.GDP.MKTP.CD",
    "gdp_growth":       "NY.GDP.MKTP.KD.ZG",
    "cpi":              "FP.CPI.TOTL.ZG",
    "inflation":        "FP.CPI.TOTL.ZG",
    "unemployment":     "SL.UEM.TOTL.ZS",
    "trade_balance":    "BN.CAB.XOKA.GD.ZS",
    "current_account":  "BN.CAB.XOKA.GD.ZS",
    "interest_rate":    "FR.INR.RINR",
    "debt_gdp":         "GC.DOD.TOTL.GD.ZS",
    "fdi":              "BX.KLT.DINV.WD.GD.ZS",
}


@tool
def get_economic_data(indicator: str, country: str = "US", periods: int = 5) -> str:
    """
    Fetch recent macroeconomic indicator data from World Bank (free, no API key).
    Supported: gdp, gdp_growth, cpi, inflation, unemployment, trade_balance,
               current_account, interest_rate, debt_gdp, fdi.

    Args:
        indicator: Indicator name (e.g. 'gdp_growth', 'inflation', 'unemployment')
        country  : ISO country code (default 'US'). Use 'WLD' for global.
        periods  : Number of recent periods to return (default 5)
    """
    if not _HAS_REQUESTS:
        return json.dumps({"error": "requests not installed."})

    key = indicator.lower().replace(" ", "_").replace("-", "_")
    wb_code = _WB_INDICATORS.get(key)

    if not wb_code:
        return json.dumps({
            "error": f"Indicator '{indicator}' not supported.",
            "available": sorted(_WB_INDICATORS.keys()),
        })

    try:
        url = (
            f"https://api.worldbank.org/v2/country/{country.upper()}"
            f"/indicator/{wb_code}?format=json&mrv={periods}"
        )
        resp = _requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        if len(data) > 1 and isinstance(data[1], list):
            points = [
                {"year": d["date"], "value": d["value"]}
                for d in data[1]
                if d.get("value") is not None
            ]
            return json.dumps(
                {
                    "indicator": indicator,
                    "country": country,
                    "source": "World Bank",
                    "data": points,
                },
                ensure_ascii=False,
            )
        return json.dumps({"error": "No data returned", "indicator": indicator})
    except Exception as e:
        return json.dumps({"error": str(e), "indicator": indicator})


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 4 — mcp_call (MCP bridge — optional)
# ═══════════════════════════════════════════════════════════════════════════

def make_mcp_tool(tool_caller: Callable[[str, Dict[str, Any]], Any]):
    """
    Factory untuk MCP tool.

    Args:
        tool_caller: callable(tool_name: str, params: dict) -> Any
                     — injected dari luar (MCP client)
    """

    @tool
    def mcp_call(tool_name: str, params_json: str = "{}") -> str:
        """
        Call a Model Context Protocol (MCP) tool.
        Use for web scraping tools or browser automation tools on the MCP server.

        Args:
            tool_name  : Name of the MCP tool (e.g. 'scrape', 'browse_page')
            params_json: JSON string of parameters (e.g. '{"url": "https://..."}')
        """
        try:
            params = json.loads(params_json) if params_json else {}
            result = tool_caller(tool_name, params)
            return json.dumps(result, default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "tool": tool_name})

    return mcp_call


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 5 — mark_done (stop signal)
# ═══════════════════════════════════════════════════════════════════════════

@tool
def mark_done(reason: str = "Sufficient data collected") -> str:
    """
    Signal that enough macroeconomic data has been collected.
    Call this when you have gathered sufficient information and are ready to stop.

    Args:
        reason: Brief explanation of why search is complete
    """
    return json.dumps({"done": True, "reason": reason})


# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 6 — Tool registry
# ═══════════════════════════════════════════════════════════════════════════

def get_default_tools(
    search_provider: str = "duckduckgo",
    search_api_key: Optional[str] = None,
    mcp_caller: Optional[Callable] = None,
    include_scraper: bool = True,
    include_econ_data: bool = True,
) -> List:
    """
    Build default tool list untuk MacroSearchGraph.

    Args:
        search_provider  : "duckduckgo" | "tavily" | "serpapi"
        search_api_key   : API key untuk Tavily / SerpAPI
        mcp_caller       : callable MCP tool caller (opsional)
        include_scraper  : include scrape_url tool
        include_econ_data: include get_economic_data tool
    """
    tools: List = [
        make_web_search_tool(
            provider=search_provider, api_key=search_api_key
        ),
        mark_done,
    ]
    if include_scraper:
        tools.append(scrape_url)
    if include_econ_data:
        tools.append(get_economic_data)
    if mcp_caller is not None:
        tools.append(make_mcp_tool(mcp_caller))
    return tools
