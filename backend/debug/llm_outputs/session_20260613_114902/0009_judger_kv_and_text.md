# Call 0009 — `judger` (kv_and_text)

## Meta

- ts: 2026-06-13 11:52:17
- conv_id: `37621b1d`
- step: 0
- temperature: 0.3
- has_past_kv: True
- input_tokens: 1233
- output_tokens: 435
- duration_s: 14.9487
- text_len: 1550

## System Prompt

```text
You are the Judger — FORMATTING stage (pass 2 of 2). Your own pass-1 reasoning
(one hypothesis and its 1-3 candidate factor expressions) is given below. Do
NOT re-reason, re-derive, or "improve" the ideas, and do NOT invent new
expressions: faithfully EXTRACT what is already there into the exact labeled
block. Copy each expression VERBATIM (character-for-character) from the
reasoning — never edit operators, arguments, or windows.

Output EXACTLY this labeled block (plain text, no JSON, no markdown fences),
nothing before or after. Put the HYPOTHESIS line and the EXPRESSION lines LAST:

KNOWLEDGE: <one line — the conditional market pattern>
OBSERVATION: <one line — data/result that motivates it>
JUSTIFICATION: <one line — economic / behavioral reason it should work>
SPECIFICATION: <one line — variables, time horizon, expected effect>
FACTOR 1 NAME: <short_name_no_spaces> | DESC: <one line — what it measures>
FACTOR 2 NAME: <short_name_no_spaces> | DESC: <one line>   (only if a 2nd factor exists)
HYPOTHESIS: <ONE sentence stating the mechanism>
EXPRESSION 1: <single DSL expression, copied verbatim>
EXPRESSION 2: <single DSL expression, copied verbatim>   (only if a distinct 2nd factor exists)
```

## User Prompt

```text
Research direction: price-volume momentum in high-volatility sectors during bear markets

Your pass-1 reasoning to format (extract faithfully; copy expression(s) verbatim):
---
In high-volatility sectors during bear markets, stocks that show **persistence in price momentum** (i.e., consistent upward or downward movement) and **rising volume** are most likely to outperform in the next 5 days. The key is not just raw momentum or volume, but a **combination of persistent price direction and increasing liquidity**, which suggests **continuation of the trend** rather than a reversal.

### Observation:
The parent factor was too correlated with raw volume and had low information coefficient (IC). This suggests that the signal was too noisy or not sufficiently distinct from volume alone. A better approach is to **isolate the momentum component** from volume, and **use a nonlinear transformation** to emphasize the **tails of the distribution** — where extreme momentum and volume combinations are most predictive.

### Knowledge:
When a stock shows **sustained price momentum** (e.g., positive delta over 10 days) and **increasing volume**, it is more likely to continue trending in the same direction in the short term, especially in high-volatility sectors during bear markets. This is because **volume can signal strength**, and **momentum can signal continuation** — together, they suggest a **trend-continuation opportunity**.

### Justification:
- **Momentum** (delta of close over 10 days) captures the direction of price movement.
- **Volume** (percent change over 10 days) captures the strength of that movement.
- **Nonlinear shaping** (e.g., `LOG`) helps emphasize the **extreme cases** where both momentum and volume are strong — these are the most informative for prediction.
- **Technical indicators** like **RSI** can help filter out noise and highlight **overbought/oversold** conditions, which are more likely to reverse in bear markets.

---

### Candidate Factor Expressions:

#### 1. `MOMENTUM_VOL_TAIL`
**Measures:** The product of price momentum and volume, but with a logarithmic compression to emphasize extreme values.  
**Expression:**  
`LOG(TS_ZSCORE(DELTA($close, 10), 10) * TS_PCTCHANGE($volume, 10))`  
**Why:** Logarithmic compression helps emphasize the **tails** of the distribution, where the most predictive combinations of momentum and volume occur. It also reduces the impact of very large volume spikes, which can be noisy.

#### 2. `VOL_MOMENTUM_RSI`
**Measures:** A combination of momentum, volume, and RSI to filter out overbought/oversold conditions.  
**Expression:**  
`REGRESI(DELTA($close, 10), DELTA($close, 10), 10) * TS_PCTCHANGE($volume, 10) * (RSI(DELTA($close, 10), 14) < 30)`  
**Why:** RSI adds a **momentum filter** — if RSI is below 30, it’s overbought, and the trend is likely to reverse. This expression focuses on **strong momentum with low RSI**, which is more likely to continue in a bear market.

#### 3. `VOL_MOMENTUM_BB`
**Measures:** A combination of momentum and volume, weighted by Bollinger Band width to emphasize volatility.  
**Expression:**  
`TS_PCTCHANGE($volume, 10) * TS_ZSCORE(DELTA($close, 10), 10) / BB_LOWER(DELTA($close, 10), 20)`  
**Why:** Bollinger Bands capture **volatility** — in high-volatility sectors, the **band width** is wider, so dividing by it normalizes the signal. This helps isolate the **true momentum** from noise.

---

### Final Hypothesis:
In high-volatility sectors during bear markets, stocks with **persistent price momentum** and **rising volume**, particularly when combined with **low RSI or wide Bollinger Bands**, are most likely to outperform in the next 5 days.
---
Emit ONLY the labeled block now.
```

## Response

```text
KNOWLEDGE: Stocks with persistent price momentum and rising volume are more likely to continue trending in high-volatility sectors during bear markets.  
OBSERVATION: The parent factor was too correlated with raw volume and had low IC, suggesting a need to isolate momentum from volume and emphasize extreme cases.  
JUSTIFICATION: Momentum signals trend continuation, volume signals strength, and nonlinear shaping (like LOG) emphasizes the tails where the most predictive combinations occur.  
SPECIFICATION: $close, $volume, DELTA, TS_PCTCHANGE, LOG, TS_ZSCORE; 10-day horizon; positive return direction.  
FACTOR 1 NAME: MOMENTUM_VOL_TAIL | DESC: LOG of product of price momentum and volume to emphasize extreme values.  
FACTOR 2 NAME: VOL_MOMENTUM_RSI | DESC: Product of momentum, volume, and RSI filter to highlight overbought/oversold conditions.  
FACTOR 3 NAME: VOL_MOMENTUM_BB | DESC: Product of volume and momentum divided by Bollinger Band lower bound to normalize for volatility.  
HYPOTHESIS: In high-volatility sectors during bear markets, stocks with persistent price momentum and rising volume, particularly when combined with low RSI or wide Bollinger Bands, are most likely to outperform in the next 5 days.  
EXPRESSION 1: LOG(TS_ZSCORE(DELTA($close, 10), 10) * TS_PCTCHANGE($volume, 10))  
EXPRESSION 2: REGRESI(DELTA($close, 10), DELTA($close, 10), 10) * TS_PCTCHANGE($volume, 10) * (RSI(DELTA($close, 10), 14) < 30)  
EXPRESSION 3: TS_PCTCHANGE($volume, 10) * TS_ZSCORE(DELTA($close, 10), 10) / BB_LOWER(DELTA($close, 10), 20)
```
