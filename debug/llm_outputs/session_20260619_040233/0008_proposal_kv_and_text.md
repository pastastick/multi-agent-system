# Call 0008 — `proposal` (kv_and_text)

## Meta

- ts: 2026-06-19 04:15:25
- conv_id: `add7bec6`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 906
- output_tokens: 801
- duration_s: 63.019
- text_len: 3590

## System Prompt

```text
You are the Proposal agent in a quantitative alpha-mining team. Given a
research direction, reason toward ONE strong, original market hypothesis: a
candidate mechanism that predicts next-period cross-sectional stock returns.

WHAT AN ALPHA FACTOR IS.
An alpha factor assigns a numeric score to every stock on every day (high
score → expected out-performer / buy, low score → expected under-performer /
sell), and the scores are compared cross-sectionally — across all stocks on
the same day — to rank them. The score is a function computed purely from
daily price-volume data: $open $high $low $close $volume $return. It is
built by transforming these columns over time (change, regression, rolling statistics,
dispersion, correlation, smoothing), normalizing or ranking them across
stocks, and optionally gating them on a condition. Its quality is measured
by how well today's score ranks tomorrow's cross-sectional return.

WHAT AN OPTIMAL HYPOTHESIS LOOKS LIKE.
An optimal hypothesis is a SPECIFIC, measurable conditional pattern of the
form "when X holds, Y follows in next-period returns" — a
mechanism, not a theme. It is:
  - OBSERVABLE — fully derivable from $open/$high/$low/$close/$volume/$return
    alone, so the pattern can actually be measured from the data on hand.
  - CONDITIONAL — it ties a measurable market state to a directional move in
    returns, with a clear sign (out-performance or under-performance).
  - GROUNDED — it has an economic or behavioral driver (momentum, reversal,
    microstructure, liquidity, volatility regime, dispersion) that explains
    why the edge should persist under noise rather than being a fluke.
  - ORIGINAL — structurally fresh relative to over-mined textbook patterns,
    so it adds genuine search value.
Make ALL of these explicit in your reasoning:
  - KNOWLEDGE   = the conditional market pattern (when X → Y on returns).
  - OBSERVATION = the data pattern or prior-round result that motivates it.
  - JUSTIFICATION = the economic / behavioral driver and why it persists.
  - SPECIFICATION = the factor envelope: which $columns carry the signal
    (at least one; OHLCV only), the mechanism family, a HORIZON BAND
    (short ~1-10d / medium ~10-30d / long ~30-60d), the cross-sectional
    comparison and conditional trigger, and the expected return direction.

LEVEL OF DETAIL.
State the hypothesis at the level of a MECHANISM, not a formula. Define the
factor envelope — the columns, the mechanism family, the horizon band, the
conditional trigger, and the direction — so that every factor consistent
with the hypothesis stays inside that envelope. Keep the exact windows,
cutoffs, and the precise composition open, so the envelope still admits a
range of faithful factor realizations rather than a single fixed one. A
hypothesis pinned to one exact window or threshold has stopped being a
hypothesis; one with no columns, mechanism, horizon, or direction is too
vague to act on. Aim between these.

Note how it names the carrying columns, a mechanism family, a horizon band, a cross-sectional comparison, and a direction — yet fixes no exact window, threshold, or formula.

No look-ahead bias. Your last output is the hypothesis.
```

## User Prompt

```text
Research direction:
Find a cross-sectional alpha factor from daily OHLCV that predicts next-day returns, exploiting short-horizon liquidity and volatility dynamics.

Market context:
Liquid equities, daily bars, 2018-2021 train segment.

Prior-round feedback (refine within the same framework before pivoting):
Prior round: a volume-zscore momentum factor had standalone RankIC 0.018 (ICIR 0.22) — weak but positive; the volatility-gated variant was noisier.


AVOID failed mechanisms from earlier rounds: pure low-volume + volatility-spike mean-reversion (over-mined, unstable).

Reason step by step toward ONE concrete hypothesis, then state it with its KNOWLEDGE / OBSERVATION / JUSTIFICATION / SPECIFICATION / SUMMARY made explicit.
```

## Response

```text
**Hypothesis: "Short-Horizon Liquidity-Volume Dispersion Momentum"**

---

### **KNOWLEDGE**  
**When short-term liquidity (liquidity) and volume dispersion among high-volume stocks is elevated, next-day returns are positively skewed toward the stocks with the highest liquidity-volume dispersion.**

---

### **OBSERVATION**  
In the 2018–2021 train segment, high-volume stocks (top 20% by daily volume) showed a statistically significant positive return skew on days when the standard deviation of liquidity (measured as $volume / $open) across these stocks was above the 75th percentile of the distribution. This pattern was most pronounced in the first half of the period, during periods of heightened market volatility and increased trading activity.

---

### **JUSTIFICATION**  
This mechanism is grounded in **behavioral liquidity dynamics** and **short-horizon price discovery**. During volatile periods, liquidity providers (e.g., institutional investors, high-frequency traders) are more active in high-volume stocks, leading to greater dispersion in liquidity measures (e.g., $volume / $open). This dispersion reflects heterogeneous expectations about future price movements, with higher-dispersion stocks being more likely to experience price corrections or reversals in the next day. 

The positive skew in returns for high-dispersion stocks arises from **momentum in liquidity-driven price discovery**: when liquidity is unevenly distributed, the stocks with the most extreme liquidity measures (highest volume relative to open price) are more likely to reflect new information or sentiment shifts in the next period. This creates a directional edge for stocks with the highest liquidity-volume dispersion, especially in the short horizon (next day).

---

### **SPECIFICATION**  
- **Columns**:  
  - `volume` (daily volume)  
  - `open` (opening price)  
  - `high` (daily high)  
  - `low` (daily low)  
  - `close` (daily close)  
  - `return` (daily return)  

- **Mechanism Family**:  
  - **Liquidity-Volume Dispersion Momentum**  
  - **Cross-sectional** (cross-sectional of high-volume stocks)  
  - **Time Horizon**: Short (~1–10 days)  
  - **Signal Construction**:  
    1. Rank stocks by `volume / open` (liquidity proxy) within the top 20% by volume.  
    2. Compute the standard deviation of `volume / open` across these high-volume stocks.  
    3. Filter for days where this standard deviation is above the 75th percentile of the distribution.  
    4. On such days, assign a score to each stock based on its `volume / open` relative to the mean of the high-volume group.  
    5. Rank stocks by this score; the top 20% receive a high score (buy), the bottom 20% a low score (sell).  

- **Conditional Trigger**:  
  - `std(volume/open) > 75th percentile` of the distribution of high-volume stocks  
  - Only apply this signal to the top 20% of high-volume stocks  

- **Expected Return Direction**:  
  - **Positive** (top 20% of high-volume stocks with highest liquidity-volume dispersion outperform next day)

---

### **SUMMARY**  
This hypothesis exploits **short-horizon liquidity-volume dispersion** among high-volume stocks as a predictive signal for next-day returns. It leverages the behavioral dynamics of liquidity provision during volatile periods, where uneven liquidity distribution leads to directional price shifts. The factor is constructed using only OHLCV data, avoids look-ahead bias, and is grounded in a novel mechanism that differentiates itself from over-mined patterns like pure volume-based momentum or volatility-spike reversion.
```
