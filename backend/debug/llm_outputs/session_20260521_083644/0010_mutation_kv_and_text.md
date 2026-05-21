# Call 0010 — `mutation` (kv_and_text)

## Meta

- ts: 2026-05-21 08:51:03
- conv_id: `3d2497d2`
- step: 0
- temperature: 0.8
- has_past_kv: False
- input_tokens: 775
- output_tokens: 113
- duration_s: 5.5435
- text_len: 443

## Variables (dari YAML placeholder)

- **parent_hypothesis** (850 chars): Hypothesis: Low-volatility stocks with shrinking intraday ranges exhibit stronger mean-reversion after large 1-day price moves. ⏎                 Concise Observation: Stocks with tight price ranges and …
- **parent_factors** (399 chars): - INTRADAY_RANGE: (MAX($high, $low) - MIN($high, $low)) / $close ⏎   Description: Measures the intraday price range relative to the closing price. ⏎ - VOLATILITY_RANK: RANK((MAX($high, $low) - MIN($high, …
- **parent_metrics** (3 chars): N/A
- **parent_feedback** (3 chars): N/A

## System Prompt

```text
Role: Mutation Agent

Mission
Diagnose the weakness in ONE parent trajectory and generate an orthogonal exploration direction for a fresh Propose agent to follow.

Evolution Context
This is a fresh start — there is no latent KV chain from the parent pipeline. You receive the parent trajectory as text only.

Orthogonality Rules (the new direction must differ on at least 2 of these 4 dimensions):
1. Market driver: different behavioral mechanism or structural inefficiency (e.g., parent used momentum → you target mean-reversion or liquidity)
2. Data dimension: different OHLCV feature combination (e.g., parent used $volume/$close → you use $high/$low/$return)
3. Time horizon: if parent used short (3-10d) signals → suggest medium (20-60d), or vice versa
4. Market condition: if parent worked in trending markets → target mean-reverting / range-bound conditions

Direction Quality Rules:
- Specific enough that Propose can form a concrete, testable hypothesis from it (name a mechanism, not just a theme)
- Do NOT suggest factor parameter tweaks — suggest a different market phenomenon entirely
- The direction should naturally produce factors with low expected correlation to the parent's

```

## User Prompt

```text
<parent_trajectory>
  <hypothesis><<<parent_hypothesis>>>
Hypothesis: Low-volatility stocks with shrinking intraday ranges exhibit stronger mean-reversion after large 1-day price moves.
                Concise Observation: Stocks with tight price ranges and low volatility show stronger price corrections after sharp moves.
                Concise Justification: Tight ranges suggest limited uncertainty, and large moves may trigger retail overreaction leading to rapid mean-reversion.
                Concise Knowledge: When intraday range ($high - $low) shrinks below 1% of $close for 3+ days, mean-reversion is amplified following large 1-day returns.
                concise Specification: Cross-sectionally rank stocks by ($high - $low)/$close over 3-day window, filter for stocks with ($high - $low)/$close < 1% and absolute $return > 2% on day 0, then score based on next-day $return.
                
<<</parent_hypothesis>>></hypothesis>
  <factors><<<parent_factors>>>
- INTRADAY_RANGE: (MAX($high, $low) - MIN($high, $low)) / $close
  Description: Measures the intraday price range relative to the closing price.
- VOLATILITY_RANK: RANK((MAX($high, $low) - MIN($high, $low)) / $close, 3)
  Description: Ranks stocks by volatility based on intraday range.
- LARGE_RETURN_FILTER: FILTER((ABS($return) > 0.02), 1)
  Description: Filters stocks with large 1-day returns.

<<</parent_factors>>></factors>
  <metrics><<<parent_metrics>>>
N/A
<<</parent_metrics>>></metrics>
  <feedback><<<parent_feedback>>>
N/A
<<</parent_feedback>>></feedback>
</parent_trajectory>

Step 1: Diagnose why the parent strategy succeeded or failed (cite specific metrics or feedback).
Step 2: Identify which 2+ orthogonality dimensions the new direction will differ on.
Step 3: Generate the orthogonal direction.

Output ONLY this JSON (no markdown fences, no commentary):
{
    "new_hypothesis": "ONE LINE. The new market hypothesis — name a specific mechanism, not a generic theme.",
    "exploration_direction": "ONE LINE. Which OHLCV dimensions and time horizon to explore.",
    "orthogonality_reason": "ONE LINE. Exactly how this differs from the parent on 2+ of the 4 dimensions.",
    "expected_characteristics": "ONE LINE. Expected signal type, time horizon, normalization approach."
}

```

## Response

```text
{
    "new_hypothesis": "Stocks with high liquidity and low trade volume exhibit weaker mean-reversion after large price moves.",
    "exploration_direction": "Examine $volume/$open and $return over 20-day horizon",
    "orthogonality_reason": "Differ on market driver (liquidity vs volatility) and time horizon (20-day vs 1-day)",
    "expected_characteristics": "Liquidity-based signal with 20-day horizon and volume-to-open normalization"
}
```
