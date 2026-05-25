# Call 0005 — `mutation_text_probe` (text_only)

## Meta

- ts: 2026-05-25 08:52:12
- conv_id: `38b69e76`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 618
- output_tokens: 121
- duration_s: 5.1949
- text_len: 491

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
  <hypothesis>Stocks with positive short-term momentum AND increasing trading volume tend to outperform over the next 5 days.</hypothesis>
  <factors>- Momentum_Volume_Confirm_5D: RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))
  Description: 5-day momentum rank multiplied by sign of 5-day volume growth.
</factors>
  <metrics>- IC: 0.0351
- annualized_return: 0.1150
- information_ratio: 0.6310
- max_drawdown: 0.0842
</metrics>
  <feedback>Factor works but leans too heavily on short-term momentum. Consider exploring liquidity or volatility-regime dimensions for diversification.</feedback>
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
    "new_hypothesis": "Stocks with high volatility and low liquidity are more likely to exhibit mean-reverting price behavior over the next 20 days.",
    "exploration_direction": "High-low range and bid-ask spread over 20-day horizon",
    "orthogonality_reason": "Differ on market driver (mean-reversion vs. momentum) and time horizon (20-day vs. 5-day)",
    "expected_characteristics": "Signal based on price range and liquidity measures, normalized by standard deviation of returns"
}
```
