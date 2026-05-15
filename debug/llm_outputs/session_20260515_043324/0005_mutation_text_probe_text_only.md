# Call 0005 — `mutation_text_probe` (text_only)

## Meta

- ts: 2026-05-15 04:33:55
- conv_id: `b129ba8d`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 516
- output_tokens: 117
- duration_s: 6.4389
- text_len: 577

## System Prompt

```text
You are the **Mutation agent** in the Evolution layer of an alpha factor mining pipeline.
• Receives: ONE parent trajectory (hypothesis, factor expressions, backtest metrics, feedback) as text. Evolution is a fresh start — no latent KV chain from the parent pipeline.
• Sends: a direction string to a new Propose agent. Propose will turn it into a fresh hypothesis.

Task: diagnose the weakest step in the parent trajectory, then output an orthogonal direction that avoids that failure mode and explores factor space the parent did not reach.

"Orthogonal" means the new direction:
1. Explores a different market hypothesis (different driver, not just different parameter).
2. Uses different data dimensions or feature types.
3. Is grounded in different investment logic or market perspective.
4. Should produce factors with low expected correlation to the parent's.

The direction must be specific enough that Propose can generate a concrete, testable hypothesis from it.

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

Generate an orthogonal new strategy direction that diagnoses why the parent failed (or partially succeeded) and points at unexplored space.

Output ONLY this JSON (no markdown fences, no commentary):
```json
{
    "new_hypothesis": "ONE LINE. The new market hypothesis.",
    "exploration_direction": "ONE LINE. Data dimensions and feature types to explore.",
    "orthogonality_reason": "ONE LINE. Why this is orthogonal to the parent.",
    "expected_characteristics": "ONE LINE. Expected factor properties (signal type, time horizon, normalization)."
}
```

```

## Response

```text
{
    "new_hypothesis": "Stocks with high institutional liquidity and low volatility during earnings announcements outperform in the following week.",
    "exploration_direction": "Institutional liquidity metrics and volatility measures around earnings events.",
    "orthogonality_reason": "Shifts from short-term momentum to liquidity and volatility around earnings events, using different data dimensions.",
    "expected_characteristics": "Signal type: liquidity-volatility contrast; time horizon: earnings event window; normalization: z-scored liquidity and volatility."
}
```
