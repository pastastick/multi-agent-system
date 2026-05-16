# Call 0026 — `mutation` (kv_and_text)

## Meta

- ts: 2026-05-16 04:05:13
- conv_id: `5592282b`
- step: 0
- temperature: 0.8
- has_past_kv: False
- input_tokens: 685
- output_tokens: 116
- duration_s: 5.0065
- text_len: 522

## Variables (dari YAML placeholder)

- **parent_hypothesis** (649 chars): Hypothesis: Low-volume days followed by high-volume days exhibit enhanced mean-reversion in price. ⏎                 Concise Observation:  ⏎                 Concise Justification: Behavioral investors may…
- **parent_factors** (548 chars): - VolumeShift_5D: TS_RANK((TS_SUM($volume, 5) - TS_MEAN($volume, 20)) / (TS_MEAN($volume, 20) + 1e-8), 5) ⏎   Description: 5-day volume shift normalized by 20-day average to capture volume trend changes…
- **parent_metrics** (3 chars): N/A
- **parent_feedback** (3 chars): N/A

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
  <hypothesis><<<parent_hypothesis>>>
Hypothesis: Low-volume days followed by high-volume days exhibit enhanced mean-reversion in price.
                Concise Observation: 
                Concise Justification: Behavioral investors may overreact on low-volume days, creating mispricings that are corrected on high-volume days due to increased liquidity and information dissemination.
                Concise Knowledge: Days with low trading volume followed by high volume show stronger price reversion to previous levels.
                concise Specification: Volume change over 5-day window, price return over 10-day horizon, applicable to equities with daily data.
                
<<</parent_hypothesis>>></hypothesis>
  <factors><<<parent_factors>>>
- VolumeShift_5D: TS_RANK((TS_SUM($volume, 5) - TS_MEAN($volume, 20)) / (TS_MEAN($volume, 20) + 1e-8), 5)
  Description: 5-day volume shift normalized by 20-day average to capture volume trend changes
- PriceReversion_10D: ZSCORE(($close - TS_MEAN($close, 10)) / (TS_STD($close, 10) + 1e-8), 10)
  Description: 10-day price reversion strength measured by z-score relative to mean
- VolumeRebound_5D: TS_RANK((TS_SUM($volume, 5) - TS_MIN($volume, 5)) / (TS_MEAN($volume, 5) + 1e-8), 5)
  Description: Volume rebound strength after low-volume period

<<</parent_factors>>></factors>
  <metrics><<<parent_metrics>>>
N/A
<<</parent_metrics>>></metrics>
  <feedback><<<parent_feedback>>>
N/A
<<</parent_feedback>>></feedback>
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
    "new_hypothesis": "High-frequency trading volume spikes in pre-market hours correlate with intraday price volatility in equities.",
    "exploration_direction": "Pre-market volume spikes, intraday volatility, high-frequency data, order book depth changes.",
    "orthogonality_reason": "Focuses on pre-market activity and high-frequency data, distinct from daily volume and price reversion.",
    "expected_characteristics": "Intraday volatility signal, short time horizon, normalized by order book depth changes."
}
```
