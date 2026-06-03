# Call 0008 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-03 07:25:12
- conv_id: `09da736c`
- step: 0
- temperature: 0.8
- has_past_kv: True
- input_tokens: 829
- output_tokens: 153
- duration_s: 6.063
- text_len: 635

## Variables (dari YAML placeholder)

- **parent_hypothesis** (78 chars): Low-volume stocks with sudden price volatility are more likely to mean-revert.
- **parent_factors** (372 chars): - Low_volume_stocks_with_sudden_pr_0: TS_ZSCORE($high, 7) - TS_ZSCORE($low, 7) ⏎   Description: Low-volume stocks with sudden price volatility are more likely to mean-revert. ⏎ - Low_volume_stocks_with_su…
- **parent_metrics** (144 chars): - IC: 0.0108 ⏎ - ICIR: 0.0790 ⏎ - RankIC: 0.0306 ⏎ - RankICIR: 0.2297 ⏎ - annualized_return: 0.0604 ⏎ - information_ratio: 0.4759 ⏎ - max_drawdown: -0.0982 ⏎ 
- **parent_feedback** (602 chars): {'Observations': 'The factor shows an IC of 0.0108 and an information ratio of 0.0789, which are below the SOTA of 0.0632. The annualized return of 0.0604 is also below the SOTA, indicating that the h…

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
Low-volume stocks with sudden price volatility are more likely to mean-revert.
<<</parent_hypothesis>>></hypothesis>
  <factors><<<parent_factors>>>
- Low_volume_stocks_with_sudden_pr_0: TS_ZSCORE($high, 7) - TS_ZSCORE($low, 7)
  Description: Low-volume stocks with sudden price volatility are more likely to mean-revert.
- Low_volume_stocks_with_sudden_pr_1: TS_ZSCORE($volume, 7) * (TS_ZSCORE($high, 7) - TS_ZSCORE($low, 7))
  Description: Low-volume stocks with sudden price volatility are more likely to mean-revert.

<<</parent_factors>>></factors>
  <metrics><<<parent_metrics>>>
- IC: 0.0108
- ICIR: 0.0790
- RankIC: 0.0306
- RankICIR: 0.2297
- annualized_return: 0.0604
- information_ratio: 0.4759
- max_drawdown: -0.0982

<<</parent_metrics>>></metrics>
  <feedback><<<parent_feedback>>>
{'Observations': 'The factor shows an IC of 0.0108 and an information ratio of 0.0789, which are below the SOTA of 0.0632. The annualized return of 0.0604 is also below the SOTA, indicating that the hypothesis does not outperform the best existing factor.', 'Feedback for Hypothesis': 'The result partially supports the hypothesis, but the signal is weak and not robust enough to surpass the SOTA. The factors are complex and may overfit, leading to a low information ratio and poor out-of-sample performance. The framework should be simplified before further refinement.', 'Replace Best Result': 'no'}
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
    "new_hypothesis": "High-liquidity stocks with recent price corrections are more likely to exhibit momentum in the next period.",
    "exploration_direction": "Combine $volume, $return, and $high/$low over 20-day windows to capture liquidity and price correction signals.",
    "orthogonality_reason": "Differs from the parent on market driver (momentum vs. mean-reversion), data dimension (liquidity vs. volume), and time horizon (20-day vs. 7-day).",
    "expected_characteristics": "A liquidity-weighted price correction signal over 20 days, normalized using TS_ZSCORE, expected to show positive momentum in the next period."
}
```
