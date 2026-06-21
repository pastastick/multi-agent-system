# Call 0100 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-20 14:11:11
- conv_id: `a9099b0e`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 1343
- output_tokens: 438
- duration_s: 17.993
- text_len: 2415

## System Prompt

```text
You are the Mutation Director in a quantitative alpha-mining team. ONE parent
result — its hypothesis, factors, metrics, and Feedback diagnosis — reaches you
either through shared latent memory or as text in the task below. YOUR JOB is to
set the DIRECTION that the Proposal agent builds on in the next round. That
direction has two parts: a concrete refinement of the existing hypothesis that
fixes the named structural weaknesses, and a suggested new hypothesis that the
failures point toward.

PIPELINE CONTEXT. Proposal inherits your reasoning. Write concretely enough that
Proposal knows exactly what to pursue: what the parent mechanism is, what was
structurally wrong with its construction, what kind of fix to apply, and — if the
parent mechanism shows limited headroom — which adjacent hypothesis the data
suggests.

This is EXPLOITATION with an escape hatch. The primary task is to push the parent
framework further by fixing what the Feedback weakness breakdown named. The
secondary task is to read what the failures reveal about the data and name an
adjacent hypothesis direction that the results motivate, not invent from scratch.

WHAT TO REASON ABOUT:
  - WHAT WORKS = the part of the parent hypothesis and factors that Block A
    confirmed, even partially. Name it explicitly so Proposal preserves it.
  - WHAT FAILS = the structural weaknesses from the Feedback weakness breakdown.
    For each named weakness, state what kind of structural change would address
    it: a different variable or transform, a longer or shorter window, adding
    cross-sectional normalization, switching function families, or adding a
    conditional gate.
  - REFINEMENT DIRECTION = ONE focused structural change that keeps the mechanism
    intact and addresses the most critical weakness. Name the type of change
    concretely, not just the goal.
  - NEW HYPOTHESIS SUGGESTION = if the failures suggest the mechanism has limited
    headroom, name ONE adjacent hypothesis that the results motivate. State what
    the parent got partially right and how the new direction extends or shifts it.
    This is for Proposal to hold in reserve; the refinement direction takes
    priority unless the mechanism verdict was refutes.

Always state the diagnosis and direction in your own words. Write in plain prose.
```

## User Prompt

```text
Parent result to develop:
**OBSERVATION**  
Factor 1 is the strongest with a RankIC of 0.031 and ICIR of 0.41, indicating moderate predictive power and reasonable stability. Factor 2 is the weakest with a RankIC of 0.009 and ICIR of 0.10, showing very low predictive power and high instability. Block B shows a combined model with a RankIC near baseline (0.052), with a MaxDrawdown of 0.170, suggesting limited added value from the factors.

---

**FACTOR DIAGNOSIS**  
- **Factor 1**: Strong, but with structural weaknesses. The signal is based on a combination of a time-series z-score of volume and a cross-sectional rank of a price range ratio. However, the factor uses raw price units in the price range calculation without normalization, and the window length of 5 days may be too short for the intended mechanism. The composition is two-step but not clearly aligned with the hypothesis of capturing market sentiment or liquidity shifts.
  
- **Factor 2**: Weak, and likely noise. The factor uses a regression beta of returns against volume over a 20-day window, which is a single-variable function. The RankIC is very low, and the ICIR is extremely low, indicating instability and lack of predictive power. The function family is misaligned with the hypothesis, as beta is not a direct measure of trend direction but rather a statistical relationship between variables.

---

**WEAKNESS BREAKDOWN**  
1. All factors use raw price or volume units without normalization, leading to scale-sensitive signals that fail to capture meaningful patterns.  
2. Factor 1 uses a 5-day window, which is too short for a medium-horizon mechanism, resulting in noisy and unstable signals.  
3. Factor 2 relies on a single raw variable (volume) without proper cross-sectional transformation, making it a weak and unstable signal.  
4. Both factors lack conditional gates, meaning they do not account for market states or regimes where the signal might be invalid or misleading.  
5. The function families used (z-score, beta) are not clearly aligned with the hypothesized mechanisms, leading to misinterpretation of the signal’s meaning.

---

**HYPOTHESIS VERDICT**  
Partially supports. The mechanisms may be sound in theory (e.g., liquidity, sentiment, trend), but the construction of the factors is weak. The expressions need refinement in terms of variable choice, normalization, window length, and conditional structure. The data does not strongly confirm the hypotheses, but there is potential for improvement.

---

**REPLACE BEST**  
No

Reason in four steps:
  Step 1 — Read the Feedback weakness breakdown. For each named weakness, decide
           whether it is fixable within the current mechanism (a structural
           adjustment suffices) or whether it suggests the mechanism itself is
           limited.
  Step 2 — Name what the parent mechanism still gets right: the part of the
           signal that showed any positive RankIC, even if small or unstable.
           This is what Proposal must preserve.
  Step 3 — For the most critical fixable weakness, name the ONE concrete
           structural change: a different variable or transform of it, a window
           adjustment, a normalization layer, a function-family switch, or a
           conditional gate. Say why that specific change addresses that weakness.
  Step 4 — Inspired by what the failures revealed about the data, name ONE
           adjacent hypothesis direction. Describe it as a market pattern with a
           driver, which variables carry it, and the expected return direction.
           Explain why the parent's failures point toward it.
Then close with this block, filling each placeholder with the actual value:
WHAT WORKS: <the part of the parent that showed signal, in one clause>
DIAGNOSIS: <the most critical structural weakness, in plain words>
DIRECTION: <the concrete structural change to try and why it targets the weakness>
NEW HYPOTHESIS: <the adjacent hypothesis the failures motivate, in two sentences>
```

## Response

```text
**Step 1 — Read the Feedback weakness breakdown.**  
The feedback identifies several structural issues: raw units without normalization, short window lengths, single-variable functions, lack of conditional gates, and misaligned function families. These are all fixable within the current mechanism through structural adjustments—no fundamental change to the hypothesis is needed.

**Step 2 — Name what the parent mechanism still gets right.**  
The parent mechanism captures some signal in Factor 1, which has a moderate RankIC and ICIR, suggesting that its underlying hypothesis about liquidity or sentiment may have partial validity.

**Step 3 — For the most critical fixable weakness, name the ONE concrete structural change.**  
The most critical weakness is the use of raw price units without normalization, leading to scale-sensitive signals. To address this, we should add a cross-sectional normalization step to the price range ratio in Factor 1. This would stabilize the signal by reducing the impact of extreme price movements across different assets, making the factor more robust and aligned with the hypothesis of capturing market sentiment or liquidity shifts.

**Step 4 — Inspired by what the failures revealed about the data, name ONE adjacent hypothesis direction.**  
The failures suggest that the current factors are too focused on single-variable or short-term signals. An adjacent hypothesis is that the market may exhibit patterns driven by multi-asset correlation shifts, where the relative movement between assets (rather than individual price changes) carries predictive power. Variables like cross-asset beta spreads or correlation rank differences could carry this signal, and the expected return direction would be positive during periods of heightened market coherence.

---

WHAT WORKS: Factor 1 captures some signal related to liquidity or sentiment despite its structural weaknesses.  
DIAGNOSIS: The factor uses raw price units without normalization, leading to unstable and scale-sensitive signals.  
DIRECTION: Add cross-sectional normalization to the price range ratio in Factor 1 to reduce scale sensitivity and improve stability.  
NEW HYPOTHESIS: The market may exhibit patterns driven by multi-asset correlation shifts, where cross-asset beta spreads or correlation rank differences carry predictive power, with positive returns during periods of heightened market coherence.
```
