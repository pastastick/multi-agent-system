# Call 0028 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-21 08:49:11
- conv_id: `582599ae`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 1322
- output_tokens: 461
- duration_s: 17.7179
- text_len: 2346

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
Factor 1 (MeanReversion_5D) has a standalone RankIC of 0.041 and ICIR of 0.22. This places it as the strongest factor in Block A, but its ICIR is low, indicating instability. The combined model (Block B) shows a high RankIC of 0.060 but also a catastrophic drawdown of 0.312 and collapses in trending regimes. This suggests that while the standalone signal is weak, it may be amplifying noise in the combined model.

---

**FACTOR DIAGNOSIS**  
- **MeanReversion_5D**: Weak due to poor stability.  
  - **Variable choice**: Uses a single raw variable (TS_MEAN($return, 5)) without transformation or ratio. A ratio or difference might better capture mean reversion.  
  - **Normalization**: Uses raw price units, which are scale-sensitive. A cross-sectional rank or z-score would reduce noise.  
  - **Window length**: 5-day window is too short for a medium-horizon mean reversion mechanism, capturing noise rather than true reversal.  
  - **Composition depth**: Single-step function (mean of returns) instead of a two-step structure (e.g., time-series mean followed by cross-sectional standardization).  
  - **Function family**: Uses a mean, which is not the right measure for mean reversion — a regression slope or trend capture would be more appropriate.  
  - **Conditional structure**: No regime gate is present; the signal may not hold in trending markets, leading to collapse in the combined model.

---

**WEAKNESS BREAKDOWN**  
1. Every factor uses a single raw variable without normalization or transformation.  
2. All factors use a short window length (5 days), which is insufficient for capturing meaningful mean reversion.  
3. Composition depth is shallow: all factors are single-step functions, missing the two-step structure needed for robust signals.  
4. Function families are not aligned with the hypothesis (e.g., using mean instead of slope or trend capture).  
5. No conditional structure is applied, leading to poor performance in trending regimes.

---

**HYPOTHESIS VERDICT**  
Partially supports. The hypothesis of mean reversion is sound, but the construction of the factor is weak. The expression needs refinement — particularly in terms of normalization, window length, and composition depth. The function family and conditional structure are not aligned with the expected behavior in trending markets.

---

**REPLACE BEST**  
Yes.

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
Each named weakness is fixable within the current mechanism. The core issue is not that the mean reversion hypothesis is invalid, but that the implementation is too simplistic and lacks robustness in handling noise, window length, and regime shifts.

**Step 2 — Name what the parent mechanism still gets right.**  
The parent mechanism correctly identifies mean reversion as a viable strategy, and the standalone factor shows some signal in the form of a positive RankIC, even if it’s unstable.

**Step 3 — For the most critical fixable weakness, name the ONE concrete structural change.**  
The most critical weakness is the use of a single raw variable without normalization or transformation. To address this, we should replace the raw variable with a **cross-sectional z-score of the 5-day return mean**, which normalizes the signal across assets and reduces noise. This change aligns with the feedback that raw price units are scale-sensitive and that a ratio or difference would better capture mean reversion.

**Step 4 — Inspired by what the failures revealed about the data, name ONE adjacent hypothesis direction.**  
The failures suggest that the current approach is too focused on cross-sectional mean reversion and lacks consideration for time-series dynamics. An adjacent hypothesis could be **trend capture with momentum**, where the driver is a persistent trend in asset prices, and variables like returns over longer windows (e.g., 20 days) and moving averages carry the pattern. The expected return direction would be positive in trending regimes, which the parent model failed to handle due to its reliance on mean reversion.

---

WHAT WORKS: The parent mechanism correctly identifies mean reversion as a viable strategy, showing a positive RankIC in the standalone factor.  
DIAGNOSIS: The use of a single raw variable without normalization or transformation leads to poor stability and noise amplification.  
DIRECTION: Replace the raw variable with a cross-sectional z-score of the 5-day return mean to normalize the signal and reduce noise.  
NEW HYPOTHESIS: A trend-capture hypothesis where persistent price trends drive returns, using longer-window returns and moving averages. The parent’s failure in trending regimes suggests this direction is motivated by the data.
```
