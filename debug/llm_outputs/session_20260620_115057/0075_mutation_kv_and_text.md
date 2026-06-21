# Call 0075 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-20 13:33:31
- conv_id: `131c0ef8`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 1156
- output_tokens: 473
- duration_s: 22.5973
- text_len: 2380

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
Factor 1 (Momentum_Raw_3D) has the only standalone RankIC of 0.005, which is near baseline, and an ICIR of 0.08, indicating very low stability. This factor is the weakest in Block A. No other factors are present in this round, so it is the sole factor under evaluation.

---

**FACTOR DIAGNOSIS**  
- **Factor 1 (Momentum_Raw_3D):** Weak. The signal is based on a single raw variable (TS_MEAN of returns over 3 days), without any normalization or transformation to reduce noise. The window length is too short for a meaningful momentum signal, and the function family (simple moving average) does not capture trend direction effectively. There is no indication of conditional structure or composition depth that would improve its predictive power.

---

**WEAKNESS BREAKDOWN**  
1. Every factor uses a single raw variable with no normalization.  
2. Window length is too short (3 days) for a momentum signal to show meaningful persistence.  
3. No cross-sectional ranking or z-score normalization is applied, leading to scale-sensitive noise.  
4. The function family (moving average) does not reflect the hypothesis of trend direction.  
5. No conditional gates or regime-based structures are included, suggesting the signal may not hold under all market states.

---

**HYPOTHESIS VERDICT**  
Partially refutes. The mechanism (momentum) is sound in theory, but the construction is weak. The factor fails to properly capture trend direction and lacks the necessary structural refinements (normalization, window length, composition depth) to make the hypothesis visible in the data.

---

**REPLACE BEST**  
No.

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
Each of the named weaknesses is fixable within the current mechanism. The signal is based on a raw variable, but normalization, window length, function family, and conditional structure can all be adjusted to improve it. The hypothesis of momentum is still valid, but its implementation is too simplistic.

**Step 2 — Name what the parent mechanism still gets right.**  
The parent mechanism still captures some level of momentum in the data, as evidenced by the near-baseline RankIC of 0.005 and the presence of a signal, even if it’s unstable and weak.

**Step 3 — For the most critical fixable weakness, name the ONE concrete structural change.**  
The most critical weakness is the use of a short window (3 days) that doesn’t allow for meaningful momentum persistence. To address this, the window should be extended to at least 10 days. A longer window allows the momentum signal to accumulate more historical data, improving its stability and ability to reflect trend direction.

**Step 4 — Inspired by what the failures revealed about the data, name ONE adjacent hypothesis direction.**  
The failures suggest that the market may not be driven by simple momentum over short periods, but rather by longer-term trends or regime shifts. An adjacent hypothesis is that the market may be influenced by multi-period trend persistence, where variables like TS_MEAN over 10 days or volatility-based indicators could better capture the underlying trend direction. This hypothesis extends the momentum idea by focusing on longer-term signals that are less noise-sensitive.

---

WHAT WORKS: The parent mechanism still captures some level of momentum in the data, even if it's unstable and weak.  
DIAGNOSIS: The use of a short window (3 days) prevents the momentum signal from accumulating enough historical data to show meaningful persistence.  
DIRECTION: Extend the window to 10 days to allow the momentum signal to accumulate more historical data, improving its stability and ability to reflect trend direction.  
NEW HYPOTHESIS: The market may be influenced by multi-period trend persistence, where variables like TS_MEAN over 10 days or volatility-based indicators could better capture the underlying trend direction. This hypothesis extends the momentum idea by focusing on longer-term signals that are less noise-sensitive.
```
