# Call 0007 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-21 09:51:23
- conv_id: `3887751c`
- step: 0
- temperature: 0.7
- has_past_kv: True
- input_tokens: 821
- output_tokens: 619
- duration_s: 23.0597
- text_len: 3055

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
adjacent hypothesis direction that the results motivate — grounded in what the
data already showed.

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
The parent result — its hypothesis, factors, metrics, and Feedback
diagnosis — is already in your shared latent memory. Read it from there.

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
**Step 1 — Read the Feedback weakness breakdown:**  
The Feedback identified several structural issues:  
- Every factor uses a single raw variable with no normalization.  
- Window of five days is too short for a medium-horizon mechanism.  
- Composition depth is shallow (single-step function instead of two-step structure).  
- Function family is unclear — the signal may not be capturing what it claims.  

These weaknesses are **fixable within the current mechanism**. For example, adding normalization, adjusting window length, and deepening the composition could improve the signal. The function family is less clear but still actionable.

---

**Step 2 — What the parent mechanism still gets right:**  
The parent mechanism still captures some positive signal, particularly in **trending, high-volume regimes**, where the combined model shows a modest return of 0.115 annualized and an information ratio of 0.63. The standalone RankIC of 0.035 suggests that the factor still has some predictive power, though it’s unstable (ICIR = 0.48).

---

**Step 3 — Most critical fixable weakness and direction:**  
The most critical weakness is the **shallow composition depth**: the factor is a single-step function (Momentum_Volume_Confirm_5D) but the hypothesis likely assumes a two-step structure (e.g., a time-series statistic followed by cross-sectional standardisation). To address this, the **composition should be deepened** by adding a cross-sectional normalization layer after the time-series mean. This would remove scale sensitivity and make the signal more robust to market conditions. For example, instead of just multiplying the rank of the mean return by the sign of the volume change, we should normalize the mean return using cross-sectional z-scores before applying the sign. This would better capture the relative strength of the momentum signal across assets.

---

**Step 4 — Adjacent hypothesis direction inspired by failures:**  
The parent’s failure to achieve strong stability (ICIR = 0.48) suggests that the signal is too sensitive to short-term noise. The data hints at a **longer-term momentum pattern** driven by **market sentiment shifts** — when investors are overly optimistic or pessimistic, certain assets outperform others over weeks rather than days. A new hypothesis could focus on **weekly momentum with cross-sectional normalization** and **regime-based gating** (e.g., using VIX or trading volume as a gate). This would target a different part of the market cycle and potentially yield a more stable, higher-information signal.

---

**WHAT WORKS:** The parent mechanism still captures some signal in trending, high-volume regimes.  
**DIAGNOSIS:** The factor lacks a proper two-step composition and normalization.  
**DIRECTION:** Add a cross-sectional normalization layer after the time-series mean to deepen the composition and reduce scale sensitivity.  
**NEW HYPOTHESIS:** A weekly momentum signal with cross-sectional normalization and regime-based gating, targeting long-term sentiment shifts in the market.
```
