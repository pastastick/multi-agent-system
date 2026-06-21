# Call 0030 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-21 12:25:11
- conv_id: `d909f301`
- step: 0
- temperature: 0.7
- has_past_kv: True
- input_tokens: 821
- output_tokens: 579
- duration_s: 23.5427
- text_len: 2994

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
**Step 1 — Read the Feedback weakness breakdown.**  
The Feedback identified several structural weaknesses:  
- Every factor uses a single raw variable with no normalization.  
- Window of five days is too short for a medium-horizon mechanism.  
- Composition depth is insufficient (single-step functions instead of two-step structures like time-series statistic + cross-sectional-sectional).  
- Function family misalignment (e.g., using a z-score instead of a regression slope for trend capture).  

These weaknesses are all fixable within the current mechanism. The data suggests that the underlying hypothesis is sound but the construction is weak, so the direction should focus on refining the existing framework.

**Step 2 — Name what the parent mechanism still gets right:**  
The parent mechanism still gets some signal, albeit weak and unstable. Factor 1 shows the most promise with a RankIC of 0.031 and ICIR of 0.41, suggesting it captures a small but somewhat stable predictive pattern in the data.

**Step 3 — For the most critical fixable weakness, name the ONE concrete structural change:**  
The most critical weakness is the lack of normalization and the use of single raw variables. To address this, we should introduce **cross-sectional rank normalization** to Factor 1. Instead of using raw volume or price variables, we can apply a cross-sectional rank or z-score to the raw variables, which would reduce noise and make the signal more robust. This change aligns with the hypothesis that the signal is driven by relative performance rather than absolute values, and it addresses the instability issue seen in the ICIR.

**Step 4 — Inspired by what the failures revealed about the data, name ONE adjacent hypothesis direction:**  
The failures suggest that the current factors are too focused on short-term price movements and not capturing longer-term trends. An adjacent hypothesis could be based on **long-term momentum patterns driven by institutional buying pressure**. Variables like institutional flow, long-term return trends, and volume concentration could carry this signal. The expected return direction would be positive in trending markets, as institutional investors tend to follow long-term trends. The parent’s failure to capture this suggests that the current factors are too short-windowed and lack the depth needed to model such a mechanism.

---

**WHAT WORKS:** Factor 1 shows a small but somewhat stable predictive pattern in the data.  
**DIAGNOSIS:** Lack of normalization and reliance on single raw variables reduce the signal's robustness.  
**DIRECTION:** Apply cross-sectional rank normalization to Factor 1 to reduce noise and improve stability.  
**NEW HYPOTHESIS:** Long-term momentum patterns driven by institutional buying pressure, carried by institutional flow and volume concentration, with positive returns in trending markets. The parent’s failure to capture this suggests the need for longer windows and deeper compositional structure.
```
