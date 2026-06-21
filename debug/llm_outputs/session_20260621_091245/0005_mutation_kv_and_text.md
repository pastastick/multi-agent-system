# Call 0005 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-21 09:15:15
- conv_id: `d13851cc`
- step: 0
- temperature: 0.7
- has_past_kv: True
- input_tokens: 821
- output_tokens: 646
- duration_s: 25.4578
- text_len: 3258

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
The Feedback identified multiple structural weaknesses across the factors, including:  
- Use of a single raw variable (Momentum_Raw_3D) without transformation or normalization.  
- No cross-sectional normalization, leading to scale-sensitive signals.  
- A window length of 3 days, which is too short for a meaningful momentum signal.  
- A single-step function (TS_MEAN) without cross-sectional standardization, which is needed for a two-step mechanism.  
- No conditional gate to distinguish market states.  

All of these are fixable within the current mechanism through structural adjustments—no fundamental mechanism change is required.

**Step 2 — What the parent mechanism still gets right:**  
The parent mechanism retains a basic momentum hypothesis, and the factor Momentum_Raw_3D shows some minimal predictive power (RankIC = 0.005), though it is extremely low and unstable (ICIR = 0.08). This suggests that the data may still contain some latent momentum signal, albeit weak and noisy.

**Step 3 — Most critical fixable weakness and direction:**  
The most critical weakness is the use of a single raw variable without normalization or cross-sectional standardization. The current factor uses TS_MEAN($return, 3), which is a raw price-based mean and lacks any form of cross-sectional normalization. To address this, we should replace the raw mean with a **cross-sectional normalized momentum statistic**, such as a **cross-sectional rank of 3-day returns standardized by volatility**. This would remove scale sensitivity, introduce more robust signal structure, and better align with the hypothesis of momentum in a cross-sectional section.

**Step 4 — Adjacent hypothesis direction inspired by failures:**  
The failures suggest that the current momentum hypothesis is too simplistic and fails to account for market state variations. The data may still contain momentum, but it is likely more pronounced in certain regimes (e.g., low volatility or trending markets). An adjacent hypothesis could be: **momentum in low-volatility regimes**, driven by liquidity imbalances or informational asymmetry. Variables that carry this pattern might include **price-to-volume ratios**, **cross-sectional volatility**, and **liquidity measures**. The expected return direction would be positive in these regimes, as momentum tends to be stronger when markets are less efficient. The parent’s failure to capture this suggests that the signal is not universally present and requires conditioning on market state.

---

**WHAT WORKS:** The parent mechanism still gets a minimal but technically present momentum signal from the 3-day return mean.  
**DIAGNOSIS:** The factor uses a raw, unnormalized mean without cross-sectional standardization, leading to a weak and unstable signal.  
**DIRECTION:** Replace the raw 3-day return mean with a cross-sectional normalized momentum statistic (e.g., rank-standardized 3-day return).  
**NEW HYPOTHESIS:** Momentum in low-volatility regimes, driven by liquidity imbalances. Variables like price-to-volume ratios and cross-volatility could capture this. The parent’s failure to detect it suggests the signal is regime-dependent and requires conditional filtering.
```
