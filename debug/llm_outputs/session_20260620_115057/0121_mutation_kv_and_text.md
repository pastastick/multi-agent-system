# Call 0121 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-20 14:22:58
- conv_id: `521c4189`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 1197
- output_tokens: 431
- duration_s: 17.7535
- text_len: 2324

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
Factor 1 (Momentum_Volume_Confirm_5D) has the strongest standalone RankIC (0.035) and a solid ICIR (0.48), placing it above other factors in predictive power and stability. Other factors are not described, but based on the replacement decision, it is clear that this factor is the best in this round.

---

**FACTOR DIAGNOSIS**  
- **Factor 1**: Strong  
  The signal is built from a combination of a time-series mean (TS_MEAN) and a sign of percentage change in volume (TS_PCTCHANGE). The RankIC suggests some predictive power, and the ICIR indicates reasonable stability. However, the use of a raw price-based mean (without normalization) and a single-step function may limit its effectiveness. The factor does not incorporate cross-sectional rank or z-score normalization, which could improve signal clarity.

---

**WEAKNESS BREAKDOWN**  
1. All factors rely on raw price or volume variables without cross-sectional normalization, leading to scale sensitivity and noise.  
2. The window length of 5 days is too short for a medium-horizon momentum mechanism, capturing too much noise.  
3. The composition depth is one-step, but the hypothesis likely requires a two-step structure (e.g., time-series statistic followed by cross-sectional standardization).  
4. The function family used (mean and sign) does not fully capture the intended trend or directionality of the hypothesis.  
5. There is no conditional gate to filter for specific market regimes, limiting the factor’s applicability in different market states.

---

**HYPOTHESIS VERDICT**  
Partially supports. The mechanism (momentum combined with volume confirmation) is sound and visible in the data, but the construction is weak. The factor needs refinement in terms of variable choice, normalization, and composition depth to better capture the intended signal.

---

**REPLACE BEST**  
Yes

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
The weaknesses point to structural issues that can be fixed within the current mechanism. The use of raw variables, short window, one-step composition, and simple function family are all fixable with specific adjustments. The lack of conditional gating suggests a more complex mechanism might be needed, but this is a secondary concern.

**Step 2 — What the parent mechanism still gets right:**  
The combination of momentum and volume confirmation produces a signal with positive RankIC, indicating that the core idea of using momentum in conjunction with volume as a confirmation factor is valid.

**Step 3 — For the most critical fixable weakness:**  
The most critical weakness is the lack of cross-sectional normalization, which leads to scale sensitivity and noise. To address this, we should add a cross-sectional z-score normalization layer to the TS_MEAN component. This would standardize the mean across assets, reducing scale bias and improving the clarity of the signal. This change directly addresses the issue of noise and scale sensitivity by ensuring the signal is comparable across different assets.

**Step 4 — Inspired by what the failures revealed about the data:**  
The data suggests that the current mechanism has limited headroom because it relies on a single-step structure and raw variables. An adjacent hypothesis direction could be a two-step momentum mechanism that first applies a time-series statistic (like a moving average) and then applies cross-sectional standardization. This approach would better capture medium-horizon trends while reducing noise, making the factor more robust in different market conditions.

---

WHAT WORKS: The combination of momentum and volume confirmation produces a signal with positive RankIC.  
DIAGNOSIS: The lack of cross-sectional normalization introduces scale sensitivity and noise.  
DIRECTION: Add a cross-sectional z-score normalization layer to the TS_MEAN component to reduce scale bias and improve signal clarity.  
NEW HYPOTHESIS: A two-step momentum mechanism that first applies a time-series statistic and then applies cross-sectional standardization. This approach would better capture medium-horizon trends while reducing noise, making the factor more robust in different market conditions.
```
