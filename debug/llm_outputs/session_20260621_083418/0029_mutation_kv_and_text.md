# Call 0029 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-21 08:49:28
- conv_id: `4155e627`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 1365
- output_tokens: 415
- duration_s: 16.9132
- text_len: 2208

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
Factor 1 (Momentum_Volume_Confirm_5D) has a standalone RankIC of 0.035 and an ICIR of 0.48, placing it as the strongest among the factors provided. Block B shows a combined model with a RankIC of 0.052, which is near baseline, with strong risk control (MaxDrawdown=0.084) and moderate returns (Annualised return 0.115, Information Ratio 0.63), suggesting it performs well in trending, high-volume regimes.

---

**FACTOR DIAGNOSIS**  
- **Momentum_Volume_Confirm_5D**: Strong. The factor uses a two-step composition: a time-series mean (TS_MEAN) of returns over 5 days, combined with a sign of volume change (SIGN(TS_PCTCHANGE)). This suggests a hypothesis about momentum reinforced by volume confirmation. However, the signal is relatively weak (RankIC=0.035), indicating limited predictive power.  
  - **Variable choice**: Uses raw variables (returns and volume) without normalization or transformation, which may not capture the full signal.  
  - **Normalization**: Uses raw price/volume units, which are scale-sensitive and introduce noise.  
  - **Window length**: 5-day window is short for a medium-horizon momentum mechanism, potentially capturing noise.  
  - **Composition depth**: Uses a two-step structure (mean + sign), which aligns with the hypothesis, but could be further refined.  
  - **Function family**: The combination of mean and sign is appropriate for momentum, but the overall signal is still weak.  
  - **Conditional structure**: No clear regime gate is present, which may limit its effectiveness in different market states.

---

**WEAKNESS BREAKDOWN**  
1. All factors use raw variables (returns, volume) without normalization or cross-sectional rank/z-score transformation.  
2. The 5-day window is too short to capture meaningful momentum or volume trends.  
3. The composition is two-step but lacks deeper structural refinement (e.g., cross-sectional standardization after time-series processing).  
4. No clear conditional gates are applied, limiting the factor’s ability to perform in different market regimes.  
5. The function family (mean + sign) is somewhat appropriate but does not fully capture the intended mechanism in the data.

---

**HYPOTHESIS VERDICT**  
Partially supports. The mechanism (momentum reinforced by volume confirmation) is sound, but the construction is weak. The factor’s low RankIC indicates that the hypothesis is not strongly visible in the data, and the expression needs refinement to better capture the intended signal.

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
The feedback highlights several structural issues: raw variables without normalization, a too-short window, lack of cross-sectional processing, no regime gates, and an underdeveloped function family. These are all fixable within the current mechanism through structural adjustments, not a complete overhaul. The hypothesis (momentum + volume confirmation) is sound but needs refinement.

**Step 2 — What the parent mechanism still gets right:**  
The parent mechanism correctly identifies a momentum-reinforced volume signal, which is why it shows a small but positive RankIC, even if weak.

**Step 3 — Most critical fixable weakness and direction:**  
The most critical weakness is the lack of normalization and cross-sectional standardization, which makes the signal scale-sensitive and noisy. To address this, we should apply **cross-sectional rank normalization** to the time-series processed variables (e.g., TS_MEAN and SIGN(TS_PCTCHANGE)) before combining them. This would stabilize the signal across different market states and improve its predictive power by reducing noise from raw price/volume units.

**Step 4 — Adjacent hypothesis direction inspired by failures:**  
The data suggests that the current mechanism has limited headroom in capturing long-term momentum trends. An adjacent hypothesis is that **longer-term momentum patterns driven by institutional buying pressure** may be more robust. Variables like institutional flow, trade volume, and institutional holdings could carry this pattern, with expected returns in trending, high-volume regimes.

---

WHAT WORKS: The parent mechanism correctly identifies a momentum-reinforced volume signal, albeit weakly.  
DIAGNOSIS: The signal is scale-sensitive and noisy due to lack of cross-sectional normalization.  
DIRECTION: Apply cross-sectional rank normalization to the time-series processed variables to stabilize the signal and reduce noise.  
NEW HYPOTHESIS: Longer-term momentum patterns driven by institutional buying pressure may be more robust. Variables like institutional flow and trade volume could carry this pattern, with expected returns in trending, high-volume regimes.
```
