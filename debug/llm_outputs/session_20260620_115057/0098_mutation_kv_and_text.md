# Call 0098 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-20 14:10:34
- conv_id: `9963501e`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 1239
- output_tokens: 462
- duration_s: 17.0827
- text_len: 2266

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
Factor 1 (MeanReversion_5D) has a standalone RankIC of 0.041 and ICIR of 0.22. This is the strongest factor in Block A, but its ICIR is low, indicating instability. Block B shows that when combined with other factors, it contributes to a catastrophic drawdown and collapses in trending regimes.

---

**FACTOR DIAGNOSIS**  
- **Factor 1 (MeanReversion_5D):** Strong signal but unstable construction. The factor uses a single raw variable (TS_MEAN($return, 5)) without normalization, leading to scale-sensitive noise. The window of 5 days is too short for a medium-horizon mean-reversion mechanism, and the function family (simple mean) does not capture the direction of trend effectively.  
- **Other factors:** Not evaluated here, but based on the Replace-best decision, it is clear that Factor 1 is the only one considered in this round.

---

**WEAKNESS BREAKDOWN**  
1. Every factor uses a single raw variable without normalization, leading to scale-sensitive noise.  
2. Window length of 5 days is too short for a medium-horizon mechanism, capturing too much noise.  
3. Lack of cross-sectional normalization (e.g., ranking or z-scoring) reduces the signal’s robustness.  
4. Function family is overly simplistic (e.g., mean) and fails to capture trend direction or momentum.  
5. No conditional structure or regime gates are applied, leading to poor performance in trending regimes.

---

**HYPOTHESIS VERDICT**  
Partially supports. The hypothesis of mean reversion is visible in the data (RankIC = 0.041), but the construction is weak. The expression is too simple, lacks normalization, and fails to account for regime shifts. The model needs refinement in terms of function family, normalization, and conditional structure.

---

**REPLACE BEST**  
Yes. The current FactorIC_mean of 0.041 is below the SOTA FactorIC_mean of 0.025, so replacement is justified.

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
The weaknesses identified are all structural and fixable within the current mechanism. The core issue is that the factor is too simple, scale-sensitive, and lacks robustness in trending regimes. The hypothesis of mean reversion is still partially valid, but its construction is inadequate.

**Step 2 — What the parent mechanism still gets right:**  
The parent mechanism shows a small but visible signal in the form of a positive RankIC of 0.041, indicating that there is some mean-reverting behavior in the data, albeit unstable and limited in scope.

**Step 3 — For the most critical fixable weakness:**  
The most critical weakness is the lack of cross-sectional normalization and the use of a single raw variable without scaling. This leads to scale-sensitive noise and weak signal robustness. To address this, we should apply **cross-sectional z-scoring** to the raw variable (TS_MEAN($return, 5)) before using it in the model. This would normalize the factor across different assets and time periods, reducing noise and improving the signal’s stability and reliability.

**Step 4 — Inspired by what the failures revealed about the data:**  
The failures suggest that the current mean-reversion approach is too simplistic and fails to adapt to trending regimes. This points toward an adjacent hypothesis where **trend-following momentum** becomes the dominant driver in certain market conditions. Variables like price momentum, volume, and volatility could carry this pattern, and the expected return direction would be positive in trending regimes, with negative returns in mean-reverting ones.

---

WHAT WORKS: The parent mechanism shows a small but visible mean-reverting signal in the form of a positive RankIC of 0.041.  
DIAGNOSIS: The factor is too simple, scale-sensitive, and lacks robustness in trending regimes.  
DIRECTION: Apply cross-sectional z-scoring to the raw variable (TS_MEAN($return, 5)) to reduce scale-sensitive noise and improve signal stability.  
NEW HYPOTHESIS: The data suggests a shift toward trend-following momentum as a dominant driver in certain regimes. Variables like price momentum and volume could carry this pattern, with expected positive returns in trending markets.
```
