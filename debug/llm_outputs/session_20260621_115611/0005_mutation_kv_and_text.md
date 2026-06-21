# Call 0005 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-21 11:58:55
- conv_id: `299b86de`
- step: 0
- temperature: 0.7
- has_past_kv: True
- input_tokens: 821
- output_tokens: 606
- duration_s: 22.9402
- text_len: 3114

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
The Feedback identified several structural weaknesses:  
- Every factor uses a single raw variable with no normalization.  
- Window length is too short (3 days) for a medium-horizon momentum mechanism.  
- No cross-sectional normalization or z-score transformation is applied.  
- Composition depth is shallow (single-step function, no time-series statistic followed by cross-sectional standardization).  
- Function family is not aligned with the hypothesis (a simple mean, not a slope or trend measure).  
- No conditional gate is used to filter for market states.

Of these, the first three (variable choice, window, and normalization) are fixable within the current mechanism. The others suggest that the mechanism may be limited in its ability to capture complex, multi-step signals or conditional dynamics.

---

**Step 2 — What the parent mechanism still gets right:**  
The parent mechanism still gets a small but positive signal in the form of a 3-day momentum, though it is very weak and unstable. The fact that any RankIC is positive suggests that there is some residual predictive power in the data, even if it’s not strong or consistent.

---

**Step 3 — Most critical fixable weakness and direction:**  
The most critical fixable weakness is the **use of a single raw variable with no normalization**. The 3-day momentum is too noisy and lacks the ability to isolate meaningful trends from price fluctuations. To address this, we should **replace the raw return with a cross-sectional z-score of returns**, which would normalize the signal across assets and remove noise. This change would align the factor with the hypothesis of capturing momentum in a more stable, cross-sectional normalized way.

---

**Step 4 — Adjacent hypothesis direction inspired by failures:**  
The failures suggest that the current momentum mechanism is too simplistic and fails to account for the decay of trend persistence over time. A more promising direction is to explore **longer-term momentum patterns driven by earnings surprises**, where the signal is derived from a 60-day moving average of earnings surprises, normalized across sectors. This hypothesis extends the parent’s focus on momentum but shifts it toward a more stable, multi-step signal that could better capture structural trends in earnings-driven markets.

---

**WHAT WORKS:** The parent mechanism still gets a small but positive signal in the form of a 3-day momentum, though it is very weak and unstable.  
**DIAGNOSIS:** The use of a single raw variable with no normalization is the most critical structural weakness.  
**DIRECTION:** Replace the raw return with a cross-sectional z-score of returns to normalize the signal and reduce noise.  
**NEW HYPOTHESIS:** Explore longer-term momentum patterns driven by earnings surprises, using a 60-day moving average of earnings surprises normalized across sectors. This hypothesis extends the parent’s focus on momentum but shifts it toward a more stable, multi-step signal that could better capture structural trends in earnings-driven markets.
```
