# Call 0049 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-21 10:36:18
- conv_id: `dc04ac20`
- step: 0
- temperature: 0.7
- has_past_kv: True
- input_tokens: 821
- output_tokens: 484
- duration_s: 21.0345
- text_len: 2537

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
The Feedback identified several structural issues: every factor uses a single raw variable with no normalization, the window length is too short (3 days) for a medium-horizon mechanism, and the function family (TS_MEAN) does not capture trend direction effectively. These are all fixable within the current mechanism through structural adjustments.

**Step 2 — What the parent mechanism still gets right:**  
The parent mechanism still gets a small but positive signal in the form of a momentum-based factor, though it is weak and unstable. The fact that it shows any RankIC at all suggests that there is some underlying momentum or trend component in the data that the model is attempting to capture.

**Step 3 — Most critical fixable weakness and concrete direction:**  
The most critical weakness is the use of a single raw variable with no normalization. This leads to scale-sensitive signals that are unreliable and unstable. To fix this, we should add a cross-sectional rank or z-score normalization to the raw return variable. This would make the signal more stable and less sensitive to price levels, aligning better with the hypothesis of momentum-based alpha.

**Step 4 — Adjacent hypothesis direction inspired by failures:**  
The failures suggest that the current momentum mechanism is too simplistic and lacks the ability to filter out noise. A more promising direction is to explore a **trend-following strategy based on relative strength between sectors**. Variables like sector returns, volume ratios, and earnings surprises could carry this pattern. The expected return direction would be positive in rising markets and negative in falling markets, as stronger sectors tend to outperform weaker ones during trends.

---

**WHAT WORKS:** The parent mechanism still gets a small but positive signal in the form of a momentum-based factor.  
**DIAGNOSIS:** The factor uses a single raw variable with no normalization, leading to scale-sensitive and unstable signals.  
**DIRECTION:** Add a cross-sectional rank or z-score normalization to the raw return variable to make the signal more stable and less sensitive to price levels.  
**NEW HYPOTHESIS:** A trend-following strategy based on relative strength between sectors. Variables like sector returns, volume ratios, and earnings surprises carry this pattern. The expected return direction is positive in rising markets and negative in falling markets, as stronger sectors tend to outperform weaker ones during trends.
```
