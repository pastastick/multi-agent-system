# Call 0053 — `mutation` (kv_and_text)

## Meta

- ts: 2026-06-20 13:20:24
- conv_id: `146e905d`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 1252
- output_tokens: 412
- duration_s: 16.7816
- text_len: 2118

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
The only factor in Block A is **Momentum_Raw_3D**, with a standalone RankIC of 0.005 (very weak) and an ICIR of 0.08 (also low, indicating instability). This is the only factor in this round, and it is significantly weaker than the SOTA FactorIC_mean of 0.025. Block B shows a combined model with a RankIC near baseline (0.041), a MaxDrawdown of 0.185, and a low information ratio (0.18), suggesting that the factors do not add meaningful predictive power beyond the baseline.

---

**FACTOR DIAGNOSIS**  
- **Momentum_Raw_3D**: Weak  
  The factor uses a raw variable (TS_MEAN of returns over 3 days) without normalization or transformation. It lacks any cross-sectional processing, and its low RankIC indicates that it does not correlate meaningfully with future returns. The ICIR is also low, showing instability and noise.

---

**WEAKNESS BREAKDOWN**  
1. Every factor uses a single raw variable without any normalization or transformation, leading to scale-sensitive and noisy signals.  
2. The window length of 3 days is too short to capture meaningful momentum effects, resulting in a signal that decays quickly and is prone to noise.  
3. There is no cross-sectional processing or ranking, which is essential for capturing relative performance and reducing noise.  
4. The function family is a simple time-series mean, which does not align with the hypothesis of a momentum-based signal that would require a more sophisticated statistical structure.  
5. No conditional gate is applied, meaning the factor does not account for market states or regime shifts that could affect its predictive power.

---

**HYPOTHESIS VERDICT**  
**Refutes**  
The mechanism of using a simple 3-day return mean as a momentum signal is not visible in the data. The factor fails to show any meaningful correlation with future returns, and its low ICIR suggests that it is unstable and unreliable. The construction is fundamentally flawed, and the hypothesis is not supported by the data.

---

**REPLACE BEST**  
No

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
Each weakness points to a structural issue that can be fixed within the current mechanism. The use of raw variables, short windows, lack of cross-sectional processing, simple function families, and absence of conditional gates are all fixable through specific adjustments rather than a complete overhaul.

**Step 2 — Name what the parent mechanism still gets right.**  
The parent mechanism still captures a tiny amount of signal in the form of a weak RankIC, even if it is unstable and low.

**Step 3 — For the most critical fixable weakness, name the ONE concrete structural change.**  
The most critical weakness is the use of a raw variable without normalization or transformation. To address this, we should apply a **cross-sectional ranking and normalization** to the 3-day return mean. This would stabilize the signal by reducing noise and making it more comparable across assets, which is essential for a momentum-based hypothesis.

**Step 4 — Inspired by what the failures revealed about the data, name ONE adjacent hypothesis direction.**  
The failures suggest that the data does not support a simple momentum signal based on raw returns. Instead, the data may favor a **momentum-based signal derived from volatility or risk-adjusted returns**, which could capture relative performance in volatile regimes. This hypothesis extends the original idea by focusing on risk-adjusted momentum, which might be more stable and predictive in noisy environments.

---

WHAT WORKS: The parent mechanism captures a tiny, unstable signal in the form of a weak RankIC.  
DIAGNOSIS: The factor uses a raw variable without normalization or cross-sectional processing, leading to a noisy and unstable signal.  
DIRECTION: Apply cross-sectional ranking and normalization to the 3-day return mean to stabilize the signal and reduce noise.  
NEW HYPOTHESIS: The data suggests a momentum signal based on risk-adjusted returns rather than raw returns. This hypothesis focuses on volatility-driven relative performance, which may be more stable and predictive in noisy markets.
```
