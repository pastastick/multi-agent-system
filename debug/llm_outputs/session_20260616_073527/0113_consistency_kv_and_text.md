# Call 0113 — `consistency` (kv_and_text)

## Meta

- ts: 2026-06-16 08:33:24
- conv_id: `8a547b48`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 199
- output_tokens: 512
- duration_s: 42.1809
- text_len: 0

## System Prompt

```text
You are the CRITIC (stage 3 of 4). The hypothesis and candidate expression(s)
are in latent memory. Silently critique and, in your reasoning, FIX flaws — do
NOT emit a final answer (the Judger does). For each expression verify:
  - FAITHFUL: measures the hypothesis with the correct sign.
  - VALID: arity correct, every leaf a real $variable, windows 1-60, no
    series-with-itself in REGBETA/REGRESI/TS_CORR/TS_COVARIANCE.
  - SIMPLE: drop overfit/bolted-on parts.
Keep only expressions that are faithful, valid, and simple; if several, each
must add a distinct angle on the SAME hypothesis.
```

## User Prompt

```text
Critique and refine the hypothesis-expression pair(s) in memory: faithfulness,
validity, parsimony.
```

## Response

```text

```
