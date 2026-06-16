# Call 0027 — `consistency` (kv_and_text)

## Meta

- ts: 2026-06-16 07:33:02
- conv_id: `1c6fe2fb`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 158
- output_tokens: 61
- duration_s: 4.4119
- text_len: 332

## System Prompt

```text
You are a Consistency Agent (refiner). The hypothesis and a candidate factor
expression are in your latent memory. Silently check and correct, in your
reasoning, any mismatch between the hypothesis intent and the expression:
  - does the expression actually measure what the hypothesis claims?
  - is any operator's arity wrong? is any variable not a runtime $variable?
  - is the expression overly complex (likely to overfit)?
Refine toward an expression that is faithful, valid, and simple. Do not
output a final answer — the Judger will.
```

## User Prompt

```text
Refine the hypothesis–expression pair held in latent memory for faithfulness,
validity, and parsimony.
```

## Response

```text
I am silently reviewing the hypothesis and expression in my latent memory. I check whether the expression accurately reflects the hypothesis, whether there are any issues with operator arity or variable usage, and whether the expression is overly complex. I am not outputting a final answer — the Judger will make the determination.
```
