# Call 0009 — `judger` (kv_and_text)

## Meta

- ts: 2026-06-13 14:06:08
- conv_id: `11059fe5`
- step: 0
- temperature: 0.6
- has_past_kv: True
- input_tokens: 200
- output_tokens: 92
- duration_s: 4.1071
- text_len: 313

## System Prompt

```text
You are the Judger — final stage (4 of 4). The Proposal, Construct, and
Consistency agents' full reasoning — ONE hypothesis and its 1-3 validated
expression(s) — is already in your memory. SOLE JOB: OUTPUT it. Do NOT
re-reason, re-derive, or invent new expressions; do NOT change any operator,
argument, or window from what was reasoned.

Output EXACTLY these lines, nothing else (no prose, no JSON, no fences):
HYPOTHESIS: <one sentence — the mechanism>
EXPRESSION 1: <single DSL expression>
EXPRESSION 2: <single DSL expression>   (only if a distinct second factor exists)
```

## User Prompt

```text
Research direction: price-volume momentum in high-volatility sectors during bear markets
Output the hypothesis and expression line(s) now, from the analysis in your memory.
```

## Response

```text
HYPOTHESIS: When high-volume stocks in volatile sectors show a close below the 20-day moving average and volume above the 30-day median, their next-day return is positively skewed and reverses from the open.
EXPRESSION 1: RANK($close < $sma_20) * ZSCORE($volume - $ts_median($volume, 30)) * TS_ZSCORE($return, 20)
```
