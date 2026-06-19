# Call 0043 — `judger` (kv_and_text)

## Meta

- ts: 2026-06-19 04:59:53
- conv_id: `ee26b36b`
- step: 0
- temperature: 0.7
- has_past_kv: True
- input_tokens: 513
- output_tokens: 101
- duration_s: 11.3674
- text_len: 264

## System Prompt

```text
You are the Formatter. A finished hypothesis and its validated factor
expression(s) are in your working memory. Your task: surface them and output
them as valid expressions in the exact format.

Because they come to you from working memory, your PRIMARY responsibility is
SYMBOLIC VALIDITY -- every expression you write must use real operator names
(exact, case-sensitive), correct arity, balanced structure, and be a
continuous cross-sectional score. Do NOT re-reason the mechanism, swap a
family, add a factor, or reconsider the research direction -- that work is
done; only render what is in memory and fix a symbol that surfaces clearly
garbled (a name not in the list below, unbalanced parentheses, a missing
window) or drop a line you cannot render cleanly.

Output EXACTLY these lines, nothing else -- ASCII, no prose, no JSON, no
fences, no markdown (no bold, no backticks, no bullets):
HYPOTHESIS: <one sentence -- the mechanism>
EXPRESSION 1: <single DSL expression -- a continuous numeric score>
EXPRESSION 2: <single DSL expression>   (only if a distinct second factor)

Valid operator names (everything else is invalid):
  Vars:    $open $high $low $close $volume $return
  1 arg:   RANK ZSCORE MEAN STD SKEW KURT MEDIAN LOG SQRT SIGN EXP ABS INV FLOOR
  (A,n):   DELTA DELAY TS_MEAN TS_SUM TS_RANK TS_ZSCORE TS_MEDIAN TS_STD TS_VAR
           TS_MIN TS_MAX TS_ARGMAX TS_ARGMIN TS_MAD TS_PCTCHANGE SUMAC HIGHDAY
           LOWDAY RSI WMA EMA PROD DECAYLINEAR POW BB_UPPER BB_MIDDLE BB_LOWER
  (A,B,n): TS_CORR TS_COVARIANCE REGBETA REGRESI   (A and B DIFFERENT series)
  3+ arg:  SMA MACD TS_QUANTILE PERCENTILE SUMIF COUNT FILTER SEQUENCE
  other:   MAX(A,B) MIN(A,B)   (C)?(A):(B)   arithmetic + - * /   logical && ||
```

## User Prompt

```text
Surface the finished hypothesis and expression line(s) from your working
memory and output them now, in the exact format. Repair only a clearly garbled
symbol.
```

## Response

```text
HYPOTHESIS: When a stock's $close is above its 20-day EMA and the $volume is above its 20-day MA, it is more likely to outperform the next day.

EXPRESSION 1: TS_ZSCORE($close, 20) > 0 AND TS_MEAN($volume, 20) > 0  
EXPRESSION 2: TS_CORR($close, $return, 20) > 0.5
```
