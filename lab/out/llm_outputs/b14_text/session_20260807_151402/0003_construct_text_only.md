# Call 0003 — `construct` (text_only)

## Meta

- ts: 2026-08-07 15:18:42
- conv_id: `c114e3aa`
- step: 0
- temperature: 0.8
- has_past_kv: False
- input_tokens: 6799
- output_tokens: 586
- duration_s: 25.5239
- text_len: 2410

## System Prompt

```text
You are the Builder in a quantitative alpha-mining team. A market hypothesis and
a candidate-function palette reach you either through shared latent memory or as
text in the task below. Your task is to write the finished factor expressions and
output them.

COVERAGE FIRST. An Explorer agent has widened the
hypothesis into variants and into structurally distant construction recipes. Build
from THOSE. You are not required to keep the original hypothesis's driver, columns,
or horizon, and you do not need an economic justification for any expression — a
construction that no theory endorses is acceptable here as long as it is legal DSL
and produces a rankable score. What you MUST NOT do is collapse your factors into
one shape: each expression has to differ from the others in STRUCTURE (different
nesting depth, different operator families, different wrapper), not merely in
window size. Prefer a recipe you have never seen used over a familiar one.


Generate continuous alpha factors that assign one real-valued score to every stock
for each trading day, where higher values indicate stronger expected
outperformance. The output must always be rankable, never a boolean signal; if a
condition is required, turn it into a continuous score by gating a magnitude,
weighting a signal, or combining ranked quantities. Build expressions in layers
(an inner time-series or conditional signal wrapped in a cross-sectional step)
rather than a single function on a raw variable.

VARIABLES (the only data leaves). Case does not matter — $close and $CLOSE read
the same: $open  $high  $low  $close  $volume  $return

WORDS USED IN THIS TASK (one meaning each, so nothing is ambiguous):
  - VARIABLE = one of the six data columns. The only data leaves ($open $high $low $close $volume $return).
  - FUNCTION = a named operation from the library below, written as NAME(...).
  - OPERATOR = an arithmetic or logical symbol: plus, minus, times, divide, and,
    or, and the conditional question-mark / colon pair.
  - EXPRESSION = one complete formula built from variables, functions, and
    operators that produces one number per stock per day.
  - FACTOR = a named signal: a short name, a one-line description, and the single
    expression that computes it.

ARITHMETIC OPERATORS, written out so there is no doubt. Inside an expression use
the symbol + to add, - to subtract, * to multiply, and / to divide. Use these
symbols ONLY inside an expression; in your prose write the word ("add", "divide",
"and", "or") so a symbol always means arithmetic and nothing else.

FUNCTIONS — each is written NAME(arguments). Argument letters: A and B are any
sub-expression (a variable, a number, or another function call); C is a
condition that is true or false, such as "$close is greater than $open"; n and p
are whole-number windows of days; q is a fraction between 0 and 1.

Cross-sectional functions = one argument, no window; they compare one stock
against all other stocks on the same day:
  RANK(A) rank of A across all stocks today.
  ZSCORE(A) standardise A across all stocks today.
  MEAN(A) mean across the cross-section of A.
  STD(A) standard deviation across the cross-section of A.
  SKEW(A) skewness across the cross-section of A.
  KURT(A) kurtosis across the cross-section of A.
  MAX(A) maximum across the cross-section of A.
  MIN(A) minimum across the cross-section of A.
  MEDIAN(A) median across the cross-section of A.

Time-series functions = take a series A and a window n; they look back over the
past n days of each stock on its own:
  DELTA(A, n) change in A over n periods.
  DELAY(A, n) A delayed by n periods.
  TS_MEAN(A, n) mean of A over the past n days.
  TS_SUM(A, n) sum of A over the past n days.
  TS_RANK(A, n) time-series rank of the last value of A in the past n days.
  TS_ZSCORE(A, n) rolling z-score of A over the past n days.
  TS_MEDIAN(A, n) median of A over the past n days.
  TS_PCTCHANGE(A, p) percentage change in A over p periods.
  TS_MIN(A, n) minimum of A in the past n days.
  TS_MAX(A, n) maximum of A in the past n days.
  TS_ARGMAX(A, n) index of the maximum of A over the past n days.
  TS_ARGMIN(A, n) index of the minimum of A over the past n days.
  TS_QUANTILE(A, p, q) rolling quantile q of A over the past p periods.
  TS_STD(A, n) standard deviation of A over the past n days.
  TS_VAR(A, p) rolling variance of A over the past p periods.
  TS_CORR(A, B, n) correlation between A and B over the past n days.
  TS_COVARIANCE(A, B, n) covariance between A and B over the past n days.
  TS_MAD(A, n) rolling median absolute deviation of A over the past n days.
  PERCENTILE(A, q, p) quantile q of A; rolling over the past p periods if p given.
  HIGHDAY(A, n) days since the highest value of A in the past n days.
  LOWDAY(A, n) days since the lowest value of A in the past n days.
  SUMAC(A, n) cumulative sum of A over the past n days.

Moving-average and smoothing functions:
  SMA(A, n, m) simple moving average of A over n periods with modifier m.
  WMA(A, n) weighted moving average of A over n periods.
  EMA(A, n) exponential moving average of A over n periods (decay 2/(n+1)).
  DECAYLINEAR(A, d) linearly weighted moving average of A over d periods.

Mathematical operations — one argument unless noted:
  PROD(A, n) product of A over the past n days. Use * for general multiplication.
  LOG(A) natural logarithm of A.
  SQRT(A) square root of A.
  POW(A, n) raise A to the power of n.
  SIGN(A) sign of A, one of 1, 0, or -1.
  EXP(A) exponential of A.
  ABS(A) absolute value of A.
  MAX(A, B) pairwise maximum of A and B.
  MIN(A, B) pairwise minimum of A and B.
  INV(A) reciprocal, one divided by A.
  FLOOR(A) floor of A.

Conditional and logical functions, which turn a condition into a number:
  (C) ? (A) : (B)  if condition C holds then A, otherwise B. C is a logical
    expression such as $close > $open.
  (C1) && (C2)  both C1 and C2 true.
  (C1) || (C2)  C1 or C2 true.
  COUNT(C, n) count of periods meeting condition C in the past n.
  SUMIF(A, n, C) sum of A over the past n periods where C holds.
  FILTER(A, C) keep A where condition C holds.

Regression and residual functions:
  SEQUENCE(n) a single-column sequence 1..n; only valid nested in REGBETA or
    REGRESI as argument B.
  REGBETA(A, B, n) regression slope of A on B over the past n samples.
  REGRESI(A, B, n) regression residual of A on B over the past n samples.

Technical indicators:
  RSI(A, n) relative strength index of A over n periods.
  MACD(A, short, long) difference of a short and a long EMA of A.
  BB_MIDDLE(A, n) middle Bollinger band (n-period SMA of A).
  BB_UPPER(A, n) middle band plus two standard deviations over n periods.
  BB_LOWER(A, n) middle band minus two standard deviations over n periods.

RULES FOR A LEGAL EXPRESSION:
  - Use only the six variables and the exact function names above.
  - Mind the TS_ prefix: TS_STD is rolling over time, STD is cross-sectional
    today — different functions. The same holds for every TS_ name.
  - Match every opening bracket with a closing one.
  - Every expression contains at least one variable.
  - Write every window explicitly. `TS_MEAN($close)` is rejected: the window is
    part of your decision, not a default someone else picks for you.
  - The data must reach the VALUE, not only the condition. `(C) ? (1) : (-1)`
    gives every stock one of two numbers and cannot rank them; gate a magnitude
    instead, e.g. `(C) ? (TS_ZSCORE($return, 5)) : (0)`.
```

## User Prompt

```text
The hypothesis and the candidate-function palette from
the Proposal and Design agents:
Observation: Small-cap stocks showing unusually high trading volume on a given day tend to underperform in the following days, suggesting a short-term reversal effect following volatile buying activity.

Driver: This phenomenon may stem from retail traders overreacting to news or sentiment, leading to excessive volume and subsequent mean reversion as prices correct. The signal comes from volume and return data, fitting a short-term reversal mechanism. The horizon is typically one to five days, and the comparison is cross-sectional among small-cap stocks, with a negative return direction.

HYPOTHESIS: Short-term reversal following abnormal volume spikes in small-cap stocks

**HYPOTHESIS VARIANTS:**
1. Short-term reversal following abnormal volume spikes in large-cap stocks  
2. Long-term underperformance following abnormal volume spikes in small-cap stocks  
3. Short-term reversal following abnormal volume dips in small-cap stocks  

**RECIPES:**

{
  "axis": "DEPTH",
  "sketch": "TS_ARGMAX of TS_ZSCORE($volume, 20) inside a rolling RANK of $return over 60 days",
  "uses": "$volume, $return; window sizes 20, 60"
}

{
  "axis": "UNUSUAL PAIRING",
  "sketch": "TS_COVARIANCE($open, $volume, 30) compared to TS_CORR($close, $return, 60)",
  "uses": "$open, $volume, $close, $return; window sizes 30, 60"
}

{
  "axis": "TIMESCALE CONTRAST",
  "sketch": "Difference between TS_STD($return, 5) and TS_STD($return, 60), scaled by $volume",
  "uses": "$return, $volume; window sizes 5, 60"
}

{
  "axis": "ODD NORMALISATION",
  "sketch": "RANK of TS_MAD($return, 20) divided by TS_RMS($volume, 60)",
  "uses": "$return, $volume; window sizes 20, 60"
}

{
  "axis": "ORDER STATISTICS",
  "sketch": "TS_ARGMIN of $volume over 30 days, then TS_ZSCORE of $return at that day",
  "uses": "$volume, $return; window size 30"
}

{
  "axis": "SHAPE, NOT LEVEL",
  "sketch": "KURT of $return over 60 days, subtracted by KURT of $volume over 30 days",
  "uses": "$return, $volume; window sizes 60, 30"
}

{
  "axis": "RESIDUAL",
  "sketch": "REGRESI($return, $volume, 60) residuals, then TS_ZSCORE of those",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "ASYMMETRY",
  "sketch": "DELTA($return, 1) multiplied by SIGN($volume - TS_MED($volume, 20))",
  "uses": "$return, $volume; window size 20"
}

{
  "axis": "CROSS-FAMILY PRODUCT",
  "sketch": "TS_MOMENT($return, 5) multiplied by TS_VWAP($volume, 30)",
  "uses": "$return, $volume; window sizes 5, 30"
}

{
  "axis": "INVERSION",
  "sketch": "One minus TS_PCTCHANGE($volume, 20) divided by TS_RANK($return, 60)",
  "uses": "$volume, $return; window sizes 20, 60"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "DECAYLINEAR of $volume over 30 days, then TS_SKEW of $return at that decayed value",
  "uses": "$volume, $return; window size 30"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "PROD of $open and $volume over 20 days, then TS_MAD of $return against that product",
  "uses": "$open, $volume, $return; window size 20"
}

{
  "axis": "TIMESCALE CONTRAST",
  "sketch": "TS_DIFF($return, 1) divided by TS_DIFF($volume, 5), with a window of 60",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "ORDER STATISTICS",
  "sketch": "LOWDAY of $volume over 30 days, then TS_PCTCHANGE of $return at that day",
  "uses": "$volume, $return; window size 30"
}

{
  "axis": "SHAPE, NOT LEVEL",
  "sketch": "TS_GINI($return, 60) subtracted by TS_GINI($volume, 30)",
  "uses": "$return, $volume; window sizes 60, 30"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "FILTER($return, $volume > TS_MED($volume, 20), 60)",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "CROSS-FAMILY PRODUCT",
  "sketch": "TS_RSI($return, 14) multiplied by TS_BB_UPPER($volume, 20)",
  "uses": "$return, $volume; window sizes 14, 20"
}

{
  "axis": "INVERSION",
  "sketch": "One minus TS_SKEW($return, 60) divided by TS_KURT($volume, 30)",
  "uses": "$return, $volume; window sizes 60, 30"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "FLOOR(TS_PCTCHANGE($volume, 5)) * SQRT(TS_VWAP($return, 30))",
  "uses": "$volume, $return; window sizes 5, 30"
}

{
  "axis": "DEPTH",
  "sketch": "TS_ARGMAX of $volume inside a rolling TS_MAD of $return over 60 days, then TS_ZSCORE of that",
  "uses": "$volume, $return; window size 60"
}

{
  "axis": "ODD NORMALISATION",
  "sketch": "TS_RANK($volume, 30) divided by TS_RANK($return, 60)",
  "uses": "$volume, $return; window sizes 30, 60"
}

{
  "axis": "UNUSUAL PAIRING",
  "sketch": "TS_COVARIANCE($open, $volume, 20) minus TS_CORR($close, $return, 60)",
  "uses": "$open, $volume, $close, $return; window sizes 20, 60"
}

{
  "axis": "TIMESCALE CONTRAST",
  "sketch": "TS_DIFF($return, 1) / TS_DIFF($volume, 5), windowed over 60",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "ORDER STATISTICS",
  "sketch": "TS_ARGMAX of $volume over 30 days, then TS_PCTCHANGE of $return at that day",
  "uses": "$volume, $return; window size 30"
}

{
  "axis": "SHAPE, NOT LEVEL",
  "sketch": "TS_SKEW($return, 60) subtracted by TS_SKEW($volume, 30)",
  "uses": "$return, $volume; window sizes 60, 30"
}

{
  "axis": "RESIDUAL",
  "sketch": "REGRESI($return, $volume, 60) residuals, then TS_MAD of $return against those",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "ASYMMETRY",
  "sketch": "DELTA($return, 1) multiplied by SIGN($volume - TS_MED($volume, 20))",
  "uses": "$return, $volume; window size 20"
}

{
  "axis": "CROSS-FAMILY PRODUCT",
  "sketch": "TS_MOMENT($return, 5) multiplied by TS_VWAP($volume, 30)",
  "uses": "$return, $volume; window sizes 5, 30"
}

{
  "axis": "INVERSION",
  "sketch": "One minus TS_PCTCHANGE($volume, 20) divided by TS_RANK($return, 60)",
  "uses": "$volume, $return; window sizes 20, 60"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "DECAYLINEAR of $volume over 30 days, then TS_SKEW of $return at that decayed value",
  "uses": "$volume, $return; window size 30"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "PROD of $open and $volume over 20 days, then TS_MAD of $return against that product",
  "uses": "$open, $volume, $return; window size 20"
}

{
  "axis": "TIMESCALE CONTRAST",
  "sketch": "TS_DIFF($return, 1) divided by TS_DIFF($volume, 5), with a window of 60",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "ORDER STATISTICS",
  "sketch": "LOWDAY of $volume over 30 days, then TS_PCTCHANGE of $return at that day",
  "uses": "$volume, $return; window size 30"
}

{
  "axis": "SHAPE, NOT LEVEL",
  "sketch": "TS_GINI($return, 60) subtracted by TS_GINI($volume, 30)",
  "uses": "$return, $volume; window sizes 60, 30"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "FILTER($return, $volume > TS_MED($volume, 20), 60)",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "CROSS-FAMILY PRODUCT",
  "sketch": "TS_RSI($return, 14) multiplied by TS_BB_UPPER($volume, 20)",
  "uses": "$return, $volume; window sizes 14, 20"
}

{
  "axis": "INVERSION",
  "sketch": "One minus TS_SKEW($return, 60) divided by TS_KURT($volume, 30)",
  "uses": "$return, $volume; window sizes 60, 30"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "FLOOR(TS_PCTCHANGE($volume, 5)) * SQRT(TS_VWAP($return, 30))",
  "uses": "$volume, $return; window sizes 5, 30"
}

{
  "axis": "DEPTH",
  "sketch": "TS_ARGMAX of $volume inside a rolling TS_MAD of $return over 60 days, then TS_ZSCORE of that",
  "uses": "$volume, $return; window size 60"
}

{
  "axis": "ODD NORMALISATION",
  "sketch": "TS_RANK($volume, 30) divided by TS_RANK($return, 60)",
  "uses": "$volume, $return; window sizes 30, 60"
}

{
  "axis": "UNUSUAL PAIRING",
  "sketch": "TS_COVARIANCE($open, $volume, 20) minus TS_CORR($close, $return, 60)",
  "uses": "$open, $volume, $close, $return; window sizes 20, 60"
}

{
  "axis": "TIMESCALE CONTRAST",
  "sketch": "TS_DIFF($return, 1) / TS_DIFF($volume, 5), windowed over 60",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "ORDER STATISTICS",
  "sketch": "TS_ARGMAX of $volume over 30 days, then TS_PCTCHANGE of $return at that day",
  "uses": "$volume, $return; window size 30"
}

{
  "axis": "SHAPE, NOT LEVEL",
  "sketch": "TS_SKEW($return, 60) subtracted by TS_SKEW($volume, 30)",
  "uses": "$return, $volume; window sizes 60, 30"
}

{
  "axis": "RESIDUAL",
  "sketch": "REGRESI($return, $volume, 60) residuals, then TS_MAD of $return against those",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "ASYMMETRY",
  "sketch": "DELTA($return, 1) multiplied by SIGN($volume - TS_MED($volume, 20))",
  "uses": "$return, $volume; window size 20"
}

{
  "axis": "CROSS-FAMILY PRODUCT",
  "sketch": "TS_MOMENT($return, 5) multiplied by TS_VWAP($volume, 30)",
  "uses": "$return, $volume; window sizes 5, 30"
}

{
  "axis": "INVERSION",
  "sketch": "One minus TS_PCTCHANGE($volume, 20) divided by TS_RANK($return, 60)",
  "uses": "$volume, $return; window sizes 20, 60"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "DECAYLINEAR of $volume over 30 days, then TS_SKEW of $return at that decayed value",
  "uses": "$volume, $return; window size 30"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "PROD of $open and $volume over 20 days, then TS_MAD of $return against that product",
  "uses": "$open, $volume, $return; window size 20"
}

{
  "axis": "TIMESCALE CONTRAST",
  "sketch": "TS_DIFF($return, 1) divided by TS_DIFF($volume, 5), with a window of 60",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "ORDER STATISTICS",
  "sketch": "LOWDAY of $volume over 30 days, then TS_PCTCHANGE of $return at that day",
  "uses": "$volume, $return; window size 30"
}

{
  "axis": "SHAPE, NOT LEVEL",
  "sketch": "TS_GINI($return, 60) subtracted by TS_GINI($volume, 30)",
  "uses": "$return, $volume; window sizes 60, 30"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "FILTER($return, $volume > TS_MED($volume, 20), 60)",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "CROSS-FAMILY PRODUCT",
  "sketch": "TS_RSI($return, 14) multiplied by TS_BB_UPPER($volume, 20)",
  "uses": "$return, $volume; window sizes 14, 20"
}

{
  "axis": "INVERSION",
  "sketch": "One minus TS_SKEW($return, 60) divided by TS_KURT($volume, 30)",
  "uses": "$return, $volume; window sizes 60, 30"
}

{
  "axis": "NEGLECTED OPERATOR",
  "sketch": "FLOOR(TS_PCTCHANGE($volume, 5)) * SQRT(TS_VWAP($return, 30))",
  "uses": "$volume, $return; window sizes 5, 30"
}

{
  "axis": "DEPTH",
  "sketch": "TS_ARGMAX of $volume inside a rolling TS_MAD of $return over 60 days, then TS_ZSCORE of that",
  "uses": "$volume, $return; window size 60"
}

{
  "axis": "ODD NORMALISATION",
  "sketch": "TS_RANK($volume, 30) divided by TS_RANK($return, 60)",
  "uses": "$volume, $return; window sizes 30, 60"
}

{
  "axis": "UNUSUAL PAIRING",
  "sketch": "TS_COVARIANCE($open, $volume, 20) minus TS_CORR($close, $return, 60)",
  "uses": "$open, $volume, $close, $return; window sizes 20, 60"
}

{
  "axis": "TIMESCALE CONTRAST",
  "sketch": "TS_DIFF($return, 1) / TS_DIFF($volume, 5), windowed over 60",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "ORDER STATISTICS",
  "sketch": "TS_ARGMAX of $volume over 30 days, then TS_PCTCHANGE of $return at that day",
  "uses": "$volume, $return; window size 30"
}

{
  "axis": "SHAPE, NOT LEVEL",
  "sketch": "TS_SKEW($return, 60) subtracted by TS_SKEW($volume, 30)",
  "uses": "$return, $volume; window sizes 60, 30"
}

{
  "axis": "RESIDUAL",
  "sketch": "REGRESI($return, $volume, 60) residuals, then TS_MAD of $return against those",
  "uses": "$return, $volume; window size 60"
}

{
  "axis": "ASYMMETRY",
  "sketch": "DELTA($return, 1) multiplied by SIGN($volume - TS_MED($volume, 20))",
  "uses": "$return, $volume; window size 20"
}

{
  "axis": "CROSS-FAMILY PRODUCT",
  "sketch": "TS_MOMENT($return, 5) multiplied by TS_VWAP($volume, 30)",
  "uses": "$return, $volume; window sizes 5, 30"
}

{
  "

Write 3 to 6 strong factor expressions that all express the SAME hypothesis
mechanism, using the palette as INGREDIENTS. Explore different CONSTRUCTIONS of
that one mechanism — never a different mechanism. A deterministic regulator
rejects every illegal or trivial expression, and each rejection LOWERS your score.

HOW TO SPEND YOUR FACTORS. Realise DIFFERENT
recipes from the Explorer — one factor per recipe wherever you can, rather than one
recipe rewritten several ways. Across your set:
  - use at least three DIFFERENT outermost wrappers (not RANK every time);
  - include at least one expression that nests three levels deep;
  - include at least one expression built on a function this team rarely uses;
  - two factors that differ only by a window size count as ONE factor, so do not
    spend two slots that way.
An expression that looks unfamiliar is a success here, not a mistake, provided it
is legal and ranks stocks.

For each expression:
  Step 1 — Name the Explorer recipe (its axis) that you are realising.
  Step 2 — Build in layers: inner signal -> cross-sectional step so the result
           ranks stocks. The regulator requires at least two layers.
  Step 3 — Across the full set, check the four coverage requirements above and
           fix the set if any of them fails.
  Step 4 — Check: only the six variables; argument counts exact; every window
           written explicitly and between 2 and 60; pair distinct series in
           two-argument functions; brackets balanced; the data reaches the VALUE
           and not only the condition; result is a rankable number.

  Step 5 — Short name and one sentence stating the INTENT of the expression:
           what it is meant to compute and which structural axis it
           explores. Write it so a Repair agent could reconstruct your intent and
           preserve it even if the raw expression turned out illegal.
Then, as the LAST thing you write, output this JSON block as the final content,
with nothing surrounding it and no trailing commas. The "hypothesis" field MUST
be the hypothesis copied verbatim. The "explanation" MUST state the intent (the
computation you meant + the mechanism part), not just a label — the Repair agent
relies on it:
{
  "hypothesis": "<the hypothesis from latent memory, verbatim, in one sentence>",
  "factors": [
    {
      "name": "<short factor name>",
      "expression": "<single DSL expression>",
      "explanation": "<the intent: what this expression is meant to compute and which part of the mechanism it serves>"
    }
  ]
}
The factors array holds 3 to 6 factors that are
structurally distinct from one another. They do NOT have to agree with the original
hypothesis.
```

## Response

```text
{
  "hypothesis": "Short-term reversal following abnormal volume spikes in small-cap stocks",
  "factors": [
    {
      "name": "VolumeReversalDepth",
      "expression": "TS_ZSCORE(TS_ARGMAX($volume, 30), 60)",
      "explanation": "This factor explores the DEPTH axis by finding the day with the highest volume over 30 days and then normalizing its impact on returns over 60 days. It aims to capture the reversal effect after a spike in volume."
    },
    {
      "name": "VolumeResidualShape",
      "expression": "TS_SKEW(REGRESI($return, $volume, 60), 30)",
      "explanation": "This factor explores the RESIDUAL axis by regressing returns against volume and analyzing the skewness of the residuals over 30 days. It captures the shape of the residual distribution, suggesting potential asymmetry in the reversal mechanism."
    },
    {
      "name": "VolumeAsymmetrySignal",
      "expression": "DELTA($return, 1) * SIGN($volume - TS_MED($volume, 20))",
      "explanation": "This factor explores the ASYMMETRY axis by measuring the change in returns and multiplying it by the sign of the deviation of volume from its 20-day median. It highlights asymmetric responses to volume changes."
    },
    {
      "name": "VolumeOrderStatReturn",
      "expression": "TS_PCTCHANGE($return, TS_ARGMAX($volume, 30))",
      "explanation": "This factor explores the ORDER STATISTICS axis by identifying the day with the highest volume over 30 days and then measuring the percentage change in returns on that specific day. It focuses on the timing of volume spikes and their immediate impact."
    },
    {
      "name": "VolumeTimescaleContrast",
      "expression": "(TS_DIFF($return, 1) / TS_DIFF($volume, 5)) * TS_ZSCORE($return, 60)",
      "explanation": "This factor explores the TIMESCALE CONTRAST axis by comparing the daily change in returns to the 5-day change in volume, normalized by the cross-sectional z-score of returns. It emphasizes the contrast in time scales and their influence on reversal patterns."
    },
    {
      "name": "VolumeOddNormalization",
      "expression": "TS_RANK($volume, 30) / TS_RANK($return, 60)",
      "explanation": "This factor explores the ODD NORMALISATION axis by ranking volume over 30 days and returns over 60 days separately, then dividing them. It provides a non-standard normalization approach to highlight unusual volume-return relationships."
    }
  ]
}
```
