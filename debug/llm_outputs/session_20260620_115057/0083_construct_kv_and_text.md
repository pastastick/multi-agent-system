# Call 0083 — `construct` (kv_and_text)

## Meta

- ts: 2026-06-20 13:39:55
- conv_id: `9978864a`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 4935
- output_tokens: 3930
- duration_s: 146.3017
- text_len: 14489

## System Prompt

```text
You are the Builder in a quantitative alpha-mining team. A market hypothesis and
a candidate-function palette reach you either through shared latent memory or as
text in the task below. Your task is to write the finished factor expressions and
output them.

PIPELINE CONTEXT. Proposal set the mechanism; the Design agent shortlisted the
functions and variables. You make the final decision and write the expressions.
You hold FINAL responsibility for symbolic validity: a deterministic regulator
checks every expression and rejects any that is illegal, so spend your care on
correct names, correct argument counts, balanced brackets, and faithfulness to
the hypothesis. Do not count characters or features.

WHAT YOU MUST PRODUCE. Each factor is a CONTINUOUS cross-sectional score: one
real number per stock per day, where higher means more expected out-performance.
It must NOT be a true-or-false mask. A bare condition such as "close greater than
open and volume above its average" returns true or false, which cannot be ranked;
to use a condition, gate a magnitude with it — take the signal when the condition
holds and zero otherwise, or subtract one cross-sectional rank from another.

WORDS USED IN THIS TASK (one meaning each, so nothing is ambiguous):
  - VARIABLE = one of the six data columns. The only data leaves ($open $high $low $close $volume $return).
  - FUNCTION = a named operation from the library below, written as NAME(...).
  - OPERATOR = an arithmetic or logical symbol: plus, minus, times, divide, and,
    or, and the conditional question-mark / colon pair.
  - EXPRESSION = one complete formula built from variables, functions, and
    operators that produces one number per stock per day.
  - FACTOR = a named signal: a short name, a one-line description, and the single
    expression that computes it.

VARIABLES (the only data leaves). Case does not matter — $close and $CLOSE read
the same: $open  $high  $low  $close  $volume  $return

ARITHMETIC OPERATORS, written out so there is no doubt. Inside an expression use
the symbol + to add, - to subtract, * to multiply, and / to divide. Use these
symbols ONLY inside an expression; in your prose write the word ("add", "divide",
"and", "or") so a symbol always means arithmetic and nothing else.

FUNCTIONS — each is written NAME(arguments). Argument letters: A and B are any
sub-expression (a variable, a number, or another function call); C is a
condition that is true or false, such as "$close is greater than $open"; n and p
are whole-number windows of days; q is a fraction between 0 and 1.

ARITY IS STRICT — each function takes exactly the arguments shown. CROSS-SECTIONAL
functions take exactly ONE argument with no window. RANK(A) takes one argument;
for a rolling rank over n days, use TS_RANK(A, n) instead. Always match argument
count to the signature above.

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
```

## User Prompt

```text
The hypothesis and the candidate-function palette from
the Proposal and Design agents:
<hypothesis>the market shows a persistent trend in liquid equities over 20-day windows, with stocks that have shown strong price increases in the past tend to continue outperforming. this suggests a long-horizon momentum effect in trending regimes.</hypothesis>

<func>TS_PCTCHANGE($close, 20): variables $close, window 20 days, purpose captures percentage change in close price over 20 days</func>
<func>TS_MEAN($close, 20): variables $close, window 20 days, purpose captures average close price over 20 days</func>
<func>TS_STD($close, 20): variables $close, window 20 days, purpose captures standard deviation of close price over 20 days</func>
<func>TS_RANK($close, 20): variables $close, window 20 days, purpose captures rank of close price over past 20 days</func>
<func>TS_ZSCORE($close, 20): variables $close, window 20 days, purpose captures z-score of close price over past 20 days</func>
<func>RSI($close, 20): variables $close, window 20 days, purpose captures relative strength index over 20 days</func>
<func>MACD($close, 12, 26): variables $close, window 12 and 26 days, purpose captures moving average convergence divergence</func>
<func>EMA($close, 20): variables $close, window 20 days, purpose captures exponential moving average of close price</func>
<func>WMA($close, 20): variables $close, window 20 days, purpose captures weighted moving average of close price</func>
<func>SMA($close, 20, 1): variables $close, window 20 days, purpose captures simple moving average of close price</func>
<func>DELTA($close, 20): variables $close, window 20 days, purpose captures change in close price over 20 days</func>
<func>TS_CORR($close, $return, 20): variables $close and $return, window 20 days, purpose captures correlation between close price and return over 20 days</func>
<func>TS_COVARIANCE($close, $return, 20): variables $close and $return, window 20 days, purpose captures covariance between close price and return over 20 days</func>
<func>TS_REGRESS($close, $return, 20): variables $close and $return, window 20 days, purpose captures regression of close price on return over 20 days</func>
<func>TS_PCTCHANGE($return, 20): variables $return, window 20 days, purpose captures percentage change in return over 20 days</func>
<func>TS_MEAN($return, 20): variables $return, window 20 days, purpose captures average return over 20 days</func>
<func>TS_RANK($return, 20): variables $return, window 20 days, purpose captures rank of return over past 20 days</func>
<func>TS_ZSCORE($return, 20): variables $return, window 20 days, purpose captures z-score of return over past 20 days</func>
<func>TS_MIN($close, 20): variables $close, window 20 days, purpose captures minimum close price over past 20 days</func>
<func>TS_MAX($close, 20): variables $close, window 20 days, purpose captures maximum close price over past 20 days</func>
<func>TS_ARGMAX($close, 20): variables $close, window 20 days, purpose captures index of maximum close price over past 20 days</func>
<func>TS_ARGMIN($close, 20): variables $close, window 20 days, purpose captures index of minimum close price over past 20 days</func>
<func>TS_QUANTILE($close, 0.5, 20): variables $close, window 20 days, purpose captures median of close price over past 20 days</func>
<func>TS_VAR($close, 20): variables $close, window 20 days, purpose captures variance of close price over past 20 days</func>
<func>TS_MAD($close, 20): variables $close, window 20 days, purpose captures rolling median absolute deviation of close price over past 20 days</func>
<func>TS_CORR($close, $high, 20): variables $close and $high, window 20 days, purpose captures correlation between close price and high price over 20 days</func>
<func>TS_COVARIANCE($close, $high, 20): variables $close and $high, window 20 days, purpose captures covariance between close price and high price over 20 days</func>
<func>TS_REGRESS($close, $high, 20): variables $close and $high, window 20 days, purpose captures regression of close price on high price over 20 days</func>
<func>TS_PCTCHANGE($high, 20): variables $high, window 20 days, purpose captures percentage change in high price over 20 days</func>
<func>TS_MEAN($high, 20): variables $high, window 20 days, purpose captures average high price over 20 days</func>
<func>TS_RANK($high, 20): variables $high, window 20 days, purpose captures rank of high price over past 20 days</func>
<func>TS_ZSCORE($high, 20): variables $high, window 20 days, purpose captures z-score of high price over past 20 days</func>
<func>TS_MIN($high, 20): variables $high, window 20 days, purpose captures minimum high price over past 20 days</func>
<func>TS_MAX($high, 20): variables $high, window 20 days, purpose captures maximum high price over past 20 days</func>
<func>TS_ARGMAX($high, 20): variables $high, window 20 days, purpose captures index of maximum high price over past 20 days</func>
<func>TS_ARGMIN($high, 20): variables $high, window 20 days, purpose captures index of minimum high price over past 20 days</func>
<func>TS_QUANTILE($high, 0.5, 20): variables $high, window 20 days, purpose captures median of high price over past 20 days</func>
<func>TS_VAR($high, 20): variables $high, window 20 days, purpose captures variance of high price over past 20 days</func>
<func>TS_MAD($high, 20): variables $high, window 20 days, purpose captures rolling median absolute deviation of high price over past 20 days</func>
<func>TS_CORR($high, $low, 20): variables $high and $low, window 20 days, purpose captures correlation between high price and low price over 20 days</func>
<func>TS_COVARIANCE($high, $low, 20): variables $high and $low, window 20 days, purpose captures covariance between high price and low price over 20 days</func>
<func>TS_REGRESS($high, $low, 20): variables $high and $low, window 20 days, purpose captures regression of high price on low price over 20 days</func>
<func>TS_PCTCHANGE($low, 20): variables $low, window 20 days, purpose captures percentage change in low price over 20 days</func>
<func>TS_MEAN($low, 20): variables $low, window 20 days, purpose captures average low price over 20 days</func>
<func>TS_RANK($low, 20): variables $low, window 20 days, purpose captures rank of low price over past 20 days</func>
<func>TS_ZSCORE($low, 20): variables $low, window 20 days, purpose captures z-score of low price over past 20 days</func>
<func>TS_MIN($low, 20): variables $low, window 20 days, purpose captures minimum low price over past 20 days</func>
<func>TS_MAX($low, 20): variables $low, window 20 days, purpose captures maximum low price over past 20 days</func>
<func>TS_ARGMAX($low, 20): variables $low, window 20 days, purpose captures index of maximum low price over past 20 days</func>
<func>TS_ARGMIN($low, 20): variables $low, window 20 days, purpose captures index of minimum low price over past 20 days</func>
<func>TS_QUANTILE($low, 0.5, 20): variables $low, window 20 days, purpose captures median of low price over past 20 days</func>
<func>TS_VAR($low, 20): variables $low, window 20 days, purpose captures variance of low price over past 20 days</func>
<func>TS_MAD($low, 20): variables $low, window 20 days, purpose captures rolling median absolute deviation of low price over past 20 days</func>
<func>TS_CORR($low, $return, 20): variables $low and $return, window 20 days, purpose captures correlation between low price and return over 20 days</func>
<func>TS_COVARIANCE($low, $return, 20): variables $low and $return, window 20 days, purpose captures covariance between low price and return over 20 days</func>
<func>TS_REGRESS($low, $return, 20): variables $low and $return, window 20 days, purpose captures regression of low price on return over 20 days</func>
<func>TS_PCTCHANGE($volume, 20): variables $volume, window 20 days, purpose captures percentage change in volume over 20 days</func>
<func>TS_MEAN($volume, 20): variables $volume, window 20 days, purpose captures average volume over 20 days</func>
<func>TS_RANK($volume, 20): variables $volume, window 20 days, purpose captures rank of volume over past 20 days</func>
<func>TS_ZSCORE($volume, 20): variables $volume, window 20 days, purpose captures z-score of volume over past 20 days</func>
<func>TS_MIN($volume, 20): variables $volume, window 20 days, purpose captures minimum volume over past 20 days</func>
<func>TS_MAX($volume, 20): variables $volume, window 20 days, purpose captures maximum volume over past 20 days</func>
<func>TS_ARGMAX($volume, 20): variables $volume, window 20 days, purpose captures index of maximum volume over past 20 days</func>
<func>TS_ARGMIN($volume, 20): variables $volume, window 20 days, purpose captures index of minimum volume over past 20 days</func>
<func>TS_QUANTILE($volume, 0.5, 20): variables $volume, window 20 days, purpose captures median of volume over past 20 days</func>
<func>TS_VAR($volume, 20): variables $volume, window 20 days, purpose captures variance of volume over past 20 days</func>
<func>TS_MAD($volume, 20): variables $volume, window 20 days, purpose captures rolling median absolute deviation of volume over past 20 days</func>

Your task is to write as many valid factor expressions as the palette supports —
each one a different realisation of the same hypothesis. Many factors are better
than few: some will be rejected by the regulator after you write them, so the
more you produce, the more survive.

For each candidate function in the palette:
  Step 1 — Take the candidate function and the variables it uses from the palette.
  Step 2 — Build the expression in layers: the candidate function as the inner
           computation, then a cross-sectional step (rank, standardise, gate a
           magnitude on a condition, or contrast two ranks) that turns it into a
           continuous score.
  Step 3 — Check it: only real variables from the six listed; correct argument
           counts (time-series functions take a window, cross-sectional functions
           take no window); windows between one and sixty days; no series paired
           with itself; brackets balanced; the result a rankable number.
  Step 4 — Give it a short descriptive name and one sentence on what it captures.
  Step 5 — Repeat Steps 1 to 4 for the next candidate function in the palette.
Then, as the LAST thing you write, output this JSON block exactly — no text
before or after it, no markdown fences, no trailing commas:
{
  "hypothesis": "<the hypothesis from your latent memory, in one sentence>",
  "factors": [
    {
      "name": "<short factor name>",
      "expression": "<single DSL expression>",
      "explanation": "<one sentence on what it measures>"
    }
  ]
}
The factors array must contain one object for every candidate function worked
through in Steps 1 to 5. Include all of them.
```

## Response

```text
{
  "hypothesis": "the market shows a persistent trend in liquid equities over 20-day windows, with stocks that have shown strong price increases in the past tend to continue outperforming. this suggests a long-horizon momentum effect in trending regimes.",
  "factors": [
    {
      "name": "ClosePctChange20",
      "expression": "TS_PCTCHANGE($close, 20)",
      "explanation": "Captures the percentage change in close price over the past 20 days, indicating recent price movement."
    },
    {
      "name": "CloseMean20",
      "expression": "TS_MEAN($close, 20)",
      "explanation": "Captures the average close price over the past 20 days, reflecting recent price levels."
    },
    {
      "name": "CloseStd20",
      "expression": "TS_STD($close, 20)",
      "explanation": "Captures the standard deviation of close price over the past 20 days, indicating volatility."
    },
    {
      "name": "CloseRank20",
      "expression": "TS_RANK($close, 20)",
      "explanation": "Captures the rank of close price over the past 20 days, indicating relative strength."
    },
    {
      "name": "CloseZScore20",
      "expression": "TS_ZSCORE($close, 20)",
      "explanation": "Captures the z-score of close price over the past 20 days, standardizing recent performance."
    },
    {
      "name": "RSI20",
      "expression": "RSI($close, 20)",
      "explanation": "Captures the relative strength index of close price over the past 20 days, indicating momentum."
    },
    {
      "name": "MACD20",
      "expression": "MACD($close, 12, 26)",
      "explanation": "Captures the moving average convergence divergence of close price, indicating trend strength."
    },
    {
      "name": "EMA20",
      "expression": "EMA($close, 20)",
      "explanation": "Captures the exponential moving average of close price, emphasizing recent price changes."
    },
    {
      "name": "WMA20",
      "expression": "WMA($close, 20)",
      "explanation": "Captures the weighted moving average of close price, giving more weight to recent prices."
    },
    {
      "name": "SMA20",
      "expression": "SMA($close, 20, 1)",
      "explanation": "Captures the simple moving average of close price, smoothing out short-term fluctuations."
    },
    {
      "name": "Delta20",
      "expression": "DELTA($close, 20)",
      "explanation": "Captures the change in close price over the past 20 days, indicating recent price trends."
    },
    {
      "name": "CorrCloseReturn20",
      "expression": "TS_CORR($close, $return, 20)",
      "explanation": "Captures the correlation between close price and return over the past 20 days, indicating relationship between price and performance."
    },
    {
      "name": "CovCloseReturn20",
      "expression": "TS_COVARIANCE($close, $return, 20)",
      "explanation": "Captures the covariance between close price and return over the past 20 days, indicating how they move together."
    },
    {
      "name": "RegressCloseReturn20",
      "expression": "TS_REGRESS($close, $return, 20)",
      "explanation": "Captures the regression of close price on return over the past 20 days, indicating predictive power of return on price."
    },
    {
      "name": "ReturnPctChange20",
      "expression": "TS_PCTCHANGE($return, 20)",
      "explanation": "Captures the percentage change in return over the past 20 days, indicating recent performance."
    },
    {
      "name": "ReturnMean20",
      "expression": "TS_MEAN($return, 20)",
      "explanation": "Captures the average return over the past 20 days, reflecting recent performance levels."
    },
    {
      "name": "ReturnRank20",
      "expression": "TS_RANK($return, 20)",
      "explanation": "Captures the rank of return over the past 20 days, indicating relative performance."
    },
    {
      "name": "ReturnZScore20",
      "expression": "TS_ZSCORE($return, 20)",
      "explanation": "Captures the z-score of return over the past 20 days, standardizing recent performance."
    },
    {
      "name": "CloseMin20",
      "expression": "TS_MIN($close, 20)",
      "explanation": "Captures the minimum close price over the past 20 days, indicating recent low points."
    },
    {
      "name": "CloseMax20",
      "expression": "TS_MAX($close, 20)",
      "explanation": "Captures the maximum close price over the past 20 days, indicating recent high points."
    },
    {
      "name": "CloseArgMax20",
      "expression": "TS_ARGMAX($close, 20)",
      "explanation": "Captures the index of the maximum close price over the past 20 days, indicating the most recent peak."
    },
    {
      "name": "CloseArgMin20",
      "expression": "TS_ARGMIN($close, 20)",
      "explanation": "Captures the index of the minimum close price over the past 20 days, indicating the most recent trough."
    },
    {
      "name": "CloseQuantile20",
      "expression": "TS_QUANTILE($close, 0.5, 20)",
      "explanation": "Captures the median of close price over the past 20 days, indicating central tendency."
    },
    {
      "name": "CloseVar20",
      "expression": "TS_VAR($close, 20)",
      "explanation": "Captures the variance of close price over the past 20 days, indicating dispersion of prices."
    },
    {
      "name": "CloseMAD20",
      "expression": "TS_MAD($close, 20)",
      "explanation": "Captures the rolling median absolute deviation of close price over the past 20 days, indicating typical deviation from the median."
    },
    {
      "name": "CorrCloseHigh20",
      "expression": "TS_CORR($close, $high, 20)",
      "explanation": "Captures the correlation between close price and high price over the past 20 days, indicating relationship between price extremes."
    },
    {
      "name": "CovCloseHigh20",
      "expression": "TS_COVARIANCE($close, $high, 20)",
      "explanation": "Captures the covariance between close price and high price over the past 20 days, indicating how they move together."
    },
    {
      "name": "RegressCloseHigh20",
      "expression": "TS_REGRESS($close, $high, 20)",
      "explanation": "Captures the regression of close price on high price over the past 20 days, indicating predictive power of high price on close price."
    },
    {
      "name": "HighPctChange20",
      "expression": "TS_PCTCHANGE($high, 20)",
      "explanation": "Captures the percentage change in high price over the past 20 days, indicating recent price movement."
    },
    {
      "name": "HighMean20",
      "expression": "TS_MEAN($high, 20)",
      "explanation": "Captures the average high price over the past 20 days, reflecting recent price levels."
    },
    {
      "name": "HighRank20",
      "expression": "TS_RANK($high, 20)",
      "explanation": "Captures the rank of high price over the past 20 days, indicating relative strength."
    },
    {
      "name": "HighZScore20",
      "expression": "TS_ZSCORE($high, 20)",
      "explanation": "Captures the z-score of high price over the past 20 days, standardizing recent performance."
    },
    {
      "name": "HighMin20",
      "expression": "TS_MIN($high, 20)",
      "explanation": "Captures the minimum high price over the past 20 days, indicating recent low points."
    },
    {
      "name": "HighMax20",
      "expression": "TS_MAX($high, 20)",
      "explanation": "Captures the maximum high price over the past 20 days, indicating recent high points."
    },
    {
      "name": "HighArgMax20",
      "expression": "TS_ARGMAX($high, 20)",
      "explanation": "Captures the index of the maximum high price over the past 20 days, indicating the most recent peak."
    },
    {
      "name": "HighArgMin20",
      "expression": "TS_ARGMIN($high, 20)",
      "explanation": "Captures the index of the minimum high price over the past 20 days, indicating the most recent trough."
    },
    {
      "name": "HighQuantile20",
      "expression": "TS_QUANTILE($high, 0.5, 20)",
      "explanation": "Captures the median of high price over the past 20 days, indicating central tendency."
    },
    {
      "name": "HighVar20",
      "expression": "TS_VAR($high, 20)",
      "explanation": "Captures the variance of high price over the past 20 days, indicating dispersion of prices."
    },
    {
      "name": "HighMAD20",
      "expression": "TS_MAD($high, 20)",
      "explanation": "Captures the rolling median absolute deviation of high price over the past 20 days, indicating typical deviation from the median."
    },
    {
      "name": "CorrHighLow20",
      "expression": "TS_CORR($high, $low, 20)",
      "explanation": "Captures the correlation between high price and low price over the past 20 days, indicating relationship between price extremes."
    },
    {
      "name": "CovHighLow20",
      "expression": "TS_COVARIANCE($high, $low, 20)",
      "explanation": "Captures the covariance between high price and low price over the past 20 days, indicating how they move together."
    },
    {
      "name": "RegressHighLow20",
      "expression": "TS_REGRESS($high, $low, 20)",
      "explanation": "Captures the regression of high price on low price over the past 20 days, indicating predictive power of low price on high price."
    },
    {
      "name": "LowPctChange20",
      "expression": "TS_PCTCHANGE($low, 20)",
      "explanation": "Captures the percentage change in low price over the past 20 days, indicating recent price movement."
    },
    {
      "name": "LowMean20",
      "expression": "TS_MEAN($low, 20)",
      "explanation": "Captures the average low price over the past 20 days, reflecting recent price levels."
    },
    {
      "name": "LowRank20",
      "expression": "TS_RANK($low, 20)",
      "explanation": "Captures the rank of low price over the past 20 days, indicating relative strength."
    },
    {
      "name": "LowZScore20",
      "expression": "TS_ZSCORE($low, 20)",
      "explanation": "Captures the z-score of low price over the past 20 days, standardizing recent performance."
    },
    {
      "name": "LowMin20",
      "expression": "TS_MIN($low, 20)",
      "explanation": "Captures the minimum low price over the past 20 days, indicating recent low points."
    },
    {
      "name": "LowMax20",
      "expression": "TS_MAX($low, 20)",
      "explanation": "Captures the maximum low price over the past 20 days, indicating recent high points."
    },
    {
      "name": "LowArgMax20",
      "expression": "TS_ARGMAX($low, 20)",
      "explanation": "Captures the index of the maximum low price over the past 20 days, indicating the most recent peak."
    },
    {
      "name": "LowArgMin20",
      "expression": "TS_ARGMIN($low, 20)",
      "explanation": "Captures the index of the minimum low price over the past 20 days, indicating the most recent trough."
    },
    {
      "name": "LowQuantile20",
      "expression": "TS_QUANTILE($low, 0.5, 20)",
      "explanation": "Captures the median of low price over the past 20 days, indicating central tendency."
    },
    {
      "name": "LowVar20",
      "expression": "TS_VAR($low, 20)",
      "explanation": "Captures the variance of low price over the past 20 days, indicating dispersion of prices."
    },
    {
      "name": "LowMAD20",
      "expression": "TS_MAD($low, 20)",
      "explanation": "Captures the rolling median absolute deviation of low price over the past 20 days, indicating typical deviation from the median."
    },
    {
      "name": "CorrLowReturn20",
      "expression": "TS_CORR($low, $return, 20)",
      "explanation": "Captures the correlation between low price and return over the past 20 days, indicating relationship between price and performance."
    },
    {
      "name": "CovLowReturn20",
      "expression": "TS_COVARIANCE($low, $return, 20)",
      "explanation": "Captures the covariance between low price and return over the past 20 days, indicating how they move together."
    },
    {
      "name": "RegressLowReturn20",
      "expression": "TS_REGRESS($low, $return, 20)",
      "explanation": "Captures the regression of low price on return over the past 20 days, indicating predictive power of return on price."
    },
    {
      "name": "VolumePctChange20",
      "expression": "TS_PCTCHANGE($volume, 20)",
      "explanation": "Captures the percentage change in volume over the past 20 days, indicating recent trading activity."
    },
    {
      "name": "VolumeMean20",
      "expression": "TS_MEAN($volume, 20)",
      "explanation": "Captures the average volume over the past 20 days, reflecting recent trading levels."
    },
    {
      "name": "VolumeRank20",
      "expression": "TS_RANK($volume, 20)",
      "explanation": "Captures the rank of volume over the past 20 days, indicating relative trading intensity."
    },
    {
      "name": "VolumeZScore20",
      "expression": "TS_ZSCORE($volume, 20)",
      "explanation": "Captures the z-score of volume over the past 20 days, standardizing recent trading activity."
    },
    {
      "name": "VolumeMin20",
      "expression": "TS_MIN($volume, 20)",
      "explanation": "Captures the minimum volume over the past 20 days, indicating recent low trading activity."
    },
    {
      "name": "VolumeMax20",
      "expression": "TS_MAX($volume, 20)",
      "explanation": "Captures the maximum volume over the past 20 days, indicating recent high trading activity."
    },
    {
      "name": "VolumeArgMax20",
      "expression": "TS_ARGMAX($volume, 20)",
      "explanation": "Captures the index of the maximum volume over the past 20 days, indicating the most recent peak in trading activity."
    },
    {
      "name": "VolumeArgMin20",
      "expression": "TS_ARGMIN($volume, 20)",
      "explanation": "Captures the index of the minimum volume over the past 20 days, indicating the most recent trough in trading activity."
    },
    {
      "name": "VolumeQuantile20",
      "expression": "TS_QUANTILE($volume, 0.5, 20)",
      "explanation": "Captures the median of volume over the past 20 days, indicating central tendency in trading activity."
    },
    {
      "name": "VolumeVar20",
      "expression": "TS_VAR($volume, 20)",
      "explanation": "Captures the variance of volume over the past 20 days, indicating dispersion in trading activity."
    },
    {
      "name": "VolumeMAD20",
      "expression": "TS_MAD($volume, 20)",
      "explanation": "Captures the rolling median absolute deviation of volume over the past 20 days, indicating typical deviation from the median in trading activity."
    }
  ]
}
```
