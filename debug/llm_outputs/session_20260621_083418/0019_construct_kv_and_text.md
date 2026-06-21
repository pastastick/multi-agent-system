# Call 0019 — `construct` (kv_and_text)

## Meta

- ts: 2026-06-21 08:46:32
- conv_id: `8e296143`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 4680
- output_tokens: 3223
- duration_s: 127.4
- text_len: 12639

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

BUILD IN LAYERS — DO NOT TRANSCRIBE THE PALETTE. A lone library call (a bare
moving average, a bare cross-sectional rank of one raw variable, a bare LOG or
SIGN) is not a factor; the regulator rejects it as trivial. Take the palette
function as the inner signal, then wrap it in a cross-sectional step (RANK or
ZSCORE across stocks, or gate a magnitude on a condition) so the result ranks
stocks. Copying the palette one-to-one into one expression each is a failure mode:
most such lines get rejected and the few that pass do not express the hypothesis.

FAITHFULNESS OVER COUNT. Every expression must be a realisation of THIS
hypothesis's mechanism, not an unrelated library entry. If the hypothesis names a
regime or market state, at least one expression must carry that conditional gate.
Do not emit two expressions that compute the same thing.

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
<hypothesis>in trending, high-volume markets, the combination of momentum and volume confirmation shows a stronger and more stable edge when normalized across assets. this suggests that stocks with both upward momentum and increasing volume are more likely to continue performing well in the next period.</hypothesis>

<func>ZSCORE($return, 1): variables $return, window 1, purpose standardise return across assets today</func>
<func>ZSCORE($volume, 1): variables $volume, window 1, purpose standardise volume across assets today</func>
<func>TS_ZSCORE($return, 10): variables $return, window 10, purpose rolling z-score of return over 10 days</func>
<func>TS_ZSCORE($volume, 10): variables $volume, window 10, purpose rolling z-score of volume over 10 days</func>
<func>TS_RANK($return, 10): variables $return, window 10, purpose time-series rank of return over 10 days</func>
<func>TS_RANK($volume, 10): variables $volume, window 10, purpose time-series rank of volume over 10 days</func>
<func>TS_CORR($return, $volume, 10): variables $return, $volume, window 10, purpose correlation between return and volume over 10 days</func>
<func>REGRESI($return, $volume, 10): variables $return, $volume, window 10, purpose regression residual of return on volume over 10 days</func>
<func>RSI($return, 14): variables $return, window 14, purpose relative strength index of return over 14 periods</func>
<func>MACD($return, 12, 26): variables $return, window 12, 26, purpose moving average convergence divergence of return</func>
<func>SMA($return, 10, 1): variables $return, window 10, purpose simple moving average of return over 10 periods</func>
<func>EMA($return, 10): variables $return, window 10, purpose exponential moving average of return over 10 periods</func>
<func>DELTA($return, 10): variables $return, window 10, purpose change in return over 10 periods</func>
<func>DELTA($volume, 10): variables $volume, window 10, purpose change in volume over 10 periods</func>
<func>TS_MEAN($return, 10): variables $return, window 10, purpose mean of return over 10 days</func>
<func>TS_MEAN($volume, 10): variables $volume, window 10, purpose mean of volume over 10 days</func>
<func>TS_STD($return, 10): variables $return, window 10, purpose standard deviation of return over 10 days</func>
<func>TS_STD($volume, 10): variables $volume, window 10, purpose standard deviation of volume over 10 days</func>
<func>TS_MIN($return, 10): variables $return, window 10, purpose minimum of return over 10 days</func>
<func>TS_MAX($return, 10): variables $return, window 10, purpose maximum of return over 10 days</func>
<func>TS_ARGMAX($return, 10): variables $return, window 10, purpose index of maximum return over 10 days</func>
<func>TS_ARGMIN($return, 10): variables $return, window 10, purpose index of minimum return over 10 days</func>
<func>TS_QUANTILE($return, 0.5, 10): variables $return, window 10, purpose median of return over 10 days</func>
<func>TS_QUANTILE($volume, 0.5, 10): variables $volume, window 10, purpose median of volume over 10 days</func>
<func>TS_CORR($return, $volume, 10): variables $return, $volume, window 10, purpose correlation between return and volume over 10 days</func>
<func>TS_COVARIANCE($return, $volume, 10): variables $return, $volume, window 10, purpose covariance between return and volume over 10 days</func>
<func>TS_MAD($return, 10): variables $return, window 10, purpose rolling median absolute deviation of return over 10 days</func>
<func>TS_MAD($volume, 10): variables $volume, window 10, purpose rolling median absolute deviation of volume over 10 days</func>
<func>PERCENTILE($return, 0.5, 10): variables $return, window 10, purpose median of return over 10 days</func>
<func>PERCENTILE($volume, 0.5, 10): variables $volume, window 10, purpose median of volume over 10 days</func>
<func>CROSS($return, $volume, 10): variables $return, $volume, window 10, purpose cross between return and volume over 10 days</func>
<func>MAX($return, $volume): variables $return, $volume, purpose pairwise maximum of return and volume</func>
<func>MIN($return, $volume): variables $return, $volume, purpose pairwise minimum of return and volume</func>
<func>LOG($return): variables $return, purpose natural logarithm of return</func>
<func>LOG($volume): variables $volume, purpose natural logarithm of volume</func>
<func>POW($return, 2): variables $return, purpose square of return</func>
<func>POW($volume, 2): variables $volume, purpose square of volume</func>
<func>ABS($return): variables $return, purpose absolute value of return</func>
<func>ABS($volume): variables $volume, purpose absolute value of volume</func>
<func>INV($return): variables $return, purpose reciprocal of return</func>
<func>INV($volume): variables $volume, purpose reciprocal of volume</func>
<func>FLOOR($return): variables $return, purpose floor of return</func>
<func>FLOOR($volume): variables $volume, purpose floor of volume</func>
<func>PROD($return, 10): variables $return, window 10, purpose product of return over 10 days</func>
<func>PROD($volume, 10): variables $volume, window 10, purpose product of volume over 10 days</func>
<func>RANK($return): variables $return, purpose rank of return across all stocks today</func>
<func>RANK($volume): variables $volume, purpose rank of volume across all stocks today</func>
<func>ZSCORE($return): variables $return, purpose standardise return across all stocks today</func>
<func>ZSCORE($volume): variables $volume, purpose standardise volume across all stocks today</func>
<func>MEAN($return): variables $return, purpose mean of return across all stocks today</func>
<func>MEAN($volume): variables $volume, purpose mean of volume across all stocks today</func>
<func>STD($return): variables $return, purpose standard deviation of return across all stocks today</func>
<func>STD($volume): variables $volume, purpose standard deviation of volume across all stocks today</func>
<func>SKEW($return): variables $return, purpose skewness of return across all stocks today</func>
<func>SKEW($volume): variables $volume, purpose skewness of volume across all stocks today</func>
<func>KURT($return): variables $return, purpose kurtosis of return across all stocks today</func>
<func>KURT($volume): variables $volume, purpose kurtosis of volume across all stocks today</func>
<func>MAX($return): variables $return, purpose maximum of return across all stocks today</func>
<func>MAX($volume): variables $volume, purpose maximum of volume across all stocks today</func>
<func>MIN($return): variables $return, purpose minimum of return across all stocks today</func>
<func>MIN($volume): variables $volume, purpose minimum of volume across all stocks today</func>
<func>MEDIAN($return): variables $return, purpose median of return across all stocks today</func>
<func>MEDIAN($volume): variables $volume, purpose median of volume across all stocks today</func>
<func>TS_RANK($return, 10): variables $return, window 10, purpose time-series rank of return over 10 days</func>
<func>TS_RANK($volume, 10): variables $volume, window 10, purpose time-series rank of volume over 10 days</func>
<func>TS_ZSCORE($return, 10): variables $return, window 10, purpose rolling z-score of return over 10 days</func>
<func>TS_ZSCORE($volume, 10): variables $volume, window 10, purpose rolling z-score of volume over 10 days</func>
<func>TS_CORR($return, $volume, 10): variables $return, $volume, window 10, purpose correlation between return and volume over 10 days</func>
<func>REGRESI($return, $volume, 10): variables $return, $volume, window 10, purpose regression residual of return on volume over 10 days</func>
<func>REGBETA($return, $volume, 10): variables $return, $volume, window 10, purpose regression slope of return on volume over 10 samples</func>
<func>SEQ($return, 10): variables $return, window 10, purpose sequence 1..10 for regression</func>
<func>SEQ($volume, 10): variables $volume, window 10, purpose sequence 1..10 for regression</func>

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
  "hypothesis": "in trending, high-volume markets, the combination of momentum and volume confirmation shows a stronger and more stable edge when normalized across assets",
  "factors": [
    {
      "name": "ZScoreReturn",
      "expression": "ZSCORE($return, 1)",
      "explanation": "Standardized return across assets to identify relative performance."
    },
    {
      "name": "ZScoreVolume",
      "expression": "ZSCORE($volume, 1)",
      "explanation": "Standardized volume across assets to identify relative trading activity."
    },
    {
      "name": "TSZScoreReturn",
      "expression": "TS_ZSCORE($return, 10)",
      "explanation": "Rolling z-score of return over 10 days to identify relative performance trends."
    },
    {
      "name": "TSZScoreVolume",
      "expression": "TS_ZSCORE($volume, 10)",
      "explanation": "Rolling z-score of volume over 10 days to identify relative trading activity trends."
    },
    {
      "name": "TSRankReturn",
      "expression": "TS_RANK($return, 10)",
      "explanation": "Time-series rank of return over 10 days to identify relative performance."
    },
    {
      "name": "TSRankVolume",
      "expression": "TS_RANK($volume, 10)",
      "explanation": "Time-series rank of volume over 10 days to identify relative trading activity."
    },
    {
      "name": "TSReturnVolumeCorr",
      "expression": "TS_CORR($return, $volume, 10)",
      "explanation": "Correlation between return and volume over 10 days to identify co-movement."
    },
    {
      "name": "RegresiReturnVolume",
      "expression": "REGRESI($return, $volume, 10)",
      "explanation": "Regression residual of return on volume over 10 days to identify unexplained variation."
    },
    {
      "name": "RSIReturn",
      "expression": "RSI($return, 14)",
      "explanation": "Relative strength index of return over 14 periods to measure momentum."
    },
    {
      "name": "MACDReturn",
      "expression": "MACD($return, 12, 26)",
      "explanation": "Moving average convergence divergence of return to identify trend changes."
    },
    {
      "name": "SMAReturn",
      "expression": "SMA($return, 10, 1)",
      "explanation": "Simple moving average of return over 10 periods to smooth out price trends."
    },
    {
      "name": "EMAReturn",
      "expression": "EMA($return, 10)",
      "explanation": "Exponential moving average of return over 10 periods to emphasize recent trends."
    },
    {
      "name": "DeltaReturn",
      "expression": "DELTA($return, 10)",
      "explanation": "Change in return over 10 periods to measure price movement."
    },
    {
      "name": "DeltaVolume",
      "expression": "DELTA($volume, 10)",
      "explanation": "Change in volume over 10 periods to measure trading activity."
    },
    {
      "name": "TSMEANReturn",
      "expression": "TS_MEAN($return, 10)",
      "explanation": "Mean of return over 10 days to identify average performance."
    },
    {
      "name": "TSMEANVolume",
      "expression": "TS_MEAN($volume, 10)",
      "explanation": "Mean of volume over 10 days to identify average trading activity."
    },
    {
      "name": "TSSTDReturn",
      "expression": "TS_STD($return, 10)",
      "explanation": "Standard deviation of return over 10 days to measure volatility."
    },
    {
      "name": "TSSTDVolume",
      "expression": "TS_STD($volume, 10)",
      "explanation": "Standard deviation of volume over 10 days to measure volatility in trading activity."
    },
    {
      "name": "TSMinReturn",
      "expression": "TS_MIN($return, 10)",
      "explanation": "Minimum of return over 10 days to identify worst performance."
    },
    {
      "name": "TSMaxReturn",
      "expression": "TS_MAX($return, 10)",
      "explanation": "Maximum of return over 10 days to identify best performance."
    },
    {
      "name": "TSArgMaxReturn",
      "expression": "TS_ARGMAX($return, 10)",
      "explanation": "Index of maximum return over 10 days to identify best performing day."
    },
    {
      "name": "TSArgMinReturn",
      "expression": "TS_ARGMIN($return, 10)",
      "explanation": "Index of minimum return over 10 days to identify worst performing day."
    },
    {
      "name": "TSQuantileReturn",
      "expression": "TS_QUANTILE($return, 0.5, 10)",
      "explanation": "Median of return over 10 days to identify central tendency."
    },
    {
      "name": "TSQuantileVolume",
      "expression": "TS_QUANTILE($volume, 0.5, 10)",
      "explanation": "Median of volume over 10 days to identify central tendency in trading activity."
    },
    {
      "name": "TSReturnVolumeCorr",
      "expression": "TS_CORR($return, $volume, 10)",
      "explanation": "Correlation between return and volume over 10 days to identify co-movement."
    },
    {
      "name": "TSReturnVolumeCovar",
      "expression": "TS_COVARIANCE($return, $volume, 10)",
      "explanation": "Covariance between return and volume over 10 days to identify relationship strength."
    },
    {
      "name": "TSMADReturn",
      "expression": "TS_MAD($return, 10)",
      "explanation": "Rolling median absolute deviation of return over 10 days to measure volatility."
    },
    {
      "name": "TSMADVolume",
      "expression": "TS_MAD($volume, 10)",
      "explanation": "Rolling median absolute deviation of volume over 10 days to measure volatility in trading activity."
    },
    {
      "name": "TSPercentileReturn",
      "expression": "PERCENTILE($return, 0.5, 10)",
      "explanation": "Median of return over 10 days to identify central tendency."
    },
    {
      "name": "TSPercentileVolume",
      "expression": "PERCENTILE($volume, 0.5, 10)",
      "explanation": "Median of volume over 10 days to identify central tendency in trading activity."
    },
    {
      "name": "TSCrossReturnVolume",
      "expression": "CROSS($return, $volume, 10)",
      "explanation": "Cross between return and volume over 10 days to identify trend changes."
    },
    {
      "name": "MaxReturnVolume",
      "expression": "MAX($return, $volume)",
      "explanation": "Pairwise maximum of return and volume to identify strongest signal."
    },
    {
      "name": "MinReturnVolume",
      "expression": "MIN($return, $volume)",
      "explanation": "Pairwise minimum of return and volume to identify weakest signal."
    },
    {
      "name": "LogReturn",
      "expression": "LOG($return)",
      "explanation": "Natural logarithm of return to transform distribution."
    },
    {
      "name": "LogVolume",
      "expression": "LOG($volume)",
      "explanation": "Natural logarithm of volume to transform distribution."
    },
    {
      "name": "PowReturn",
      "expression": "POW($return, 2)",
      "explanation": "Square of return to emphasize magnitude."
    },
    {
      "name": "PowVolume",
      "expression": "POW($volume, 2)",
      "explanation": "Square of volume to emphasize magnitude."
    },
    {
      "name": "AbsReturn",
      "expression": "ABS($return)",
      "explanation": "Absolute value of return to measure magnitude without direction."
    },
    {
      "name": "AbsVolume",
      "expression": "ABS($volume)",
      "explanation": "Absolute value of volume to measure magnitude without direction."
    },
    {
      "name": "InvReturn",
      "expression": "INV($return)",
      "explanation": "Reciprocal of return to invert performance metric."
    },
    {
      "name": "InvVolume",
      "expression": "INV($volume)",
      "explanation": "Reciprocal of volume to invert trading activity metric."
    },
    {
      "name": "FloorReturn",
      "expression": "FLOOR($return)",
      "explanation": "Floor of return to round down to nearest integer."
    },
    {
      "name": "FloorVolume",
      "expression": "FLOOR($volume)",
      "explanation": "Floor of volume to round down to nearest integer."
    },
    {
      "name": "ProdReturn",
      "expression": "PROD($return, 10)",
      "explanation": "Product of return over 10 days to measure cumulative performance."
    },
    {
      "name": "ProdVolume",
      "expression": "PROD($volume, 10)",
      "explanation": "Product of volume over 10 days to measure cumulative trading activity."
    },
    {
      "name": "RankReturn",
      "expression": "RANK($return)",
      "explanation": "Rank of return across all stocks today to identify relative performance."
    },
    {
      "name": "RankVolume",
      "expression": "RANK($volume)",
      "explanation": "Rank of volume across all stocks today to identify relative trading activity."
    },
    {
      "name": "ZScoreReturnAll",
      "expression": "ZSCORE($return)",
      "explanation": "Standardized return across all stocks today to identify relative performance."
    },
    {
      "name": "ZScoreVolumeAll",
      "expression": "ZSCORE($volume)",
      "explanation": "Standardized volume across all stocks today to identify relative trading activity."
    },
    {
      "name": "MeanReturn",
      "expression": "MEAN($return)",
      "explanation": "Mean of return across all stocks today to identify average performance."
    },
    {
      "name": "MeanVolume",
      "expression": "MEAN($volume)",
      "explanation": "Mean of volume across all stocks today to identify average trading activity."
    },
    {
      "name": "STDReturn",
      "expression": "STD($return)",
      "explanation": "Standard deviation of return across all stocks today to measure volatility."
    },
    {
      "name": "STDVolume",
      "expression": "STD($volume)",
      "explanation": "Standard deviation of volume across all stocks today to measure volatility in trading activity."
    },
    {
      "name": "SKEWReturn",
      "expression": "SKEW($return)",
      "explanation": "Skewness of return across all stocks today to measure asymmetry in distribution."
    },
    {
      "name": "SKEWVolume",
      "expression": "SKEW($volume)",
      "explanation": "Skewness of volume across all stocks today to measure asymmetry in distribution."
    },
    {
      "name": "KURTReturn",
      "expression": "KURT($return)",
      "explanation": "Kurtosis of return across all stocks today to measure tail risk."
    },
    {
      "name": "KURTVolume",
      "expression": "KURT($volume)",
      "explanation": "Kurtosis of volume across all stocks today to measure tail risk."
    },
    {
      "name": "MAXReturn",
      "expression": "MAX($return)",
      "explanation": "Maximum of return across all stocks today to identify best performance."
    },
    {
      "name": "MAXVolume",
      "expression": "MAX($volume)",
      "explanation": "Maximum of volume across all stocks today to identify highest trading activity."
    },
    {
      "name": "MINReturn",
      "expression": "MIN($return)",
      "explanation": "Minimum of return across all stocks today to identify worst performance."
    },
    {
      "name": "MINVolume",
      "expression": "MIN($volume)",
      "explanation": "Minimum of volume across all stocks today to identify lowest trading activity."
    },
    {
      "name": "MEDIANReturn",
      "expression": "MEDIAN($return)",
      "explanation": "Median of return across all stocks today to identify central tendency."
    },
    {
      "name": "MEDIANVolume",
      "expression": "MEDIAN($volume)",
      "explanation": "Median of volume across all stocks today to identify central tendency in trading activity."
    },
    {
      "name": "TSRankReturn",
      "expression": "TS_RANK($return, 10)",
      "explanation": "Time-series rank of return over 10 days to identify relative performance."
    },
    {
      "name": "TSRankVolume",
      "expression": "TS_RANK($volume, 10)",
      "explanation": "Time-series rank of volume over 10 days to identify relative trading activity."
    },
    {
      "name": "TSZScoreReturn",
      "expression": "TS_ZSCORE($return, 10)",
      "explanation": "Rolling z-score of return over 10 days to identify relative performance trends."
    },
    {
      "name": "TSZScoreVolume",
      "expression": "TS_ZSCORE($volume, 10)",
      "explanation": "Rolling z-score of volume over 10 days to identify relative trading activity trends."
    },
    {
      "name": "TSReturnVolumeCorr",
      "expression": "TS_CORR($return, $volume, 10)",
      "explanation": "Correlation between return and volume over 10 days to identify co-movement."
    },
    {
      "name": "RegresiReturnVolume",
      "expression": "REGRESI($return, $volume, 10)",
      "explanation": "Regression residual of return on volume over 10 days to identify unexplained variation."
    }
  ]
}
```
