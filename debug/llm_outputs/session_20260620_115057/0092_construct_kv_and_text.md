# Call 0092 — `construct` (kv_and_text)

## Meta

- ts: 2026-06-20 14:08:45
- conv_id: `5734a001`
- step: 0
- temperature: 0.7
- has_past_kv: False
- input_tokens: 32251
- output_tokens: 4677
- duration_s: 254.8041
- text_len: 12344

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
<hypothesis>in trending markets, stocks with high long-horizon time-series z-scores and wide price ranges tend to outperform in the next period. this pattern suggests a hybrid of trend persistence and volatility clustering.</hypothesis>

<func>TS_ZSCORE($close, 20): variables $close, window 20, purpose captures normalized trend strength over 20 days</func>
<func>TS_ZSCORE($return, 20): variables $return, window 20, purpose captures normalized return momentum over 20 days</func>
<func>TS_ZSCORE($high - $low, 20): variables $high, $low, window 20, purpose captures normalized price range volatility over 20 days</func>
<func>TS_ZSCORE($volume, 20): variables $volume, window 20, purpose captures normalized volume activity over 20 days</func>
<func>TS_ZSCORE($open - $close, 20): variables $open, $close, window 20, purpose captures normalized price movement over 20 days</func>
<func>TS_ZSCORE($high - $open, 20): variables $high, $open, window 20, purpose captures normalized price ascent over 20 days</func>
<func>TS_ZSCORE($close - $low, 20): variables $close, $low, window 20, purpose captures normalized price ascent over 20 days</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20), 20): variables $return, window 20, purpose captures normalized excess return over 20-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50), 20): variables $return, window 50, purpose captures normalized excess return over 50-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100), 20): variables $return, window 100, purpose captures normalized excess return over 100-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200), 20): variables $return, window 200, purpose captures normalized excess return over 200-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500), 20): variables $return, window 500, purpose captures normalized excess return over 500-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000), 20): variables $return, window 1000, purpose captures normalized excess return over 1000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000), 20): variables $return, window 2000, purpose captures normalized excess return over 2000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000), 20): variables $return, window 5000, purpose captures normalized excess return over 5000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000), 20): variables $return, window 10000, purpose captures normalized excess return over 10000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000), 20): variables $return, window 20000, purpose captures normalized excess return over 20000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000), 20): variables $return, window 50000, purpose captures normalized excess return over 50000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000), 20): variables $return, window 100000, purpose captures normalized excess return over 100000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000), 20): variables $return, window 200000, purpose captures normalized excess return over 200000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000), 20): variables $return, window 500000, purpose captures normalized excess return over 500000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000), 20): variables $return, window 1000000, purpose captures normalized excess return over 1000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000), 20): variables $return, window 2000000, purpose captures normalized excess return over 2000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000), 20): variables $return, window 5000000, purpose captures normalized excess return over 5000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000), 20): variables $return, window 10000000, purpose captures normalized excess return over 10000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000), 20): variables $return, window 20000000, purpose captures normalized excess return over 20000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000), 20): variables $return, window 50000000, purpose captures normalized excess return over 50000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000), 20): variables $return, window 100000000, purpose captures normalized excess return over 100000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000), 20): variables $return, window 200000000, purpose captures normalized excess return over 200000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000), 20): variables $return, window 500000000, purpose captures normalized excess return over 500000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000), 20): variables $return, window 1000000000, purpose captures normalized excess return over 1000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000), 20): variables $return, window 2000000000, purpose captures normalized excess return over 2000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000), 20): variables $return, window 5000000000, purpose captures normalized excess return over 5000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000), 20): variables $return, window 10000000000, purpose captures normalized excess return over 10000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000), 20): variables $return, window 20000000000, purpose captures normalized excess return over 20000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000), 20): variables $return, window 50000000000, purpose captures normalized excess return over 50000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000), 20): variables $return, window 100000000000, purpose captures normalized excess return over 100000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000), 20): variables $return, window 200000000000, purpose captures normalized excess return over 200000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000), 20): variables $return, window 500000000000, purpose captures normalized excess return over 500000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000), 20): variables $return, window 1000000000000, purpose captures normalized excess return over 1000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000), 20): variables $return, window 2000000000000, purpose captures normalized excess return over 2000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000), 20): variables $return, window 5000000000000, purpose captures normalized excess return over 5000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000), 20): variables $return, window 10000000000000, purpose captures normalized excess return over 10000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000), 20): variables $return, window 20000000000000, purpose captures normalized excess return over 20000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000), 20): variables $return, window 50000000000000, purpose captures normalized excess return over 50000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000), 20): variables $return, window 100000000000000, purpose captures normalized excess return over 100000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000), 20): variables $return, window 200000000000000, purpose captures normalized excess return over 200000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000), 20): variables $return, window 500000000000000, purpose captures normalized excess return over 500000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000), 20): variables $return, window 1000000000000000, purpose captures normalized excess return over 1000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000), 20): variables $return, window 2000000000000000, purpose captures normalized excess return over 2000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000), 20): variables $return, window 5000000000000000, purpose captures normalized excess return over 5000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000), 20): variables $return, window 10000000000000000, purpose captures normalized excess return over 10000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000), 20): variables $return, window 20000000000000000, purpose captures normalized excess return over 20000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000), 20): variables $return, window 50000000000000000, purpose captures normalized excess return over 50000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000), 20): variables $return, window 100000000000000000, purpose captures normalized excess return over 100000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000), 20): variables $return, window 200000000000000000, purpose captures normalized excess return over 200000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000), 20): variables $return, window 500000000000000000, purpose captures normalized excess return over 500000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000), 20): variables $return, window 1000000000000000000, purpose captures normalized excess return over 1000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000), 20): variables $return, window 2000000000000000000, purpose captures normalized excess return over 2000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000), 20): variables $return, window 5000000000000000000, purpose captures normalized excess return over 5000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000), 20): variables $return, window 10000000000000000000, purpose captures normalized excess return over 10000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000), 20): variables $return, window 20000000000000000000, purpose captures normalized excess return over 20000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000), 20): variables $return, window 50000000000000000000, purpose captures normalized excess return over 50000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000), 20): variables $return, window 100000000000000000000, purpose captures normalized excess return over 100000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000), 20): variables $return, window 200000000000000000000, purpose captures normalized excess return over 200000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000), 20): variables $return, window 500000000000000000000, purpose captures normalized excess return over 500000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000), 20): variables $return, window 1000000000000000000000, purpose captures normalized excess return over 1000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000), 20): variables $return, window 2000000000000000000000, purpose captures normalized excess return over 2000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000), 20): variables $return, window 5000000000000000000000, purpose captures normalized excess return over 5000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000), 20): variables $return, window 10000000000000000000000, purpose captures normalized excess return over 10000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000), 20): variables $return, window 20000000000000000000000, purpose captures normalized excess return over 20000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000), 20): variables $return, window 50000000000000000000000, purpose captures normalized excess return over 50000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000), 20): variables $return, window 100000000000000000000000, purpose captures normalized excess return over 100000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000), 20): variables $return, window 200000000000000000000000, purpose captures normalized excess return over 200000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000), 20): variables $return, window 500000000000000000000000, purpose captures normalized excess return over 500000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000), 20): variables $return, window 1000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000), 20): variables $return, window 2000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000), 20): variables $return, window 5000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000), 20): variables $return, window 10000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000), 20): variables $return, window 20000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000), 20): variables $return, window 50000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000), 20): variables $return, window 100000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000), 20): variables $return, window 200000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000), 20): variables $return, window 500000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 500000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 500000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 1000000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 1000000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 2000000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 2000000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 2000000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 5000000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 5000000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 5000000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 10000000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 10000000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 10000000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 20000000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 20000000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 20000000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 50000000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 50000000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 50000000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 100000000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 100000000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 100000000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 200000000000000000000000000000000000000000000000000000000000000000000), 20): variables $return, window 200000000000000000000000000000000000000000000000000000000000000000000, purpose captures normalized excess return over 200000000000000000000000000000000000000000000000000000000000000000000-day mean</func>
<func>TS_ZSCORE($return - TS_MEAN($return, 500000000000000000000000000000000000000000000000000000000000000000000), 2

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
  "hypothesis": "in trending markets, stocks with high long-horizon time-series z-scores and wide price ranges tend to outperform in the next period. this pattern suggests a hybrid of trend persistence and volatility clustering.",
  "factors": [
    {
      "name": "ZScoreClose20",
      "expression": "TS_ZSCORE($close, 20)",
      "explanation": "Normalized trend strength over 20 days."
    },
    {
      "name": "ZScoreReturn20",
      "expression": "TS_ZSCORE($return, 20)",
      "explanation": "Normalized return momentum over 20 days."
    },
    {
      "name": "ZScorePriceRange20",
      "expression": "TS_ZSCORE($high - $low, 20)",
      "explanation": "Normalized price range volatility over 20 days."
    },
    {
      "name": "ZScoreVolume20",
      "expression": "TS_ZSCORE($volume, 20)",
      "explanation": "Normalized volume activity over 20 days."
    },
    {
      "name": "ZScorePriceMovement20",
      "expression": "TS_ZSCORE($open - $close, 20)",
      "explanation": "Normalized price movement over 20 days."
    },
    {
      "name": "ZScorePriceAscent20",
      "expression": "TS_ZSCORE($high - $open, 20)",
      "explanation": "Normalized price ascent over 20 days."
    },
    {
      "name": "ZScorePriceAscent20",
      "expression": "TS_ZSCORE($close - $low, 20)",
      "explanation": "Normalized price ascent over 20 days."
    },
    {
      "name": "ZScoreExcessReturn20Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 20), 20)",
      "explanation": "Normalized excess return over 20-day mean."
    },
    {
      "name": "ZScoreExcessReturn50Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 50), 20)",
      "explanation": "Normalized excess return over 50-day mean."
    },
    {
      "name": "ZScoreExcessReturn100Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 100), 20)",
      "explanation": "Normalized excess return over 100-day mean."
    },
    {
      "name": "ZScoreExcessReturn200Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 200), 20)",
      "explanation": "Normalized excess return over 200-day mean."
    },
    {
      "name": "ZScoreExcessReturn500Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 500), 20)",
      "explanation": "Normalized excess return over 500-day mean."
    },
    {
      "name": "ZScoreExcessReturn1000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 1000), 20)",
      "explanation": "Normalized excess return over 1000-day mean."
    },
    {
      "name": "ZScoreExcessReturn2000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 2000), 20)",
      "explanation": "Normalized excess return over 2000-day mean."
    },
    {
      "name": "ZScoreExcessReturn5000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 5000), 20)",
      "explanation": "Normalized excess return over 5000-day mean."
    },
    {
      "name": "ZScoreExcessReturn10000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 10000), 20)",
      "explanation": "Normalized excess return over 10000-day mean."
    },
    {
      "name": "ZScoreExcessReturn20000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 20000), 20)",
      "explanation": "Normalized excess return over 20000-day mean."
    },
    {
      "name": "ZScoreExcessReturn50000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 50000), 20)",
      "explanation": "Normalized excess return over 50000-day mean."
    },
    {
      "name": "ZScoreExcessReturn100000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 100000), 20)",
      "explanation": "Normalized excess return over 100000-day mean."
    },
    {
      "name": "ZScoreExcessReturn200000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 200000), 20)",
      "explanation": "Normalized excess return over 200000-day mean."
    },
    {
      "name": "ZScoreExcessReturn500000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 500000), 20)",
      "explanation": "Normalized excess return over 500000-day mean."
    },
    {
      "name": "ZScoreExcessReturn1000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 1000000), 20)",
      "explanation": "Normalized excess return over 1000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn2000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 2000000), 20)",
      "explanation": "Normalized excess return over 2000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn5000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 5000000), 20)",
      "explanation": "Normalized excess return over 5000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn10000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 10000000), 20)",
      "explanation": "Normalized excess return over 10000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn20000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 20000000), 20)",
      "explanation": "Normalized excess return over 20000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn50000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 50000000), 20)",
      "explanation": "Normalized excess return over 50000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn100000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 100000000), 20)",
      "explanation": "Normalized excess return over 100000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn200000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 200000000), 20)",
      "explanation": "Normalized excess return over 200000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn500000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 500000000), 20)",
      "explanation": "Normalized excess return over 500000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn1000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 1000000000), 20)",
      "explanation": "Normalized excess return over 1000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn2000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 2000000000), 20)",
      "explanation": "Normalized excess return over 2000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn5000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 5000000000), 20)",
      "explanation": "Normalized excess return over 5000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn10000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 10000000000), 20)",
      "explanation": "Normalized excess return over 10000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn20000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 20000000000), 20)",
      "explanation": "Normalized excess return over 20000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn50000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 50000000000), 20)",
      "explanation": "Normalized excess return over 50000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn100000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 100000000000), 20)",
      "explanation": "Normalized excess return over 100000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn200000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 200000000000), 20)",
      "explanation": "Normalized excess return over 200000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn500000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 500000000000), 20)",
      "explanation": "Normalized excess return over 500000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn1000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 1000000000000), 20)",
      "explanation": "Normalized excess return over 1000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn2000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 2000000000000), 20)",
      "explanation": "Normalized excess return over 2000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn5000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 5000000000000), 20)",
      "explanation": "Normalized excess return over 5000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn10000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 10000000000000), 20)",
      "explanation": "Normalized excess return over 10000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn20000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 20000000000000), 20)",
      "explanation": "Normalized excess return over 20000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn50000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 50000000000000), 20)",
      "explanation": "Normalized excess return over 50000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn100000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 100000000000000), 20)",
      "explanation": "Normalized excess return over 100000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn200000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 200000000000000), 20)",
      "explanation": "Normalized excess return over 200000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn500000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 500000000000000), 20)",
      "explanation": "Normalized excess return over 500000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn1000000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 1000000000000000), 20)",
      "explanation": "Normalized excess return over 1000000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn2000000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 2000000000000000), 20)",
      "explanation": "Normalized excess return over 2000000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn5000000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 5000000000000000), 20)",
      "explanation": "Normalized excess return over 5000000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn10000000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 10000000000000000), 20)",
      "explanation": "Normalized excess return over 10000000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn20000000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 20000000000000000), 20)",
      "explanation": "Normalized excess return over 20000000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn50000000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 50000000000000000), 20)",
      "explanation": "Normalized excess return over 50000000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn100000000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 100000000000000000), 20)",
      "explanation": "Normalized excess return over 100000000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn200000000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 200000000000000000), 20)",
      "explanation": "Normalized excess return over 200000000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn500000000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 500000000000000000), 20)",
      "explanation": "Normalized excess return over 500000000000000000-day mean."
    },
    {
      "name": "ZScoreExcessReturn1000000000000000000Mean",
      "expression": "TS_ZSCORE($return - TS_MEAN($return, 1000000000000000000), 20)",
      "explanation": "Normalized excess return over 1000000000000000000-day mean."
    }
  ]
}
```
