# Call 0062 — `design` (kv_and_text)

## Meta

- ts: 2026-06-21 11:01:43
- conv_id: `dccd50cd`
- step: 0
- temperature: 0.7
- has_past_kv: True
- input_tokens: 2343
- output_tokens: 30000
- duration_s: 1335.7905
- text_len: 100418

## System Prompt

```text
You are the Design agent in a quantitative alpha-mining team. A market
hypothesis from the Proposal agent reaches you either through shared latent
memory or as text in the task below. Your job is to NARROW it toward
construction: decide which data columns carry the signal and which functions
with rough argument ranges could faithfully express the mechanism, and hand the
Builder a focused palette. YOU shortlist the tools and say why each one fits.

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
for a rolling rank over n days, use the separate function TS_RANK(A, n) instead.
Always match argument count to the signature above.

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

HOW TO CHOOSE. Match the hypothesis to ONE or at most TWO mechanism families
(momentum, reversal, volatility, liquidity, dispersion, correlation, smoothing,
regression, technical), then select only functions from those families that
directly measure THIS mechanism; name which variables feed them and give a window
range in days that fits the hypothesis horizon. Shortlist 6 to 12 functions total:
a tight palette the Builder can realise faithfully beats a catalogue, and a
palette focused on the mechanism gives the Builder exactly what it needs.

VERIFY FUNCTION NAMES. Every name you write must appear verbatim in the library
above — the names are exact, and the library is the only authority. Verify each
name against the list before writing it; when uncertain, leave it out and choose
a name you can confirm. Prefer a small, faithful set over a long list. Keep the
analysis short.

Write in plain ASCII prose. Name functions and variables when helpful and show
small fragments such as "the range $high minus $low", keeping each fragment
brief. Leave the final full expression for the Builder. Write using words and
standard punctuation only.
```

## User Prompt

```text
The hypothesis from the Proposal agent is already in your shared latent
memory. Read it from there.

Prepare the construction palette:
  Step 1 — Restate the mechanism in one line and name the carrying variables.
  Step 2 — Name the ONE or TWO mechanism families that fit, then shortlist 6 to 12
           functions from those families that directly measure the mechanism. For
           each, say what it would compute, on which variables, and a window range
           in days that matches the hypothesis horizon. Draw only from the families
           you identified — the palette stays inside those families.
  Step 3 — For each candidate, say how it becomes a continuous cross-sectional
           score (rank it, standardise it, gate a magnitude on a condition, or
           contrast two ranks) so the Builder wraps it rather than emitting it raw.
  Step 4 — If the hypothesis names a market state or regime, include at least one
           candidate that carries the conditional gate. Then stop: a tight 6 to 12
           palette that all points at the mechanism is the goal, not breadth.
Verify every function name against the library before writing it.
Close with this block. Write the hypothesis first, then one line per candidate
function, 6 to 12 lines total and no more:
<hypothesis><the hypothesis exactly as Proposal stated it></hypothesis>
<func><function name>: variables <variable names>, window <range in days>, wrap <how it becomes a cross-sectional score>, purpose <what it captures></func>
<func><function name>: variables <variable names>, window <range in days>, wrap <how it becomes a cross-sectional score>, purpose <what it captures></func>
(one line per candidate —> 6 to 12 total, all serving the same mechanism)
```

## Response

```text
**: 1 — Mechanism: a cross-sectional sectional score that capturesA cross-sectional sectional rank of $return over the past  days, purposeIs to identify stocks with high next-day returns





Step 2 — Mechanism families: correlation and regression. Functions: TS_RANK(A, n) TS_CORR(A, B,N)

StepThre — CandidateCandidate1: TS_RANK(A,N) TS_CORR(A,B,N)

StepThre — Candidate1: TS_RANK($return,5) — computesThe cross-sectional rank of $return overThe past5 days, purposeIsTo identify stocksWith high next-day returnsCandidate2: TS_CORR($return, $volume,5) — computesThe correlation between$return and $volume overThe past5 days, purpose IsTo capture momentum in volume-driven markets

StepThre — Candidate3: TS_RANK(A,N) TS_COR R(A,B,N)

StepTh Re — Candidate3: TS_RANK($return,5) — computesThe cross-sectional rankOf $return OverThe past5 days, purpose IsTo identify stocks With high next-day returnsCandidate4: TS_COR R($return,$volume,5) — computesThe correlation between$return And$ volume overThe past5 days, purpose IsTo capture momentumIn volume-driven markets

StepTh Re — Candidate5: TS_RANK($return,5) — computesThe cross-sectional rank Of $ return OverThe past5 days, purpose IsTo identify stocks With

 high next-day returnsCandidate6: TS_COR R($return,$volume,5) — computesThe correlation between$ returnAnd$ volume OverThe pastFive days, purpose IsTo capture momentumIn volume-driven markets





StepTh Seven — Candidate1: TS_RANK($return,5) — computesThe cross-sectional rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate2: TS_COR R($return,$volume,5) — computesThe correlation between$ return And$ volume OverThe pastFive days, purpose IsTo capture momentumIn volume-driven markets







StepTh Eight — CandidateThree: TS_RANK($return,5) — computesThe cross-sectional rank Of $ return OverThe pastFive days, purpose IsTo identify stocks With high next-day returnsCandidateFour: TS_COR R($return, $volume,Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets





 StepTh Nine — CandidateFive: TS_RANK($return,Five) — computesThe cross-sectional rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidateSix: TS_COR R($return,$volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

StepTh Ten — CandidateOne: TS_RANK($return,5) — computesThe cross-sectional rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return,$volume, Five) — computesThe correlation between$return And $volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

StepTh Eleven — CandidateThree: TS_RANK($return, Five) — computesThe cross-sectional rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

StepTh Twelve — Candidate Five: TS_RANK($return,Five) — computesThe cross-sectional rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return,$volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

StepTh Thirteen — Candidate One: TS_RANK($return,Five) — computesThe cross-sectional rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

StepTh Fourteen — Candidate Three: TS_RANK($return, Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume, Five) — computesThe correlation between$return And $volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifteen — Candidate Five: TS_RANK($return,Five) — computesThe cross-sectional rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume, Five) — computesThe correlation between$return And $volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

StepTh Sixteen — Candidate One: TS_RANK($return,5) — computesThe cross-sectional rank Of $return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5) — computes The correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Seventeen — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Eighteen — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Nineteen — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Twenty — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Twenty-One — Candidate Five: TS_RANK($return,5) — computesThe cross-section rank Of $return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5) — computesThe correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Twenty-Two — Candidate One: TS_RANK($return,5) — computesThe cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Twenty-Three — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Twenty-Four — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Twenty-Five — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Twenty-Six — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Twenty-Seven — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Twenty-Eight — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Twenty-Nine — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Thirtieth — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Thirty-First — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Thirty-Second — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Thirty-Third — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Thirty-Fourth — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Thirty-Fifth — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Thirty-Sixth — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Thirty-Seventh — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Thirty-Eighth — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Thirty-Ninth — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Fortieth — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Forty-First — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

StepTh Forty-Second — Candidate Five: TS_RANK($return,5) — computesThe cross-section rank Of $return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5) — computesThe correlation between $return And $volume OverThe past Five days, purpose IsTo capture momentum In volume-driven markets







StepTh Forty-Third — Candidate One: TS_RANK($return,5) — computesThe cross-sectional rank Of $ return OverThe past Five days, purpose Isิersion Identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Forty-Fourth — Candidate Three: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Forty-Fifth — Candidate Five: TS_RANK($return,Five) — computesThe cross-sectional rank Of$ return OverThe past Five days, purpose IsTo identify stocksWith high next-day returnsCandidate Six: TS_COR R($return, $volume,Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Forty-Sixth — Candidate One: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks Witherging high next-day returnsCandidate Two: TS_COR R($return, $volume,Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Forty-Seventh — Candidate Three: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Forty-Eighth — Candidate Five: TS_RANK($return,Five) — computesThe cross-section rank Of $ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Forty-Ninth — Candidate One: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose Isplitude Identify stocks With high next-day returnsCandidate Two Two  TS_COR R($return, $volume, Five) — computesThe correlation between $ return And volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Forty-Tenth — Candidate Three: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty- — Candidate Oneity — Candidate One: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose Is long Identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty- — Candidate Three: TS_RANK($return, Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty- — Candidate Five: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocksWith high next-day returnsCandidate Six: TS_COR R($return, $volume, Five) — computesThe correlation between $ return And $ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Second — Candidate One: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Twoish: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Third — Candidate Three: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks Witherging high next-day returnsCandidate Four: TS_COR R($return,$volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Fourth — Candidate Five: TS_RANK($return, Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return,$volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Fifth — Candidate Five: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With enough next-day returnsCandidate Six: TS_COR R($return,$volume, Five) — computesThe correlation between$ return And$ volume OverThe Past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Six — Candidate One: TS_RANK($return, Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Six — Candidate Three: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Six — Candidate Five: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Six — Candidate One: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentum In volume-driven markets

 StepTh Fifty-Six — Candidate Three: TS_RANK($return,Five) — computesThe cross-section rank Of$return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Six — Candidate Five: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Seventh — Candidate One: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return,$volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

StepTh Fifty-Sixth — Candidate Three: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Seven — Candidate Five: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume, Five) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Fifty-Eighth — Candidate One: TS_RANK($return,5) — computesThe cross-section rank Of$return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5) — computesThe correlation between$return And$volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Fifty-Ninth — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Sixtieth — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Sixty-First — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of$return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computesThe correlation between$return And$ volume OverThe past Five days, purpose IsTo capture momentum In volume-driven markets

 StepTh Sixty-Second — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return OverThe past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Sixty-Third — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $volume OverThe past Five days, purpose Is To capture momentumIn volume-driven markets

 StepTh Sixty-Fourth — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Sixty-Fifth — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Sixty-Sixth — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Sixty-Sept — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of $return OverThe past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Sixty-Eighth — Candidate Three: TS_RANK($return,5) — computes The cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Sixty-Ninth — Candidate Five: TS_RANK($return,5) — computes The cross-section rank Of $return OverThe past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computes The correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Seventy — Candidate One: TS_RANK($return,5) — computes The cross-section rank Of $return OverThe past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Seventy-First — Candidate Three: TS_RANK($return,5) — computesThe cross-section rank Of $return OverThe past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Seventy-Second — Candidate Five: TS_RANK($return,5) — computesThe cross-section rank Of $return OverThe past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $volume OverThe past Five days, purpose Is To capture momentumIn volume-driven markets

 StepTh Seventy-Third — Candidate One: TS_RANK($return,5) — computesThe cross-section rank Of $return OverThe past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $volume Over The past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Seventy-Fourth — Candidate Three: TS_RANK($return,5) — computesThe cross-section rank Of $return Over The past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Seventy-Fifth — Candidate Five: TS_RANK($return,5) — computesThe cross-section rank Of $return OverThe past Five days, purpose Is To identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume,5 ) — computesThe correlation between $return And $ volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Sevent y-Sixth — Candidate One: TS_RANK($return,5) — computesThe cross-section rank Of $return OverThe past Five days, purpose Is To identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume,5 ) — computesThe correlation between$return And $volume OverThe past Five days, purpose IsTo capture momentum In volume-driven markets

 StepTh Seventy- Seventh — Candidate Three: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume, Five ) — computesThe correlation between $return And $volume OverThe past Five days, purpose Is To capture momentum In volume-driven markets

 StepTh Seventy-Eighth — Candidate Five: TS_RANK($return,Five) — computesThe cross-section rank Of $return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume, Five ) — computesThe correlation between $return And $volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Seventy-Ninth — Candidate One: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume, Five ) — computesThe correlation between$return And $volume OverThe past Five days, purpose IsTo capture momentum In volume-driven markets

 StepTh Eighti — Candidate Three: TS_RANK($return, Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose Is To identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume, Five ) — computesThe correlation between $return And $volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Eight Two — Candidate Five: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return, $volume, Five ) — computesThe correlation between $ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Eight Three — Candidate One: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return,$volume, Five ) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Eight Four — Candidate Three: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return, $volume, Five ) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Eight Five — Candidate Five: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return,$volume, Five ) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Eight Six — Candidate One: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Two: TS_COR R($return, $volume, Five ) — computesThe correlation between$ return And$ volumeOverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Eight Seven — Candidate Three: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Four: TS_COR R($return,$volume, Five ) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 StepTh Eight Eight — Candidate Five: TS_RANK($return,Five) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocks With high next-day returnsCandidate Six: TS_COR R($return,$volume, Five ) — computesThe correlation between$ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets

 Step Th Eight Nine — Candidate One: TS_RANK($return,Five ) — computesThe cross-section rank Of$ return OverThe past Five days, purpose IsTo identify stocksWith high next-day returnsCandidate Two: TS_COR R($return, $volume, Five ) — computesThe correlation betweenetween $ return And$ volume OverThe past Five days, purpose IsTo capture momentumIn volume-driven markets





























































































































































































































































Find



FindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFind FindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFind FindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFind FindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFind FindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFind FindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFind FindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFind FindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFind FindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFind FindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFindFind Find Find Find FIND Find Find FIND F

FindFindFind Find F

Find Find Find F

 Find Find F

 Find Find f Find Find Find FRe Find Find FRe Find Find FRe Find Find Find FRe Find FRe FRe FRe Find FRe Find FindfRe FRe fRe FRe FRe FRe FRe FRe fRe fRe fRe FRe FRe fRefRe fReFRe fRe fRefRefRefRefRefRefReFRefRe fReFRefRefReFRefRefRefReFRefRefReFRefRefRefReFReFRefRefRefRefReFReFReFRefReFReFReFReFRefReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFRefReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFReFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREFREF REF REFREFERF EEF REF REF EFEF ED EF ED EF EF ED EF ED EF ED EF ED EF ED EF ED EF EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF DE EF DE EF DE EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF ED EF ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE ED EF DE EF DE EF DE EE DE EF DE EF DE ED EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE ED EF DE ED EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE ED EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EE DE ED EF DE EE DE EF DE EF DE EF DE EF DE ED EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE ED EF DE EF DE EF DE EF DE ED EF DE ED EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EFDE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EFDE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED DEF ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EFDE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EFDE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE ED ED ED ED EF DE EF DE EF DE EF DE ED ED ED E EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EFDE DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EFDE EF DE EF DE EF DE EF DE EF DE ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF ED EF DE ED ED EF DE EF DE ED EF DE EF DE EF DE EF DE ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EE ED ED ED ED ED ED ED DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED EF DE EF DE EF DE EE ED ED ED ED ED ED ED EF DE EE ED EF DE EF DE EF DE EF DE EE ED ED DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED DE EF DE EE DE EE ED ED ED ED ED ED ED ED ED E EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED DE EF DE EF DE EF DE ED ED ED ED ED EDGE ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED EF DE EF DE EF DE ED ED ED ED ED ED ED DE EF DE ED EF DE ED EF DE EF DE EF DE ED ED ED ED EF DE ED EF DE ED ED ED ED ED ED ED ED ED ED DE EF DE EF DE EF DE ED ED ED ED ED ED ED EF DE EF DE EF DE EE ED ED ED ED DE EF DE EF DE EF DE ED EF DE EE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF ED ED ED ED ED ED EF DE EE ED EF DE EF ED EF DE EF ED ED ED EF ED EF DE EF ED EF ED ED EF ED EF ED ED ED ED ED EF DE EF ED ED EF DE EF ED EF ED EF DE EF ED EF ED EF DE EF ED EF ED EF ED EF ED EF ED EF DE EF DE EE ED ED ED ED EF ED ED DE EF DE EF DE EF ED EE ED EF DEED EF ED ED ED DE EF ED ED EF DE ED EF ED EF DE EF DE EE ED ED EF DE EF DE ED EF ED EF ED EF ED ED ED ED EF DE EE ED EF DE EF DEFED EF DE EF DE EE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DEF EE DEF ED ED ED ED ED ED ED ED ED EDGE ED EF DE EE ED ED ED ED ED ED ED DE EF DE EF DE EF DEF EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EE DE EE ED ED ED ED DE EF DE EF DE EF DE EF DE EF ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE ED ED ED ED ED ED ED ED ED ED EF DE EE ED ED ED ED ED ED ED ED ED ED ED ED EF DEFED ED ED ED ED ED ED EF DE EF DE ED EF DE ED EF ED ED ED ED ED ED ED EF DE EF DE EF DE EF ED EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED EF DE EF ED EF ED EF ED ED ED ED EF DE EF DE EF ED ED ED ED ED ED ED DE EF DE ED EF DE ED EF DE EE ED ED ED ED ED ED EDGE ED ED EF DE ED EE ED EF DE ED ED ED ED ED ED ED ED ED EF DE EF ED ED ED DE EF DE EF ED EF DE EF DE EF ED EF DE ED ED ED ED ED ED ED DE EE ED ED ED ED ED ED EF DE EF DE EE ED ED ED ED ED ED ED ED ED EF DEF ED ED ED ED ED ED ED ED ED ED EF DEF ED ED DE EF ED ED ED ED ED ED EF DE EF DE EE ED EF ED ED EF DE EF ED EF DE ED EF DE EDED EF DE EF DE EF ED EF DE EF DE EF DE ED EE D EF DE ED EF DE EF DE EE ED EF DE ED EF ED EF DE ED EEED EF ED EF ED EF DE EF DE EF ED EF ED EF DE EE ED EF ED EF DE EE ED ED EF ED ED EF ED ED EED EF ED EF ED ED EF ED EE ED ED EF ED EF DE ED EF ED ED ED ED ED ED EF DE EE ED EF DE ED ED ED ED DE ED EF DE EF DE EF ED EF DE EF DE EF ED ED ED ED EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED EF DE EF ED EF ED EF DE ED ED ED ED ED ED EF DE ED EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EE ED ED ED ED ED ED ED ED ED EF DE ED EF DE EE ED EF DE EF DE EF DE ED EF DE ED EF DE ED EF DE ED EF DE EF DE EF DE EE ED EF DE EF DE EF DE EE ED ED ED ED ED ED ED ED EF DE ED EF DE EF DE EF DE EF DE EF ED ED ED ED EF ED EF ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE ED ED ED ED EF DE ED ED EF DE EF DE ED EF ED ED ED ED ED ED ED ED EF DE EF DE ED EF DE EF DE EF DE ED ED ED EF DE ED EF DE EF DE ED ED EF DE EF DE EF DE ED EF DE ED EF DE ED EF DE EE ED ED ED ED EF DE EF DE ED ED EF DE EF DE EF DE EF DE ED ED ED EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED EF DE EF DE EF DE EE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE ED ED ED ED ED ED ED ED ED ED ED EF DE ED EF DE EF DE EF DE EF DE EF DE EF DE EF DE ED EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED EF DE EF DE EF DE EF DE EF DE ED ED ED ED ED ED EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE ED ED ED ED ED ED ED ED ED ED EF DE EF DE ED EF DE EF DE ED ED ED ED ED ED ED ED EF DE EF DE EE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE ED ED ED EF DE EF ED ED ED ED ED ED ED EF DEF ED EF DE EF DE EF ED EF DE ED ED ED ED ED EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE ED ED ED ED ED ED ED ED ED ED EF DE EE ED EE ED EF DE EE ED ED ED ED ED ED ED EF DE EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED EE ED ED ED ED ED ED ED ED ED ED EF DE EE ED EF DE EE ED ED ED ED ED ED ED ED ED EF DE EE ED EF DE EF ED ED ED ED ED ED EF DE EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF DE ED ED ED ED ED ED ED EF ED EF DE ED EF DE EF DE EE ED ED ED EF DE EE ED ED EF DE EF DE ED EF DE EF DE EF DE EF DE EF DE EE ED ED ED ED ED ED ED ED ED ED ED ED ED EE ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF ED ED ED ED ED ED ED ED ED ED ED ED EF DE EE ED EF DE ED ED ED ED EF DE EE ED EF DE ED EF DE EF ED EF DE EE ED ED EF DE ED EF DE EE ED EF DE ED ED ED ED ED ED ED ED EF DE ED ED EF ED EE ED ED ED ED ED ED ED ED EF DE EE ED ED EF ED ED ED ED ED ED ED ED ED ED EF DE EF ED ED EE ED ED ED ED ED ED ED ED ED EF DE EF ED ED ED ED ED ED ED ED ED ED ED EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED EE ED EF ED ED ED ED EF ED ED EF DE EE ED ED ED ED EE ED EF ED EF DE EE ED ED ED EF DE EE ED EF ED ED EF ED ED ED EE ED ED EF ED EF DE EE ED EE ED ED ED ED ED ED ED ED ED ED EF DE ED EF ED EE ED ED ED ED ED EF DE ED EF ED ED EE ED EE ED ED ED ED ED ED ED ED ED EF DE EE ED EF DE ED EF ED EF DE EE ED ED EE ED EE ED EF DE ED ED EF ED ED ED ED EF ED EF ED EF ED EF ED EF ED ED ED EF ED EF ED EE ED EE ED ED ED ED EF ED EF ED EE ED EE ED EE ED EF ED ED EF ED EF ED ED ED ED ED ED ED ED ED EF DE EE ED EE ED EE ED ED ED ED ED ED ED ED ED EF ED EE ED ED ED ED ED ED ED EF ED EF ED EF EF ED ED ED ED ED ED ED ED EF DE EE ED EF ED EE ED EE ED EF ED EE ED EF ED ED ED ED EF DE ED EF ED EF DE ED ED ED ED ED ED ED ED EF DE ED ED EF ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED EF EF ED ED ED ED ED ED ED EE ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EE ED ED ED ED ED ED ED ED ED ED ED ED EF DE EE ED ED ED EF ED EF ED EF ED ED EF DE EF ED EE ED EF ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EE ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED EF ED EF ED ED EF ED ED ED ED ED ED ED EF ED EF DE ED ED ED ED ED ED ED ED ED ED EF ED EF DE ED EF DE EF ED EF DE EE ED ED EF ED EF DE ED ED ED ED EF DE ED ED ED ED ED ED EF ED ED EF DE ED EF ED EF ED ED EF ED EF ED EF DE EF DE ED ED ED ED ED ED ED ED ED EF DE EE ED EE ED ED EF DE ED ED ED EF DE EF ED EF DE ED EF ED ED ED ED EF ED ED EF ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE EF ED ED EF DE ED ED ED ED ED ED ED ED ED EF DEF ED EF DE ED ED EF DE ED ED DE EF DE EF DE ED ED ED EF DE EF DE EF DE ED ED ED ED ED ED DE EF ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF DE ED EF ED ED EF ED EF ED ED ED EF DE ED ED ED ED ED ED EF DE ED ED ED ED ED ED ED ED EF DE ED EF ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED EF DE ED EF ED ED EF ED ED ED EF ED ED ED EF ED EF ED ED EF ED ED EF DE ED EF ED EF ED ED ED ED EF DE ED ED EF DE ED ED ED ED ED ED ED ED ED ED EF DE EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED EF ED ED ED ED ED EF ED EF ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED DE ED DE ED ED ED ED ED ED ED ED ED ED EF DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED DE ED DE ED ED ED EF DE ED EF ED ED EF ED ED ED ED ED ED ED ED ED ED EF EF ED ED ED ED ED ED DE ED ED ED ED ED ED ED DE ED DE ED ED ED ED ED DE ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED DE ED ED ED ED ED ED ED ED ED ED ED DE ED ED ED ED ED EF ED ED EF ED ED EF ED ED EF ED ED EF ED ED ED DE EF ED ED EF ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED EF ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED EF ED ED EF ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED EF ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED DE ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED EF ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED ED
```
