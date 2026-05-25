# Call 0008 — `construct` (kv_and_text)

## Meta

- ts: 2026-05-25 08:10:51
- conv_id: `3bcab616`
- step: 0
- temperature: 0.7
- has_past_kv: True
- input_tokens: 2260
- output_tokens: 90
- duration_s: 4.7632
- text_len: 267

## Variables (dari YAML placeholder)

- **experiment_output_format** (1223 chars): Write 2-3 factors. For EACH factor write exactly three lines, in this order: ⏎  ⏎ NAME: <short factor name, letters/digits/underscore only, no spaces> ⏎ DESC: <one sentence — what the factor measures> ⏎ EXPR:…
- **target_hypothesis_oneline** (101 chars): High-frequency order book imbalance in the pre-market leads to exaggerated intraday price volatility.
- **function_lib_description** (6332 chars): Only the following operations are allowed in expressions: ⏎ ### **Cross-sectional Functions** ⏎ - **RANK(A)**: Ranking of each element in the cross-sectional dimension of A. ⏎ - **ZSCORE(A)**: Z-score of ea…
- **targets** (7 chars): factors

## System Prompt

```text
You are translating a quantitative alpha hypothesis (already in the prior context) into 2-3 symbolic factor expressions for backtesting.

Expression rules (downstream will discard violators):
- Use ONLY $open, $close, $high, $low, $volume, $return and the operators listed in the user message
- Length ≤ 250 chars per expression; ≤ 6 distinct base features
- Free parameters ≤ 50% of nodes; time windows 1–60 days (nested totals ≤ 60)

The 2-3 factors must vary structurally — different operator families (e.g. one correlation-based, one conditional, one regression). Prefer simple compositions; if prior complexity warnings appear in context, choose a simpler structure than before.

Output format — follow it exactly:
<<<experiment_output_format>>>
Write 2-3 factors. For EACH factor write exactly three lines, in this order:

NAME: <short factor name, letters/digits/underscore only, no spaces>
DESC: <one sentence — what the factor measures>
EXPR: <the factor expression on one line>

Separate factors with one blank line. Output ONLY these lines — no JSON, no
markdown, no commentary. EXPR must use only $open/$close/$high/$low/$volume/$return
and the allowed operators.

Composition guideline (abstract — invent your own operators, variables, and windows):
- A single function applied to one variable is almost always too weak. Combine at least two operators that reinforce each other.
- Typical compositions:
    * <cross_sectional_op>( <time_series_op>(<variable>, <window>) )
    * <time_series_op_A>(<variable>, <window>) <arith> <time_series_op_B>(<variable>, <window>)
    * <aggregator>( <transform_A>(<variable_X>), <transform_B>(<variable_Y>), <window> )
- Free to mix cross-sectional, time-series, conditional/logical, regression, and smoothing operators in any combination — as long as Expression Rules above are satisfied.
- Do NOT copy the structural placeholders literally; replace them with concrete operators that best capture the hypothesis mechanism.
<<</experiment_output_format>>>
```

## User Prompt

```text
Hypothesis to implement:
<<<target_hypothesis_oneline>>>
High-frequency order book imbalance in the pre-market leads to exaggerated intraday price volatility.
<<</target_hypothesis_oneline>>>

Allowed operators:
<<<function_lib_description>>>
Only the following operations are allowed in expressions:
### **Cross-sectional Functions**
- **RANK(A)**: Ranking of each element in the cross-sectional dimension of A.
- **ZSCORE(A)**: Z-score of each element in the cross-sectional dimension of A.
- **MEAN(A)**: Mean value of each element in the cross-sectional dimension of A.
- **STD(A)**: Standard deviation in the cross-sectional dimension of A.
- **SKEW(A)**: Skewness in the cross-sectional dimension of A.
- **KURT(A)**: Kurtosis in the cross-sectional dimension of A.
- **MAX(A)**: Maximum value in the cross-sectional dimension of A.
- **MIN(A)**: Minimum value in the cross-sectional dimension of A.
- **MEDIAN(A)**: Median value in the cross-sectional dimension of A

### **Time-Series Functions**
- **DELTA(A, n)**: Change in value of A over n periods.
- **DELAY(A, n)**: Value of A delayed by n periods.
- **TS_MEAN(A, n)**: Mean value of sequence A over the past n days.
- **TS_SUM(A, n)**: Sum of sequence A over the past n days.
- **TS_RANK(A, n)**: Time-series rank of the last value of A in the past n days.
- **TS_ZSCORE(A, n)**: Z-score for each sequence in A over the past n days.
- **TS_MEDIAN(A, n)**: Median value of sequence A over the past n days.
- **TS_PCTCHANGE(A, p)**: Percentage change in the value of sequence A over p periods.
- **TS_MIN(A, n)**: Minimum value of A in the past n days.
- **TS_MAX(A, n)**: Maximum value of A in the past n days.
- **TS_ARGMAX(A, n)**: The index (relative to the current time) of the maximum value of A over the past n days.
- **TS_ARGMIN(A, n)**: The index (relative to the current time) of the minimum value of A over the past n days.
- **TS_QUANTILE(A, p, q)**: Rolling quantile of sequence A over the past p periods, where q is the quantile value between 0 and 1.
- **TS_STD(A, n)**: Standard deviation of sequence A over the past n days.
- **TS_VAR(A, p)**: Rolling variance of sequence A over the past p periods.
- **TS_CORR(A, B, n)**: Correlation coefficient between sequences A and B over the past n days.
- **TS_COVARIANCE(A, B, n)**: Covariance between sequences A and B over the past n days.
- **TS_MAD(A, n)**: Rolling Median Absolute Deviation of sequence A over the past n days.
- **PERCENTILE(A, q, p)**: Quantile of sequence A, where q is the quantile value between 0 and 1. If p is provided, it calculates the rolling quantile over the past p periods.
- **HIGHDAY(A, n)**: Number of days since the highest value of A in the past n days.
- **LOWDAY(A, n)**: Number of days since the lowest value of A in the past n days.
- **SUMAC(A, n)**: Cumulative sum of A over the past n days.

### **Moving Averages and Smoothing Functions**
- **SMA(A, n, m)**: Simple moving average of A over n periods with modifier m.
- **WMA(A, n)**: Weighted moving average of A over n periods, with weights decreasing from 0.9 to 0.9^(n).
- **EMA(A, n)**: Exponential moving average of A over n periods, where the decay factor is 2/(n+1).
- **DECAYLINEAR(A, d)**: Linearly weighted moving average of A over d periods, with weights increasing from 1 to d.

### **Mathematical Operations**
- **PROD(A, n)**: Product of values in A over the past n days. Use `*` for general multiplication.
- **LOG(A)**: Natural logarithm of each element in A.
- **SQRT(A)**: Square root of each element in A.
- **POW(A, n)**: Raise each element in A to the power of n.
- **SIGN(A)**: Sign of each element in A, one of 1, 0, or -1.
- **EXP(A)**: Exponential of each element in A.
- **ABS(A)**: Absolute value of A.
- **MAX(A, B)**: Maximum value between A and B.
- **MIN(A, B)**: Minimum value between A and B.
- **INV(A)**: Reciprocal (1/x) of each element in sequence A.
- **FLOOR(A)**: Floor of each element in sequence A.

### **Conditional and Logical Functions**
- **COUNT(C, n)**: Count of samples satisfying condition C in the past n periods. Here, C is a logical expression, e.g., `$close > $open`.
- **SUMIF(A, n, C)**: Sum of A over the past n periods if condition C is met. Here, C is a logical expression.
- **FILTER(A, C)**: Filtering multi-column sequence A based on condition C. Here, C is presented in a logical expression form, with the same size as A.
- **(C1)&&(C2)**: Logical operation "and". Both C1 and C2 are logical expressions, such as A > B.
- **(C1)||(C2)**: Logical operation "or". Both C1 and C2 are logical expressions, such as A > B.
- **(C1)?(A):(B)**: Logical operation "If condition C1 holds, then A, otherwise B". C1 is a logical expression, such as A > B.

### **Regression and Residual Functions**
- **SEQUENCE(n)**: A single-column sequence of length n, ranging from 1 to integer n. `SEQUENCE()` should always be nested in `REGBETA()` or `REGRESI()` as argument B.
- **REGBETA(A, B, n)**: Regression coefficient of A on B using the past n samples, where A MUST be a multi-column sequence and B a single-column or multi-column sequence.
- **REGRESI(A, B, n)**: Residual of regression of A on B using the past n samples, where A MUST be a multi-column sequence and B a single-column or multi-column sequence.

### **Technical Indicators**
- **RSI(A, n)**: Relative Strength Index of sequence A over n periods. Measures momentum by comparing the magnitude of recent gains to recent losses.
- **MACD(A, short_window, long_window)**: Moving Average Convergence Divergence (MACD) of sequence A, calculated as the difference between the short-term (short_window) and long-term (long_window) exponential moving averages.
- **BB_MIDDLE(A, n)**: Middle Bollinger Band, calculated as the n-period simple moving average of sequence A.
- **BB_UPPER(A, n)**: Upper Bollinger Band, calculated as middle band plus two standard deviations of sequence A over n periods.
- **BB_LOWER(A, n)**: Lower Bollinger Band, calculated as middle band minus two standard deviations of sequence A over n periods.

Notes:
- Only the variables provided in data (e.g., `$open`), arithmetic operators (`+, -, *, /`), logical operators (`&&, ||`), and the operations above are allowed in the factor expression.
- Make sure each factor expression contains at least one variable (e.g. $open) combined with registered operations above. Do NOT use any undeclared variable (e.g. 'n', 'w_1') or undefined symbols (e.g., '=').
- Pay attention to the distinction between TS-prefix operations (e.g., `TS_STD()`) and non-TS ones (e.g., `STD()`).
<<</function_lib_description>>>


Write 2-3 independent <<<targets>>>
factors
<<</targets>>> that implement the hypothesis above. Each factor MUST structurally encode the specific mechanism described — do NOT produce trivial single-function wrappers like RANK($volume). Pick operator families that capture the causal chain: correlation, conditional, regression, count-based, smoothing, or combinations. Now output the factors in the required format.
```

## Response

```text
NAME: PREV_VOL_RANK
DESC: Rank of volume in the pre-market period
EXPR: RANK($volume, 10)

NAME: HIGHLOW_RANGE
DESC: Intraday range relative to close
EXPR: ($high - $low)/$close

NAME: VOL_ATR
DESC: Volume relative to ATR volatility
EXPR: $volume / TS_STD($close, 20)
```
