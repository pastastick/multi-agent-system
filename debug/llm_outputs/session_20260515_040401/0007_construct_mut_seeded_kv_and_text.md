# Call 0007 — `construct_mut_seeded` (kv_and_text)

## Meta

- ts: 2026-05-15 04:04:53
- conv_id: `139296c6`
- step: 0
- temperature: 0.7
- has_past_kv: True
- input_tokens: 3190
- output_tokens: 320
- duration_s: 16.4987
- text_len: 964

## System Prompt

```text
You are the **Expression agent** — translate a hypothesis into symbolic factor expressions.
Domain context is in latent KV representation.

<scenario>
Background of the scenario:
The factor is a characteristic or variable used in quant investment that can help explain the returns and risks of a portfolio or a single asset. Factors are used by investors to identify and exploit sources of excess returns, and they are central to many quantitative investment strategies.
Each number in the factor represents a physics value to an instrument on a day.
User will train a model to predict the next several days return based on the factor values of the previous days.
The factor is defined in the following parts:
1. Name: The name of the factor.
2. Description: The description of the factor.
3. Formulation: The formulation of the factor.
4. Variables: The variables or functions used in the formulation of the factor.
The factor might not provide all the parts of the information above since some might not be applicable.
Please specifically give all the hyperparameter in the factors like the window size, look back period, and so on. One factor should statically defines one output with a static source data. For example, last 10 days momentum and last 20 days momentum should be two different factors.


====== Runtime Environment ======
You have following environment to run the code:
Python 3.10, venv-managed, pandas, numpy, qlib.


Strategy context:
Ensure that for every step of data processing, the data format (including indexes) is clearly explained through comments.
Each transformation or calculation should be accompanied by a detailed description of how the data is structured, especially focusing on key aspects like whether the data has multi-level indexing, how to access specific columns or index levels, and any operations that affect the data shape (e.g., `reset_index()`, `groupby()`, `merge()`).
This step-by-step explanation will ensure clarity and accuracy in data handling. For example:
1. **Start with multi-level index**:  
  ```python
  # The initial DataFrame has a multi-level index with 'datetime' and 'instrument'.
  # To access the 'datetime' index, use df.index.get_level_values('datetime').
  datetime_values = df.index.get_level_values('datetime')
  ```

2. **Reset the index if necessary**:  
  ```python
  # Resetting the index to move 'datetime' and 'instrument' from the index to columns.
  # This operation flattens the multi-index structure.
  df = df.reset_index()
  ```

3. **Perform groupby operations**:  
  ```python
  # Grouping by 'datetime' and 'instrument' to aggregate the data.
  # After groupby, the result will maintain 'datetime' and 'instrument' as a multi-level index.
  df_grouped = df.groupby(['datetime', 'instrument']).sum()
  ```

4. **Ensure consistent datetime formats**:  
  ```python
  # Before merging, ensure that the 'datetime' column in both DataFrames is of the same format.
  # Convert to datetime format if necessary.
  df['datetime'] = pd.to_datetime(df['datetime'])
  other_df['datetime'] = pd.to_datetime(other_df['datetime'])
  ```

5. **Merge operations**:  
  ```python
  # When merging DataFrames, ensure you are merging on both 'datetime' and 'instrument'.
  # If these are part of the index, reset the index before merging.
  merged_df = pd.merge(df, other_df, on=['datetime', 'instrument'], how='inner')
  ```

Experiment setting:
| Dataset 📊 | Model 🤖    | Factors 🌟       | Data Split  🧮                                   |
|---------|----------|---------------|-------------------------------------------------|
| CSI300  | LGBModel | Alpha158 Plus | Train: 2008-01-01 to 2014-12-31 <br> Valid: 2015-01-01 to 2016-12-31 <br> Test &nbsp;: 2017-01-01 to 2020-08-01 |
</scenario>

Hard limits (exceeding any = discard): SL≤250 chars, ER≤6 distinct features, PC≤50% free params.
Prior complexity warnings → this round must be fundamentally simpler.

Output ONLY the JSON below. No markdown, no commentary.
Output ONLY this JSON. No markdown, no commentary.
{"<factor_name>": {"description": "...", "variables": {"$sym": "meaning"}, "formulation": "LaTeX", "expression": "..."}}

Expression syntax examples (mean-reversion / volume / volatility-momentum):
"($close - TS_MEAN($close, 5)) / (TS_MEAN($close, 5) + 1e-8)"
"RANK(TS_STD($volume, 10) / (TS_MEAN($volume, 20) + 1e-8))"
"RANK(($high - $low) / ($close + 1e-8)) * SIGN(TS_MEAN($return, 5))"
```

## User Prompt

```text
<target_hypothesis>

</target_hypothesis>


<history>

<round id="1">
  <hypothesis>Hypothesis: Simple 10-day price momentum predicts forward return.</hypothesis>
  <observations>IC = 0.012, very low. Max drawdown 15%.</observations>
  <evaluation>Momentum alone is insufficient; volume confirmation is needed.</evaluation>
  <next_direction>Combine momentum with volume growth.</next_direction>
  <reasoning>Volume adds conviction context to the price move.</reasoning>
  <improved_sota>False</improved_sota>
</round>

<round id="2">
  <hypothesis>Hypothesis: Price momentum × volume confirmation predicts 5-day forward return.</hypothesis>
  <observations>IC = 0.035, improved over baseline. IR is still 0.4.</observations>
  <evaluation>Direction is correct, but the factor structure can be refined.</evaluation>
  <next_direction>Add ZSCORE normalization on volume and extend the window.</next_direction>
  <reasoning>Normalization reduces noise; longer windows improve stability.</reasoning>
  <improved_sota>True</improved_sota>
</round>

</history>
Note: rounds older than 2 are preserved in the latent KV-cache.


<allowed_variables>
All daily-level: $open, $close, $high, $low, $volume, $return
</allowed_variables>

<allowed_functions>
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
</allowed_functions>



Rules: use returns/ratios (not raw price), add 1e-8 to denominators, use RANK/ZSCORE for cross-sectional normalization. Factors must be independent.
Generate 2-3 new factor. Strictly follow allowed functions and variables. Now output the JSON.
```

## Response

```text
{
  "Momentum_Volume_Confirm_5D": {
    "description": "5-day momentum rank multiplied by sign of 5-day volume growth.",
    "variables": {
      "$sym": "momentum_rank * volume_growth_sign"
    },
    "formulation": "RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))",
    "expression": "RANK(TS_MEAN($return, 5)) * SIGN(TS_PCTCHANGE($volume, 5))"
  },
  "Volume_Imbalance_ZScore": {
    "description": "Z-score of volume imbalance over 20 days.",
    "variables": {
      "$sym": "volume_imbalance_zscore"
    },
    "formulation": "ZSCORE(TS_MEAN($volume - $open, 20))",
    "expression": "ZSCORE(TS_MEAN($volume - $open, 20))"
  },
  "Volatility_Regime_Momentum": {
    "description": "Momentum in high-volatility regimes.",
    "variables": {
      "$sym": "volatility_regime_momentum"
    },
    "formulation": "TS_MEAN($return, 5) * ZSCORE(TS_STD($high - $low, 20))",
    "expression": "TS_MEAN($return, 5) * ZSCORE(TS_STD($high - $low, 20))"
  }
}
```
