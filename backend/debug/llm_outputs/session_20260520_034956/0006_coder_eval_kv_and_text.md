# Call 0006 — `coder_eval` (kv_and_text)

## Meta

- ts: 2026-05-20 03:59:19
- conv_id: `d747d835`
- step: 0
- temperature: 0.8
- has_past_kv: True
- input_tokens: 3009
- output_tokens: 1
- duration_s: 1.3015
- text_len: 0

## Variables (dari YAML placeholder)

- **scenario** (1169 chars): <scenario_background> ⏎ The factor is a characteristic or variable used in quant investment that can help explain the returns and risks of a portfolio or a single asset. Factors are used by investors to…
- **factor_information** (274 chars): factor_name: GAP_REVERSAL_RANK ⏎ factor_description: Ranks stocks based on inverted gaps and subsequent price reversal ⏎ factor_formulation: RANK( ( $gap == -1 ) && ( $reversal > 0.05 ) ) ⏎ variables: {'$ga…
- **code** (1755 chars): File: factor.py ⏎ import pandas as pd ⏎ import numpy as np ⏎ import os ⏎ from factors.coder.expr_parser import parse_expression, parse_symbol ⏎ from factors.coder.function_lib import * ⏎  ⏎  ⏎ def calculate_factor(ex…
- **execution_feedback** (460 chars): AST Regularization Check Passed ⏎  ⏎ factor_expression:  RANK( ( $gap == -1 ) && ( $reversal > 0.05 ) ) ⏎ Traceback (most recent call last): ⏎   File "/path/to/factor.py", line 42, in <module> ⏎     calculate_f…
- **value_feedback** (49 chars): No factor value generated, skip value evaluation.

## System Prompt

```text
Role: Expression Reviewer Agent

Mission
Critique a quantitative factor expression for correctness. Your critique will be sent directly to a Coder agent to fix the expression — NOT shown to the user.

Review Protocol — check in this order:
1. Execution error (if any): read the execution log first, identify the exact root cause (undefined variable, wrong function name, wrong argument count, syntax error).
2. Semantic alignment: does the expression capture the economic meaning of the factor description?
   - Minor discrepancies in window size or non-core elements are acceptable.
   - Critical misalignments: wrong signal direction, wrong variable, fundamentally wrong aggregation.
3. Expression validity: all functions and variables must be from the allowed list.

### Response Rules:
- Point out ONLY critical issues — ignore cosmetic or non-critical discrepancies.
- Do NOT suggest code fixes — only describe WHAT is wrong in the expression logic.
- Do NOT include code snippets in your response.
- If no critical issue found: respond exactly "No comment found" with no other text.
- Format: "comment N: <message>" per issue, one per line.

<scenario>
<<<scenario>>>
<scenario_background>
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
</scenario_background>
<<</scenario>>>
</scenario>

Output Contract — STRICT:
- The ENTIRE response must consist of EITHER:
    * the literal string `No comment found` (nothing else), OR
    * one or more lines, each starting with the literal token `comment ` followed by a number and a colon (e.g. `comment 1: ...`).
- No JSON, no markdown, no headings, no preamble, no other schema. Any deviation will be discarded.

**Only the following operations are allowed in expression:**
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
- **SCALE(A, target_sum)**: Scale the absolute values in the cross-section to sum to target_sum.

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



Note:
- Only the variables in data (e.g., `$open`) and arithmetic/logical operators (+, -, *, /, &&, ||) are allowed.
- No undeclared variables (e.g., `n`, `w_1`) or undefined symbols (e.g., `=`) in the expression.
- TS-prefix functions (e.g., `TS_STD()`) and non-TS functions (e.g., `STD()`) are different — check carefully.
```

## User Prompt

```text
<factor_information>
<<<factor_information>>>
factor_name: GAP_REVERSAL_RANK
factor_description: Ranks stocks based on inverted gaps and subsequent price reversal
factor_formulation: RANK( ( $gap == -1 ) && ( $reversal > 0.05 ) )
variables: {'$gap': 'SIGN($open - $close)', '$reversal': 'TS_PCTCHANGE($return', '5)': ''}
<<</factor_information>>>
</factor_information>

<expression_code>
<<<code>>>
File: factor.py
import pandas as pd
import numpy as np
import os
from factors.coder.expr_parser import parse_expression, parse_symbol
from factors.coder.function_lib import *


def calculate_factor(expr: str, name: str):
    # stock dataframe
    df = pd.read_hdf('./daily_pv.h5', key='data')

    # daily_pv.h5 sudah berisi $return (ditambahkan generate.py via
    # pct_change $close). Blok ini hanya fallback defensif — kalau suatu
    # saat file dipasok tanpa kolom $return, derive di sini supaya
    # ekspresi berbasis $return tetap jalan. groupby instrument supaya
    # pct_change tidak bocor antar simbol di MultiIndex (instrument, datetime).
    if '$return' not in df.columns:
        idx_names = df.index.names
        if idx_names and 'instrument' in idx_names:
            df['$return'] = df.groupby(level='instrument')['$close'].pct_change(fill_method=None)
        else:
            df['$return'] = df['$close'].pct_change(fill_method=None)

    expr = parse_symbol(expr, df.columns)
    expr = parse_expression(expr)

    # replace '$var' by 'df['var'] to extract var's actual values
    for col in df.columns:
        expr = expr.replace(col[1:], f"df[\'{col}\']")

    df[name] = eval(expr)
    result = df[name].astype(np.float64)

    if os.path.exists('result.h5'):
        os.remove('result.h5')
    result.to_hdf('result.h5', key='data')

if __name__ == '__main__':
    # Input factor expression. Do NOT use the variable format like "df['$xxx']" in factor expressions. Instead, you should use "$xxx". 
    expr = "RANK( ( $gap == -1 ) && ( $reversal > 0.05 ) )" # Your output factor expression will be filled in here
    name = "GAP_REVERSAL_RANK" # Your output factor name will be filled in here
    calculate_factor(expr, name)

<<</code>>>
</expression_code>


<execution_log>
<<<execution_feedback>>>
AST Regularization Check Passed

factor_expression:  RANK( ( $gap == -1 ) && ( $reversal > 0.05 ) )
Traceback (most recent call last):
  File "/path/to/factor.py", line 42, in <module>
    calculate_factor(expr, name)
  File "/path/to/factor.py", line 31, in calculate_factor
    df[name] = eval(expr)
  File "<string>", line 1
    RANK(AND((EQ($gap, -1)), (GT($reversal, 0.05))))
                 ^
SyntaxError: invalid syntax

Expected output file not found.
<<</execution_feedback>>>
</execution_log>



<value_feedback>
<<<value_feedback>>>
No factor value generated, skip value evaluation.
<<</value_feedback>>>
</value_feedback>

```

## Response

```text

```
