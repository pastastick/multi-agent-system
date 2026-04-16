import pandas as pd
import numpy as np
import os
from factors.coder.expr_parser import parse_expression, parse_symbol
from factors.coder.function_lib import *


def calculate_factor(expr: str, name: str):
    # stock dataframe
    df = pd.read_hdf('./daily_pv.h5', key='data')
    
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
    expr = "TS_MEAN($return, 5) / (TS_STD($return, 30) + 1e-8) * TS_PCTCHANGE($return, 7)" # Your output factor expression will be filled in here
    name = "Short_Term_Momentum_Volatility_Factor_5_30_7" # Your output factor name will be filled in here
    calculate_factor(expr, name)