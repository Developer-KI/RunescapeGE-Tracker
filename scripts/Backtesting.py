import pandas as pd
import numpy as np

def PNL(value_matrix: pd.Series) -> float:
    first_value = value_matrix.iloc[0]
    last_value = value_matrix.iloc[-1]
    return (last_value - first_value) / first_value

def MaxDrawdown(value_matrix: pd.Series) -> float:
    first_value = value_matrix.iloc[0]
    lowest_value = value_matrix.min()
    return (first_value - lowest_value) / first_value

def Sharpe(value_matrix: pd.Series, risk_free_rate: float = 0) -> float:
    returns = value_matrix.pct_change()
    port_risk = returns.std()
    excess_return = returns - risk_free_rate
    return np.mean(excess_return) / port_risk

def Sortino(value_matrix: pd.Series, risk_free_rate: float = 0) -> float: 
    returns = value_matrix.pct_change()
    downside_returns = returns[returns < risk_free_rate]
    downside_risk = np.std(downside_returns)
    excess_return = returns - risk_free_rate
    return np.mean(excess_return) / downside_risk
