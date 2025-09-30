"""Portfolio analytics utilities (placeholders).

Provides simple mean/variance-based portfolio construction helpers used across
Problems 3–5. Real implementation would incorporate estimation risk handling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = [
    'annualize_mean', 'annualize_vol', 'sharpe_ratio', 'tangency_portfolio'
]

def annualize_mean(r: pd.Series | pd.DataFrame, periods_per_year: int) -> pd.Series:
    m = r.mean() * periods_per_year
    return m

def annualize_vol(r: pd.Series | pd.DataFrame, periods_per_year: int) -> pd.Series:
    v = r.std(ddof=1) * np.sqrt(periods_per_year)
    return v

def sharpe_ratio(r: pd.Series | pd.DataFrame, rf: float = 0.0, periods_per_year: int = 12) -> pd.Series:
    excess = r.mean() - rf / periods_per_year
    vol = r.std(ddof=1)
    return np.sqrt(periods_per_year) * excess / vol

def tangency_portfolio(returns: pd.DataFrame, rf: float = 0.0) -> dict:
    """Compute tangency portfolio weights using sample mean & covariance.

    Parameters
    ----------
    returns : DataFrame (T x N)
    rf : float, annual risk-free (simple) assumed constant (placeholder)

    Returns
    -------
    dict with weights (Series), mean, stdev, sharpe.
    """
    mu = returns.mean()
    Sigma = returns.cov()
    ones = np.ones(len(mu))
    Sigma_inv = np.linalg.pinv(Sigma.values)
    excess_mu = mu.values - rf / 12.0  # crude monthly adjustment
    w_unscaled = Sigma_inv @ excess_mu
    w = w_unscaled / w_unscaled.sum()
    w_series = pd.Series(w, index=returns.columns, name='weight')
    port_ret = (returns @ w)
    out = {
        'weights': w_series,
        'mean': float(port_ret.mean()),
        'stdev': float(port_ret.std(ddof=1)),
        'sharpe': float(sharpe_ratio(port_ret.to_frame('p'), rf=rf, periods_per_year=12).iloc[0])
    }
    return out

