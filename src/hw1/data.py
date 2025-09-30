"""Data loading and preparation utilities (placeholders).

In practice these would pull from WRDS/CSV/Excel. For now they provide
interfaces and simple synthetic data generators for testing pipeline code.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .paths import RAW_DATA, PROCESSED_DATA


def load_market_returns(kind: str = "value", freq: str = "D", n: int = 800) -> pd.Series:
    """Return synthetic market returns series for scaffolding.

    Parameters
    ----------
    kind : 'value' or 'equal'
    freq : pandas frequency alias
    n : length of series
    """
    rng = np.random.default_rng(0 if kind == "value" else 1)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq=freq)
    # Add tiny AR(1) structure for realism
    eps = rng.standard_normal(n) * 0.01
    r = np.zeros(n)
    phi = 0.05 if kind == "value" else 0.03
    for t in range(1, n):
        r[t] = phi * r[t-1] + eps[t]
    return pd.Series(r, index=dates, name=f"{kind}_weighted_return")


def load_ff25_monthly(n: int = 120) -> pd.DataFrame:
    """Synthetic Fama-French 25 portfolio returns (monthly) for scaffolding."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().to_period('M').to_timestamp(how='end'), periods=n, freq='M')
    base = rng.multivariate_normal(mean=np.zeros(25), cov=0.0025*np.eye(25), size=n)
    cols = [f"P{(i//5)+1}{(i%5)+1}" for i in range(25)]
    return pd.DataFrame(base, index=dates, columns=cols)


def load_consumption_and_rf(n: int = 160) -> pd.DataFrame:
    """Synthetic quarterly consumption growth and risk-free returns.

    Columns: cons_growth, Rm, Rf (all simple returns for scaffolding)
    """
    rng = np.random.default_rng(7)
    dates = pd.period_range(end=pd.Timestamp.today().to_period('Q'), periods=n, freq='Q').to_timestamp(how='end')
    cons_g = rng.normal(0.005, 0.01, size=n)  # consumption growth
    rm = rng.normal(0.015, 0.07, size=n)
    rf = rng.normal(0.0025, 0.001, size=n)
    return pd.DataFrame({'cons_growth': cons_g, 'Rm': rm, 'Rf': rf}, index=dates)

__all__ = [
    'load_market_returns', 'load_ff25_monthly', 'load_consumption_and_rf'
]

