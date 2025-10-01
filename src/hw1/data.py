"""Data loading and preparation utilities (placeholders).

In practice these would pull from WRDS/CSV/Excel. For now they provide
interfaces and simple synthetic data generators for testing pipeline code.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import wrds
from .paths import RAW_DATA, PROCESSED_DATA


def load_market_returns(kind: str = "value", freq: str = "D", n: int = 800) -> pd.Series:
    """Return synthetic market returns series for scaffolding.

    Parameters
    ----------
    kind : 'value' or 'equal'
    freq : pandas frequency alias
    n : length of series
    """

    # Define file path for cached data
    file_path = PROCESSED_DATA / "market_returns.csv"

    # If file exists, load from it
    if file_path.exists():
        market_data = pd.read_csv(file_path, index_col='date', parse_dates=['date'])
    else:
        # If file doesn't exist, download from WRDS
        db = wrds.Connection(wrds_username="travisj")
        market_data = db.get_table(library='crsp', table='dsi', columns=['date', 'vwretd', 'ewretd'])
        db.close()

        # Set date as index and save to file
        market_data = market_data.set_index('date')
        market_data.to_csv(file_path)

    # Select the correct return series
    if kind == "value":
        return market_data['vwretd']
    elif kind == "equal":
        return market_data['ewretd']
    else:
        raise ValueError("kind must be 'value' or 'equal'")


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
