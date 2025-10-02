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


def load_ff25_data() -> pd.DataFrame:
    """
    Downloads and processes the Fama-French 25 portfolio and risk-free rate data.
    Caches the data to 'processed/ff25_monthly.csv' to avoid repeated downloads.

    Returns
    -------
    pd.DataFrame
        A DataFrame with dates as the index, columns for each of the 25 portfolio
        returns, and a column for the risk-free rate 'rf'.
    """
    # Define file path for cached data
    file_path = PROCESSED_DATA / "ff25_monthly.csv"

    # If file exists, load from it
    if file_path.exists():
        print("Loading cached Fama-French 25 data...")
        ff_data = pd.read_csv(file_path, index_col='date', parse_dates=['date'])
        return ff_data

    # If file doesn't exist, download from WRDS
    print("Connecting to WRDS to download Fama-French data...")
    db = wrds.Connection(wrds_username="travisj")

    # Download Fama-French 25 Portfolio Returns
    portfolio_cols = [f's{s}b{b}_vwret' for s in range(1, 6) for b in range(1, 6)]
    ff_ports = db.get_table(
        library='ff',
        table='portfolios25',
        obs=1000000,
        columns=['date'] + portfolio_cols
    )

    # Download Risk-Free Rate
    ff_factors = db.get_table(
        library='ff',
        table='factors_monthly',
        obs=1000000,
        columns=['date', 'rf']
    )
    db.close()
    print("Data download complete.")

    # Data Cleaning and Merging
    # Convert date columns to datetime objects.
    ff_ports['date'] = pd.to_datetime(ff_ports['date'])
    ff_factors['date'] = pd.to_datetime(ff_factors['date'])

    # Normalize dates to month-end to ensure proper alignment for the join.
    # This is a common source of errors when merging financial time series.
    ff_ports['date'] = ff_ports['date'] + pd.offsets.MonthEnd(0)
    ff_factors['date'] = ff_factors['date'] + pd.offsets.MonthEnd(0)

    # Set date as the index for both DataFrames.
    ff_ports = ff_ports.set_index('date')
    ff_factors = ff_factors.set_index('date')
    merged_data = ff_ports.join(ff_factors['rf'], how='inner')

    # Save the merged data to the cache file
    merged_data.to_csv(file_path)
    print(f"Fama-French 25 data cached to {file_path}")

    return merged_data


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
    'load_market_returns', 'load_ff25_data', 'load_consumption_and_rf'
]
