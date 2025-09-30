"""Consumption data related helpers (placeholder).

Centralizes any transformations for consumption growth series used in Problems 4 and 5.
"""
from __future__ import annotations
import pandas as pd

__all__ = ["real_growth_rate"]

def real_growth_rate(consumption: pd.Series) -> pd.Series:
    """Return (C_t / C_{t-1} - 1) for a level consumption series.

    If data are already growth rates, user should skip this.
    """
    return consumption.pct_change().dropna()

