"""Problem 2: Return forecasting with penalized regressions and a simple NN.

This module scaffolds feature engineering and model fitting for daily market
return prediction using past lags and aggregated windows.

Feature spec (linear models):
- r_{t-1}, r_{t-2}, ..., r_{t-5}
- Sum of returns over windows: 10, 21, 42, 63, 84, 105, 126, ..., 252 (step 21 after 126)

Neural network (placeholder): all 252 individual lags.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .ml import (fit_lasso, fit_ridge, fit_elastic_net, fit_mlp, SKLEARN_AVAILABLE, ModelResult)

__all__ = [
    'build_linear_feature_matrix', 'build_nn_feature_matrix', 'run_linear_models', 'run_nn_model'
]

LINEAR_LAG_WINDOWS = [1, 2, 3, 4, 5]
AGG_WINDOWS = [10, 21, 42, 63, 84, 105, 126] + list(range(147, 253, 21))


def build_linear_feature_matrix(returns: pd.Series) -> pd.DataFrame:
    """Build feature matrix for linear penalized models.

    Returns DataFrame with columns: lag_1..lag_5, sum_10, sum_21, ..., and target y.
    Observations with insufficient history are dropped.
    """
    r = returns.sort_index().rename('r')
    df = pd.DataFrame({'y': r})
    for lag in LINEAR_LAG_WINDOWS:
        df[f'lag_{lag}'] = r.shift(lag)
    for w in AGG_WINDOWS:
        df[f'sum_{w}'] = r.rolling(w).sum().shift(1)  # exclude current day
    df = df.dropna()
    cols = [c for c in df.columns if c != 'y'] + ['y']
    return df[cols]


def build_nn_feature_matrix(returns: pd.Series, max_lag: int = 252) -> pd.DataFrame:
    r = returns.sort_index().rename('r')
    df = pd.DataFrame({'y': r})
    for lag in range(1, max_lag + 1):
        df[f'lag_{lag}'] = r.shift(lag)
    df = df.dropna()
    cols = [c for c in df.columns if c != 'y'] + ['y']
    return df[cols]


def run_linear_models(feature_df: pd.DataFrame) -> list[ModelResult]:
    results: list[ModelResult] = []
    # Simple fixed hyperparameters for scaffold; real workflow would CV tune
    try:
        results.append(fit_lasso(feature_df, target_col='y', alpha=0.0005))
        results.append(fit_ridge(feature_df, target_col='y', alpha=5.0))
        results.append(fit_elastic_net(feature_df, target_col='y', alpha=0.0005, l1_ratio=0.3))
    except ImportError:  # scikit-learn missing
        pass
    return results


def run_nn_model(feature_df: pd.DataFrame) -> ModelResult | None:
    if not SKLEARN_AVAILABLE:
        return None
    # Basic small network; tuning left for future refinement
    return fit_mlp(feature_df, target_col='y', hidden_layer_sizes=(64,), max_iter=150)

