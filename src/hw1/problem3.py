"""Problem 3: Portfolio construction and asset pricing relations (scaffold).

Implements placeholder logic to:
- Extract four extreme Fama-French 25 portfolios (corners of size/BM grid)
- Compute tangency (maximum Sharpe) portfolio among them
- Verify linear pricing relation with tangency portfolio as factor
- Compute log-utility optimal weights across the four
- Compute tangency portfolio across all 25 assets

All functions use synthetic FF25 data produced by data.load_ff25_monthly unless
real data are supplied. Numerical routines are intentionally simple.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Any
from .portfolio import tangency_portfolio, sharpe_ratio

__all__ = [
    'select_extremes', 'pricing_relation', 'log_utility_optimal_weights', 'analyze_problem3'
]

@dataclass
class PricingFit:
    beta: float
    alpha: float
    r2: float


def select_extremes(ff25: pd.DataFrame) -> pd.DataFrame:
    """Select four 'extreme' portfolios.

    Placeholder logic: assume columns ordered such that (row groups=Size buckets, col groups=BM buckets)
    and corners correspond to first, fifth, twenty-first, twenty-fifth columns.
    Adjust this when mapping real FF25 naming.
    """
    cols = ff25.columns
    extreme_cols = [cols[0], cols[4], cols[20], cols[24]]  # corners
    return ff25[extreme_cols]


def pricing_relation(assets: pd.DataFrame, factor_returns: pd.Series, rf: float = 0.0) -> Dict[str, PricingFit]:
    """Run time-series regression r_i - rf = alpha + beta (f - rf) with OLS for each asset.
    Returns dict of PricingFit.
    """
    out: Dict[str, PricingFit] = {}
    f_excess = factor_returns - rf/12.0

    for col in assets.columns:
        y = assets[col] - rf/12.0
        X = np.vstack([np.ones(len(y)), f_excess.values]).T
        beta_hat = np.linalg.lstsq(X, y.values, rcond=None)[0]
        y_hat = X @ beta_hat
        resid = y.values - y_hat
        tss = ((y - y.mean())**2).sum()
        rss = (resid**2).sum()
        r2 = 1 - rss / tss if tss != 0 else np.nan
        out[col] = PricingFit(beta=beta_hat[1], alpha=beta_hat[0], r2=r2)
    return out


def log_utility_optimal_weights(returns: pd.DataFrame, n_iter: int = 5_000, seed: int = 0) -> pd.Series:
    """Crude Monte Carlo search for weights maximizing sample mean log(1 + w'r_t).

    Enforces weights >=0 and sum=1 (long-only). For real work, use convex optimization.
    """
    rng = np.random.default_rng(seed)
    best_w = None
    best_val = -np.inf
    r = returns.values
    for _ in range(n_iter):
        w = rng.random(returns.shape[1])
        w /= w.sum()
        port = r @ w
        if np.any(1 + port <= 0):
            continue
        val = np.mean(np.log1p(port))
        if val > best_val:
            best_val = val
            best_w = w
    return pd.Series(best_w, index=returns.columns, name='log_utility_weight')


def analyze_problem3(ff25: pd.DataFrame, rf: float = 0.02) -> Dict[str, Any]:
    extremes = select_extremes(ff25)
    tang_ext = tangency_portfolio(extremes, rf=rf)
    pricing = pricing_relation(extremes, tang_ext['weights'] @ extremes.T, rf=rf)
    log_w = log_utility_optimal_weights(extremes)
    tang_all = tangency_portfolio(ff25, rf=rf)
    summary = {
        'extreme_tangency': tang_ext,
        'pricing_relation': pricing,
        'log_utility_weights': log_w,
        'all25_tangency': tang_all,
        'all25_sharpe': tang_all['sharpe']
    }
    return summary

