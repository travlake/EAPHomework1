"""Quick demonstration script for HW1 scaffold.

Usage:
    python run_all.py

Produces minimal console output verifying core placeholder functions operate.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure 'src' directory is on path for direct execution without install
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hw1 import data, problem1, problem2, problem3, problem4, problem5  # noqa: E402


def main():  # pragma: no cover - simple orchestration
    # Problem 1 demo
    mkt_vw = data.load_market_returns(kind="value", n=300)
    ac_res = problem1.compute_autocorrelation(mkt_vw, max_lag=5)
    print("Problem 1 autocorr (first 5 lags):")
    print(ac_res.autocorr.head())

    # Problem 2 feature matrix (linear only)
    feat = problem2.build_linear_feature_matrix(mkt_vw)
    print("Problem 2 feature matrix shape (linear):", feat.shape)

    # Problem 3 extremes tangency (synthetic data)
    ff25 = data.load_ff25_monthly(n=60)
    summary3 = problem3.analyze_problem3(ff25)
    print("Problem 3 extreme tangency Sharpe:", summary3['extreme_tangency']['sharpe'])

    # Problem 4 Hansen-Jagannathan placeholder
    cons_df = data.load_consumption_and_rf(n=40)
    hj = problem4.hj_bound_summary(cons_df['Rm'], cons_df['Rf'], cons_df['cons_growth'])
    print("Problem 4 HJ summary rows:", len(hj))

    # Problem 5 lambda moments
    lam = problem5.compute_lambda_moments(cons_df['cons_growth'])
    print("Problem 5 lambda moments gammas:", lam['gamma'].tolist())

if __name__ == "__main__":
    main()
