"""
Solution for problem 1.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import chi2
# Ensure 'src' directory is on path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hw1.data import load_market_returns
from hw1.paths import PROCESSED_DATA
from statsmodels.regression.linear_model import OLS
from tqdm import tqdm


def _vectorized_autocorr(data_matrix, lag):
    """Vectorized autocorrelation calculation for a matrix of time series."""
    m, n = data_matrix.shape
    mu = data_matrix.mean(axis=1, keepdims=True)
    demeaned_data = data_matrix - mu
    
    numerator = (demeaned_data[:, lag:] * demeaned_data[:, :-lag]).sum(axis=1)
    denominator = (demeaned_data**2).sum(axis=1)
    
    return numerator / denominator

def solve_problem1():
    """
    Generates the table for problem 1.
    """
    # Define the number of simulations for bias/SE calculation
    n_simulations = 50000

    # Create panels for Value-Weighted and Equal-Weighted returns
    panel_a = create_panel("value", n_simulations)
    panel_b = create_panel("equal", n_simulations)

    # Combine panels into the final table
    result_table = pd.concat([panel_a, panel_b], keys=['Panel A: Value-Weighted', 'Panel B: Equal-Weighted'])

    # Save the results to a CSV file
    output_path = PROCESSED_DATA / "problem1_solution.csv"
    result_table.to_csv(output_path)

    print("Problem 1 solution saved to:", output_path)
    return result_table

def create_panel(kind: str, n_simulations: int) -> pd.DataFrame:
    """
    Creates a data panel for a given return type (value or equal weighted).
    """
    # Load the full daily return series
    daily_returns = load_market_returns(kind=kind)

    # Resample to get monthly and annual returns
    monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    annual_returns = daily_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)

    # Analyze each frequency
    results_daily = analyze_returns(daily_returns, n_simulations)
    results_monthly = analyze_returns(monthly_returns, n_simulations)
    results_annual = analyze_returns(annual_returns, n_simulations)

    # Combine results for the panel
    panel = pd.concat([results_daily, results_monthly, results_annual], keys=['Daily', 'Monthly', 'Annual'])
    return panel

def analyze_returns(returns: pd.Series, n_simulations: int) -> pd.DataFrame:
    """
    Performs autocorrelation analysis for a given return series.
    """
    T = len(returns)
    lags = range(1, 6)

    # --- Autocorrelation Estimates ---
    rhos = [returns.autocorr(lag=l) for l in lags]

    # --- IID Simulations ---
    # Vectorized approach for performance
    # 1. Get numpy array of returns
    returns_np = returns.to_numpy()
    
    # 2. Create a matrix of shuffled returns
    # Each row is a simulated time series
    shuffled_returns_matrix = np.array([np.random.permutation(returns_np) for _ in tqdm(range(n_simulations), desc=f"Simulating {returns.name}")])

    # 3. Vectorized autocorrelation for all simulations and lags
    simulated_rhos = np.array([_vectorized_autocorr(shuffled_returns_matrix, l) for l in lags]).T

    # --- Bias and Standard Errors ---
    bias_asy = -1 / T
    bias_sim = simulated_rhos.mean(axis=0)
    se_asy = np.sqrt((1 - np.array(rhos)**2) / T)
    se_sim = simulated_rhos.std(axis=0)

    # --- Joint Significance Tests ---
    # Q(5) - Box-Pierce
    q_stat = T * (np.array(rhos)**2).sum()
    simulated_q_stats = T * (simulated_rhos**2).sum(axis=1)
    p_val_q_asy = chi2.sf(q_stat, df=len(lags)) # Chi2(5) p-value
    p_val_q_sim = (simulated_q_stats > q_stat).mean()

    # VR(5) - Variance Ratio
    # The variance of q-period returns should be q times the variance of 1-period returns
    q = 5
    var_1 = returns.var()
    var_q = returns.rolling(q).sum().var()
    vr_stat = var_q / (q * var_1)

    # Vectorized simulation for VR(5)
    sim_var_1 = shuffled_returns_matrix.var(axis=1)
    # Calculate rolling sums for each simulation (row)
    cumsum_matrix = np.cumsum(shuffled_returns_matrix, axis=1)
    rolling_sum_matrix = cumsum_matrix[:, q-1:] - np.pad(cumsum_matrix[:, :-q], ((0,0), (1,0)), 'constant')
    sim_var_q = rolling_sum_matrix.var(axis=1)
    simulated_vr_stats = sim_var_q / (q * sim_var_1)
    
    p_val_vr_sim = (np.abs(np.array(simulated_vr_stats) - 1) > np.abs(vr_stat - 1)).mean()

    # delta(5) - Long-run return regression
    # Regress cumulative return on a constant
    X = pd.DataFrame({'const': 1}, index=returns.index)
    y = returns.rolling(5).sum().shift(-4) # 5-period ahead cumulative return

    # Drop NaNs from alignment
    valid_idx = y.dropna().index
    y = y.loc[valid_idx]
    X = X.loc[valid_idx]

    model = OLS(y, X).fit()
    delta_stat = model.params['const']

    # Vectorized simulation for delta(5)
    # The delta stat is just the mean of the q-period-ahead cumulative returns
    simulated_delta_stats = rolling_sum_matrix.mean(axis=1)

    p_val_delta_sim = (np.abs(np.array(simulated_delta_stats)) > np.abs(delta_stat)).mean()


    # --- Assemble Results Table ---
    results = []
    for i, l in enumerate(lags):
        results.append({
            'Statistic': f'rho({l})',
            'Estimate': rhos[i],
            'Bias (Asy)': bias_asy,
            'Bias (Sim)': bias_sim[i],
            'SE (Asy)': se_asy[i],
            'SE (Sim)': se_sim[i]
        })

    # Add joint tests
    results.append({'Statistic': 'Q(5)', 'Estimate': q_stat, 'p-val (Asy)': p_val_q_asy, 'p-val (Sim)': p_val_q_sim})
    results.append({'Statistic': 'VR(5)', 'Estimate': vr_stat, 'p-val (Sim)': p_val_vr_sim})
    results.append({'Statistic': 'delta(5)', 'Estimate': delta_stat, 'p-val (Sim)': p_val_delta_sim})

    df = pd.DataFrame(results).set_index('Statistic')
    return df

if __name__ == '__main__':
    solve_problem1()
