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
from hw1.paths import OUTPUT_DIR
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
    # n_simulations = 5000
    n_simulations = 50000

    # Create panels for Value-Weighted and Equal-Weighted returns
    panel_a = create_panel("value", n_simulations)
    panel_b = create_panel("equal", n_simulations)

    # Combine panels into the final table with Panel A and Panel B as separate sections
    result_table = pd.concat([panel_a, panel_b], keys=['Panel A: Value-Weighted', 'Panel B: Equal-Weighted'])

    # Save the results to a CSV file
    output_path = OUTPUT_DIR / "problem1_solution.csv"
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
    results_daily = analyze_returns(daily_returns, n_simulations, freq_name='Daily')
    results_monthly = analyze_returns(monthly_returns, n_simulations, freq_name='Monthly')
    results_annual = analyze_returns(annual_returns, n_simulations, freq_name='Annually')

    # Combine results for the panel - stack vertically
    panel = pd.concat([results_daily, results_monthly, results_annual])
    return panel

def analyze_returns(returns: pd.Series, n_simulations: int, freq_name: str) -> pd.DataFrame:
    """
    Performs autocorrelation analysis for a given return series.
    Returns a DataFrame with rows for each measure and columns for each statistic.
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
    shuffled_returns_matrix = np.array([np.random.permutation(returns_np) for _ in tqdm(range(n_simulations), desc=f"Simulating {freq_name}")])

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
    vr_stat = var_q / (q * var_1) - 1

    vr_se_asy = np.sqrt(4 * sum( ((q - j) / q)**2 for j in range(1, q + 1)) / T)
    p_val_vr_asy = 2 * (1 - chi2.cdf((vr_stat / vr_se_asy)**2, df=1)) # Two-sided p-value

    # Vectorized simulation for VR(5)
    sim_var_1 = shuffled_returns_matrix.var(axis=1)
    # Calculate rolling sums for each simulation (row)
    cumsum_matrix = np.cumsum(shuffled_returns_matrix, axis=1)
    rolling_sum_matrix = cumsum_matrix[:, q-1:] - np.pad(cumsum_matrix[:, :-q], ((0,0), (1,0)), 'constant')
    sim_var_q = rolling_sum_matrix.var(axis=1)
    simulated_vr_stats = sim_var_q / (q * sim_var_1) - 1

    bias_vr_sim = simulated_vr_stats.mean()
    p_val_vr_sim = (np.abs(np.array(simulated_vr_stats)) > np.abs(vr_stat)).mean()

    # delta(5) - Long-run return regression
    # Regress r_{t+1} on constant and sum of r_t through r_{t-4}
    q = 5
    # Create lagged sum of returns (from t to t-4)
    X_reg = pd.DataFrame({
        'const': 1,
        'lagged_sum': returns.rolling(q).mean()
    }, index=returns.index)
    y_reg = returns.shift(-1)  # r_{t+1}

    # Drop NaNs from alignment
    valid_idx = y_reg.dropna().index
    valid_idx = valid_idx.intersection(X_reg.dropna().index)
    y_reg = y_reg.loc[valid_idx]
    X_reg = X_reg.loc[valid_idx]

    model = OLS(y_reg, X_reg).fit()
    delta_stat = model.params['lagged_sum']
    p_val_delta_asy = model.pvalues['lagged_sum']

    # Vectorized simulation for delta(5)
    # Calculate rolling sum for each simulation
    simulated_delta_stats = []
    for i in range(n_simulations):
        sim_returns = shuffled_returns_matrix[i, :]
        # Rolling sum of q periods
        lagged_sum = pd.Series(sim_returns).rolling(q).sum().values
        # Next period return
        next_return = np.concatenate([sim_returns[1:], [np.nan]])

        # Remove NaNs
        valid_mask = ~(np.isnan(lagged_sum) | np.isnan(next_return))
        lagged_avg_clean = lagged_sum[valid_mask] / q
        next_return_clean = next_return[valid_mask]

        # Simple regression coefficient: beta = cov(x,y) / var(x)
        if len(next_return_clean) > 0 and np.var(lagged_avg_clean) > 0:
            delta_coef = np.cov(lagged_avg_clean, next_return_clean)[0, 1] / np.var(lagged_avg_clean)
            simulated_delta_stats.append(delta_coef)

    simulated_delta_stats = np.array(simulated_delta_stats)
    bias_delta_sim = simulated_delta_stats.mean()
    p_val_delta_sim = (np.abs(simulated_delta_stats) > np.abs(delta_stat)).mean()

    # --- Assemble Results Table in the required format ---
    # Create a dictionary with columns as statistics and rows as measures
    data = {}

    # ρ(1) through ρ(5) columns
    for i, l in enumerate(lags):
        col_name = f'$\\rho({l})$'
        data[col_name] = {
            'Estimate': rhos[i],
            'Bias (Asy)': bias_asy,
            'Bias (Sim)': bias_sim[i],
            'SE (Asy)': se_asy[i],
            'SE (Sim)': se_sim[i]
        }

    # Empty column separator
    data[''] = {
        'Estimate': np.nan,
        'Bias (Asy)': np.nan,
        'Bias (Sim)': np.nan,
        'SE (Asy)': np.nan,
        'SE (Sim)': np.nan
    }

    # Q(5) column
    data['$Q(5)$'] = {
        'Estimate': q_stat,
        'Bias (Asy)': np.nan,
        'Bias (Sim)': np.nan,
        'SE (Asy)': p_val_q_asy,
        'SE (Sim)': p_val_q_sim
    }

    # VR(5) column
    data['$VR(5)$'] = {
        'Estimate': vr_stat,
        'Bias (Asy)': np.nan,
        'Bias (Sim)': bias_vr_sim,
        'SE (Asy)': p_val_vr_asy,
        'SE (Sim)': p_val_vr_sim
    }

    # δ(5) column
    data['$\\delta(5)$'] = {
        'Estimate': delta_stat,
        'Bias (Asy)': bias_asy*q,
        'Bias (Sim)': bias_delta_sim,
        'SE (Asy)': p_val_delta_asy,
        'SE (Sim)': p_val_delta_sim
    }

    # Create DataFrame and transpose to get desired structure
    df = pd.DataFrame(data)

    # Add the frequency name as the first level index
    df.index = pd.MultiIndex.from_product([[freq_name], df.index], names=['Frequency', 'Measure'])

    return df

if __name__ == '__main__':
    solve_problem1()
