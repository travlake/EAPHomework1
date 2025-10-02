"""
Solution for Problem 3 of Homework 1.

This script downloads Fama-French 25 portfolio data from WRDS, calculates
the tangency and log-utility portfolio weights, and generates summary tables
and a plot of the resulting Security Market Line.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wrds
from scipy.optimize import minimize

# --- Path Setup ---
# To allow for running this script directly from an IDE, we need to add the project's 'src'
# directory to the Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hw1.paths import OUTPUT_DIR
from hw1.data import load_ff25_data

# --- Main Function ---

def solve_problem3():
    """
    Orchestrates the solution for Problem 3, following the multi-part structure.
    """
    print("Starting Problem 3: Tangency and Log Utility Portfolios...")

    # --- Part 1: Data Loading and Preparation ---
    ff25 = load_ff25_data()
    excess_returns_all = ff25.subtract(ff25['rf'], axis=0).drop(columns='rf')
    mu_excess_all = excess_returns_all.mean()

    # --- Part 2: Analysis of 4 Extreme Portfolios (Parts a, b, c) ---
    print("\nAnalyzing the 4 extreme portfolios...")
    extreme_portfolios = ['s1b1_vwret', 's1b5_vwret', 's5b1_vwret', 's5b5_vwret']
    excess_returns_4 = excess_returns_all[extreme_portfolios]
    mu_excess_4 = excess_returns_4.mean()
    Sigma_4 = excess_returns_4.cov()
    Sigma_inv_4 = np.linalg.inv(Sigma_4)
    ones_4 = np.ones(len(mu_excess_4))

    # (a) Tangency portfolio for the 4 extreme assets
    w_tangency_4 = (Sigma_inv_4 @ mu_excess_4) / (ones_4.T @ Sigma_inv_4 @ mu_excess_4)
    ret_tangency_4 = excess_returns_4 @ w_tangency_4

    # (b) Sharpe Ratio for the 4-asset tangency portfolio (annualized)
    sharpe_tangency_4 = (ret_tangency_4.mean() / ret_tangency_4.std()) * np.sqrt(12)

    # (c) Log utility portfolio for the 4 extreme assets
    # Find w* = argmax E[log(1 + w'*r)] using numerical optimization
    w_log_utility_4 = calculate_log_utility_portfolio(excess_returns_4)

    # (a, cont.) Assess the pricing model for all 25 assets
    # Beta is calculated with respect to the 4-asset tangency portfolio
    cov_with_tangency_4 = excess_returns_all.apply(lambda x: x.cov(ret_tangency_4))
    var_tangency_4 = ret_tangency_4.var()
    betas_vs_4 = cov_with_tangency_4 / var_tangency_4

    # Create and save the SML plot based on the 4-asset model
    create_sml_plot(
        betas=betas_vs_4,
        mean_returns=mu_excess_all,
        mean_tangency_return=ret_tangency_4.mean(),
        filename="problem3a_sml_4_asset_model.png",
        title="SML using 4 Extreme Portfolios as the Factor"
    )

    # --- Part 3: Analysis of all 25 Portfolios (Part d) ---
    print("\nAnalyzing all 25 portfolios...")
    Sigma_25 = excess_returns_all.cov()
    Sigma_inv_25 = np.linalg.inv(Sigma_25)
    ones_25 = np.ones(len(mu_excess_all))

    # (d) Tangency portfolio for all 25 assets
    w_tangency_25 = (Sigma_inv_25 @ mu_excess_all) / (ones_25.T @ Sigma_inv_25 @ mu_excess_all)
    ret_tangency_25 = excess_returns_all @ w_tangency_25
    sharpe_tangency_25 = (ret_tangency_25.mean() / ret_tangency_25.std()) * np.sqrt(12)

    # --- Part 4: Table Generation ---
    print("\nGenerating output tables...")
    # Table for parts a, b, c
    create_4_asset_results_table(w_tangency_4, w_log_utility_4, sharpe_tangency_4, extreme_portfolios)

    # Table for part d
    create_25_asset_grid_table(w_tangency_25, sharpe_tangency_25, excess_returns_all.columns)

    print("\nProblem 3 solved successfully. Files saved to the 'output' directory.")


# --- Analysis and Output Functions ---

def create_sml_plot(betas: pd.Series, mean_returns: pd.Series, mean_tangency_return: float, filename: str, title: str):
    """
    Generates and saves a plot of the Security Market Line (SML).
    """
    # Annualize returns for plotting
    mean_returns_annual = mean_returns * 12
    mean_tangency_return_annual = mean_tangency_return * 12

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each of the 25 portfolios as a scatter point
    ax.scatter(betas, mean_returns_annual, c='blue', alpha=0.7, label='25 Fama-French Portfolios')

    # Highlight the 4 extreme portfolios used to build the model
    extreme_portfolios = ['s1b1_vwret', 's1b5_vwret', 's5b1_vwret', 's5b5_vwret']
    ax.scatter(betas[extreme_portfolios], mean_returns_annual[extreme_portfolios], c='red', s=100, label='4 Extreme Portfolios (Model Assets)')

    # Add labels for each point
    for i, txt in enumerate(betas.index):
        ax.annotate(txt, (betas[i], mean_returns_annual[i]), fontsize=7, alpha=0.6,
                    xytext=(5, -5), textcoords='offset points')

    # Plot the SML: E[R_i] = beta_i * E[R_mkt]
    sml_x = np.array([0, betas.max() * 1.1])
    sml_y = sml_x * mean_tangency_return_annual
    ax.plot(sml_x, sml_y, color='red', linestyle='--', label='Security Market Line (SML)')

    # Formatting
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r'Beta ($\beta$) with respect to Tangency Portfolio', fontsize=12)
    ax.set_ylabel('Annualized Average Monthly Excess Return', fontsize=12)
    ax.legend()
    ax.grid(True)

    # Save the figure
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"SML plot saved to: {output_path}")
    plt.close()


def create_4_asset_results_table(w_tangency: np.ndarray, w_log: np.ndarray, sharpe: float, portfolio_names: list):
    """
    Creates and saves a table summarizing the results from the 4-asset analysis.
    """
    # Part (a) and (c): Portfolio Weights
    weights_df = pd.DataFrame({
        'Tangency Portfolio': w_tangency,
        'Log Utility Portfolio': w_log
    }, index=portfolio_names)
    weights_df.index.name = 'Portfolio'

    # Part (b): Tangency Portfolio Summary
    summary_stats = pd.Series({'Sharpe Ratio': sharpe}, name='Summary')
    summary_df = pd.DataFrame([summary_stats])

    # Save to a single CSV file in the output directory
    output_path = OUTPUT_DIR / "problem3abc_4_asset_results.csv"
    with open(output_path, 'w', newline='') as f:
        f.write("Parts (a) & (c): Portfolio Weights for 4 Extreme Portfolios\n")
        weights_df.to_csv(f)
        f.write("\nPart (b): Tangency Portfolio Summary\n")
        summary_df.to_csv(f, index=False)

    print(f"4-asset analysis table saved to: {output_path}")


def create_25_asset_grid_table(w_tangency: np.ndarray, sharpe: float, portfolio_names: pd.Index):
    """
    Creates and saves a 5x5 grid of the 25-asset tangency portfolio weights.
    """
    # Create a series with the correct portfolio names as the index
    weights_series = pd.Series(w_tangency, index=portfolio_names)

    # Manually define the Size and B/M categories for the grid
    size_cats = ['SMALL', '2', '3', '4', 'BIG']
    bm_cats = ['LoBM', '2', '3', '4', 'HiBM']

    # Create an empty DataFrame for the grid
    weights_grid = pd.DataFrame(index=size_cats, columns=bm_cats)
    weights_grid.index.name = 'Size'
    weights_grid.columns.name = 'Book-to-Market'

    # Populate the grid based on portfolio names
    for i, s_cat in enumerate(size_cats):
        for j, bm_cat in enumerate(bm_cats):
            # Find the weight for the corresponding portfolio name, e.g., 's1b1_vwret'
            port_name = f's{i+1}b{j+1}_vwret'
            if port_name in weights_series.index:
                weights_grid.loc[s_cat, bm_cat] = weights_series[port_name]

    # Save to CSV in the output directory
    output_path = OUTPUT_DIR / "problem3d_25_asset_weights_grid.csv"
    with open(output_path, 'w', newline='') as f:
        f.write("Part (d): Tangency Weights for all 25 Portfolios (5x5 Grid)\n")
        weights_grid.to_csv(f)
        f.write(f"\nSummary Sharpe Ratio,{sharpe}\n")

    print(f"25-asset weights grid saved to: {output_path}")


def calculate_log_utility_portfolio(excess_returns: pd.DataFrame) -> np.ndarray:
    """
    Calculate the log utility portfolio by maximizing E[log(1 + w'*r)].

    Args:
        excess_returns: DataFrame of excess returns for the assets

    Returns:
        Optimal portfolio weights as numpy array
    """
    n_assets = excess_returns.shape[1]

    def negative_log_utility(weights):
        """Negative of expected log utility (to minimize)"""
        portfolio_returns = excess_returns @ weights
        # Add small epsilon to avoid log(0) or log(negative)
        epsilon = 1e-8
        log_returns = np.log(1 + portfolio_returns + epsilon)
        return -np.mean(log_returns)

    # Initial guess: equal weights
    x0 = np.ones(n_assets) / n_assets

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # Bounds: allow both long and short positions
    bounds = [(-4, 4) for _ in range(n_assets)]

    # Optimize
    result = minimize(
        negative_log_utility,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )

    if not result.success:
        print(f"Warning: Log utility optimization did not converge: {result.message}")

    return result.x


# --- Execution ---

if __name__ == "__main__":
    # This block allows the script to be run directly.
    # It's useful for testing and development.
    try:
        solve_problem3()
    except wrds.errors.WrdsError as e:
        print(f"WRDS connection failed: {e}")
        print("Please ensure your WRDS credentials are set up correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
