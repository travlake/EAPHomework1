from __future__ import annotations
import sys
from pathlib import Path

# Ensure 'src' directory is on path for direct execution
# This is src/hw1, so project root is 2 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hw1.paths import DATA_ROOT, OUTPUT_DIR

def solve_problem5():
    """
    Recreates the solution to Problem 4 from the MATLAB code in problem5.py.
    """
    # Load the data
    df = pd.read_csv(DATA_ROOT / "raw" / "Hw1p45.csv")

    # Extract the necessary series
    Rf = df['realTbillret'].values
    Rm = df['realstockret'].values
    nondur = df['Nondur'].values

    # Define the range of beta and gamma values
    beta_vals = np.arange(0.98, 1.0002, 0.0002)
    gamma_vals = np.arange(0.1, 10.1, 0.1)

    # Calculate lambda (consumption growth)
    lambda_ = nondur[1:] / nondur[:-1]

    # Initialize matrices for ERf and ERmRf
    ERf = np.zeros((len(beta_vals), len(gamma_vals)))
    ERmRf = np.zeros((len(beta_vals), len(gamma_vals)))

    # Populate the matrices
    for i, beta in enumerate(beta_vals):
        for j, gamma in enumerate(gamma_vals):
            ERf[i, j] = 1 / np.mean(beta * lambda_**(-gamma))
            ERmRf[i, j] = np.mean(lambda_) / np.mean(beta * lambda_**(1 - gamma)) - ERf[i, j]

    # Find the frontier pairs
    ERf_frontier = []
    ERmRf_frontier = []
    for i in range(ERf.shape[0]):
        for j in range(ERf.shape[1]):
            # A point is on the frontier if no other point has both lower ERf and higher ERmRf
            if np.sum((ERf < (ERf[i, j] + 0.0001)) & (ERmRf > (ERmRf[i, j] - 0.000001))) == 1:
                ERf_frontier.append(ERf[i, j])
                ERmRf_frontier.append(ERmRf[i, j])

    # Sort the frontier for plotting
    frontier_points = sorted(zip(ERf_frontier, ERmRf_frontier))
    ERf_frontier_sorted = [p[0] for p in frontier_points]
    ERmRf_frontier_sorted = [p[1] for p in frontier_points]

    # Plot the results
    plt.figure(figsize=(10, 6))
    # Annualize and convert to percent
    plt.plot(400 * (np.array(ERf_frontier_sorted) - 1), 400 * np.array(ERmRf_frontier_sorted), label='Consumption-Based Asset Pricing Frontier')

    # Plot the data point
    plt.plot(np.mean(400 * Rf), np.mean(400 * (Rm - Rf)), 'ko', label='Data (Mean Real T-Bill and Equity Premium)')

    # Formatting
    plt.xlim([0, 4])
    # plt.ylim([0, 2.5]) # Match MATLAB output if desired
    plt.xlabel("Annualized Risk-Free Rate (%)")
    plt.ylabel("Annualized Equity Risk Premium (%)")
    plt.title("Equity Premium Puzzle")
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(OUTPUT_DIR / "problem5_equity_premium_puzzle.png")

    print("Problem 5 solved. Figure saved to output/problem5_equity_premium_puzzle.png")

if __name__ == "__main__":
    solve_problem5()
