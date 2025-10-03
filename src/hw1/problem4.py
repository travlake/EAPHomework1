"""
Solution for Problem 4 of Homework 1.

This script replicates the empirical test of the Hansen and Jagannathan (1991) bound
using quarterly real returns on the CRSP value-weighted index and Treasury Bills,
along with non-durables consumption data.
"""
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


def solve_problem4():
    """
    Recreates the Hansen-Jagannathan (1991) bound test.

    The HJ bound provides a lower bound on the variance of any valid SDF given its mean.
    We test whether a consumption-based SDF (CRRA utility) can satisfy this bound.
    """
    print("Starting Problem 4: Hansen-Jagannathan Bound Test...")

    # --- Load the data ---
    df = pd.read_csv(DATA_ROOT / "raw" / "Hw1p45.csv")

    # Extract the necessary series
    Rf = df['realTbillret'].values
    Rm = df['realstockret'].values
    nondur = df['Nondur'].values

    # --- Parameters ---
    # Subjective discount factor (given in lectures)
    beta = 0.99

    # Range of nu values to trace out the HJ bound
    # nu represents E[m], the expected value of the SDF
    nus = np.linspace(0.94, 1.05, 250)
    numnus = len(nus)

    # Range of gamma values (coefficient of relative risk aversion)
    # to test the consumption-based CRRA model
    gammas = np.arange(0, 425, 5)
    numgammas = len(gammas)

    # --- Calculate consumption growth ---
    # lambda = C_{t+1} / C_t
    lambda_ = nondur[1:] / nondur[:-1]

    # --- Compute the HJ bound ---
    # The HJ bound states that for any valid SDF m with E[m] = nu:
    # Var(m) >= nu^2 * (E[R] - 1/nu * 1)' * Sigma^{-1} * (E[R] - 1/nu * 1)
    # where R is the vector of gross returns

    # Construct gross return matrix
    Rmat = np.column_stack([1 + Rm, 1 + Rf])

    # Calculate mean and covariance of gross returns
    mean_R = np.mean(Rmat, axis=0)
    Sigma = np.cov(Rmat, rowvar=False)
    Sigma_inv = np.linalg.inv(Sigma)

    # For each nu, compute the HJ variance bound
    sdfvar_hjbound = np.zeros(numnus)
    for i, nu in enumerate(nus):
        # HJ bound formula
        excess = mean_R - (1 / nu)
        sdfvar_hjbound[i] = (nu ** 2) * (excess.T @ Sigma_inv @ excess)

    # --- Compute SDF moments from consumption-based model ---
    # For CRRA utility: m = beta * (C_{t+1} / C_t)^{-gamma} = beta * lambda^{-gamma}

    sdfmean_data = np.zeros(numgammas)
    sdfvar_data = np.zeros(numgammas)

    for i, gamma in enumerate(gammas):
        # SDF realizations for this gamma value
        sdfreal = beta * (lambda_ ** (-gamma))

        sdfmean_data[i] = np.mean(sdfreal)
        sdfvar_data[i] = np.var(sdfreal, ddof=1)  # Use sample variance

    # --- Create the plot ---
    plt.figure(figsize=(10, 6))

    # Plot the HJ bound frontier line
    plt.plot(nus, np.sqrt(sdfvar_hjbound), 'r-', linewidth=2, label='HJ Bound Frontier')

    # Fill the acceptable region (above the HJ bound)
    plt.fill_between(nus, np.sqrt(sdfvar_hjbound), 6,
                     alpha=0.2, color='green', label='Acceptable Region (Above HJ Bound)')

    # Plot the consumption-based SDF mean/std pairs as a continuous line
    plt.plot(sdfmean_data, np.sqrt(sdfvar_data), 'b-', linewidth=1.5,
             label='CRRA SDF (γ = 1 to 425)')

    # Find which gamma values satisfy the bound
    satisfies_bound = []
    satisfies_bound_indices = []
    for i, gamma in enumerate(gammas):
        # Interpolate to find the HJ bound at this mean
        if sdfmean_data[i] >= min(nus) and sdfmean_data[i] <= max(nus):
            hj_std_at_mean = np.interp(sdfmean_data[i], nus, np.sqrt(sdfvar_hjbound))
            if np.sqrt(sdfvar_data[i]) >= hj_std_at_mean:
                satisfies_bound.append(gamma)
                satisfies_bound_indices.append(i)

    # Plot dots only for gamma values that satisfy the bound
    if len(satisfies_bound_indices) > 0:
        satisfies_means = sdfmean_data[satisfies_bound_indices]
        satisfies_stds = np.sqrt(sdfvar_data[satisfies_bound_indices])
        plt.plot(satisfies_means, satisfies_stds, 'ro', markersize=5,
                label=f'CRRA satisfying bound (γ ≥ {min(satisfies_bound)})')

        # Label a few key gamma values
        label_gammas = [gamma for gamma in satisfies_bound if gamma % 5 == 0 and 395 <= gamma <= 410]
        for gamma in label_gammas:
            idx = list(gammas).index(gamma)
            plt.text(sdfmean_data[idx] - 0.005, np.sqrt(sdfvar_data[idx]) + 0.08,
                    f'γ={gamma}', fontsize=8, ha='right')

    # Formatting
    # plt.xlim([min(nus), max(nus)])
    plt.ylim([0, 6])
    plt.xlabel('SDF Mean', fontsize=12)
    plt.ylabel('SDF Standard Deviation', fontsize=12)
    plt.title('Hansen-Jagannathan Bound Test', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # Save the figure
    output_path = OUTPUT_DIR / "problem4_hj_bound.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"HJ bound plot saved to: {output_path}")
    plt.close()

    # --- Summary statistics ---
    print("\n=== Hansen-Jagannathan Bound Test Results ===")
    print(f"\nAssets used: CRSP Value-Weighted Index and Treasury Bills")
    print(f"Number of observations: {len(lambda_)}")
    print(f"\nGross Returns Summary:")
    print(f"  Mean Stock Return: {mean_R[0]:.6f}")
    print(f"  Mean T-Bill Return: {mean_R[1]:.6f}")
    print(f"\nConsumption Growth Summary:")
    print(f"  Mean: {np.mean(lambda_):.6f}")
    print(f"  Std Dev: {np.std(lambda_, ddof=1):.6f}")

    # Find which gamma values (approximately) satisfy the bound
    # A point satisfies the bound if it's in the acceptable region
    satisfies_bound = []
    for i, gamma in enumerate(gammas):
        # Interpolate to find the HJ bound at this mean
        if sdfmean_data[i] >= min(nus) and sdfmean_data[i] <= max(nus):
            hj_std_at_mean = np.interp(sdfmean_data[i], nus, np.sqrt(sdfvar_hjbound))
            if np.sqrt(sdfvar_data[i]) >= hj_std_at_mean:
                satisfies_bound.append(gamma)

    if satisfies_bound:
        print(f"\nGamma values satisfying HJ bound (approximately): {satisfies_bound[:5]}...")
        print(f"  Minimum gamma: {min(satisfies_bound)}")
    else:
        print("\nNo gamma values in the tested range satisfy the HJ bound.")
        print("This illustrates the challenge of reconciling consumption-based models with asset returns.")

    print("\n=== Interpretation ===")
    print("The HJ bound provides a minimum variance for any valid SDF given its mean.")
    print("Points above the frontier satisfy the bound and are valid SDFs.")
    print("Points below the frontier violate the bound and cannot be valid SDFs.")
    print("The CRRA consumption-based SDF only satisfies the bound at very high risk aversion (γ ≥ 395).")
    print("This highlights the difficulty of explaining asset returns with consumption data.")

    print("\nProblem 4 solved successfully.")


if __name__ == "__main__":
    solve_problem4()
