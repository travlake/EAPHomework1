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

from hw1 import problem1, problem2, problem3, problem4, problem5  # noqa: E402


def main():  # pragma: no cover - simple orchestration
    print("=" * 60)
    print("Running all homework solutions...")
    print("=" * 60)

    # Problem 1: Autocorrelation and small-sample bias
    print("\n[1/5] Running Problem 1: Autocorrelation Analysis...")
    problem1.solve_problem1()
    print("✓ Problem 1 complete")

    # Problem 2: ML forecasting models
    print("\n[2/5] Running Problem 2: Return Forecasting Models...")
    problem2.solve_problem2()
    print("✓ Problem 2 complete")

    # Problem 3: FF25 portfolio analysis
    print("\n[3/5] Running Problem 3: Fama-French 25 Analysis...")
    problem3.solve_problem3()
    print("✓ Problem 3 complete")

    # Problem 4: Hansen-Jagannathan bound
    print("\n[4/5] Running Problem 4: Hansen-Jagannathan Bound...")
    problem4.solve_problem4()
    print("✓ Problem 4 complete")

    # Problem 5: Equity Premium Puzzle
    print("\n[5/5] Running Problem 5: Equity Premium Puzzle...")
    problem5.solve_problem5()
    print("✓ Problem 5 complete")

    print("\n" + "=" * 60)
    print("All problems solved successfully!")
    print("Outputs saved in: output/")
    print("=" * 60)

if __name__ == "__main__":
    main()
