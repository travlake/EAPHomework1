# Finance 395.4 – Empirical Methods in Asset Pricing

Repository for Homework 1 (and supportive lecture references) for Finance 395.4 "Empirical Methods in Asset Pricing" at the McCombs School of Business.

**Instructor:** Travis L. Johnson  
**Term:** Fall 2025  
**Assignment Focus:** Empirical methods, factor models, consumption-based asset pricing, and modern ML/AI tools applied to asset pricing research.

---
## Repository Layout
```
ref/                          LaTeX sources + distributed PDFs for homework & lectures
src/hw1/                      Python package for all problem solutions
    problem1.py               Autocorrelation + small-sample bias simulation
    problem2.py               Return forecasting (LASSO / Ridge / Elastic Net / NN)
    problem3.py               FF25 portfolios: tangency, SML, log utility optimization
    problem4.py               Hansen–Jagannathan bound with CRRA utility
    problem5.py               Mehra–Prescott equity premium puzzle illustration
    data.py                   Data loaders for market returns, FF25, and consumption
    portfolio.py              Portfolio math helpers (Sharpe, tangency, mean-variance)
    ml.py                     Wrappers around scikit-learn models for forecasting
    to_latex.py               Converts solution CSVs to formatted LaTeX tables
    paths.py                  Centralized path constants
run_all.py                    Driver script to execute all problem solutions
requirements.txt              Python dependencies
data/
  raw/                        Source data files (gitignored except .gitkeep)
    Hw1p45.csv                Market returns, T-bill, and consumption data
  processed/                  Cleaned datasets (gitignored except .gitkeep)
    market_returns.csv        CRSP value-weighted and equal-weighted returns
    ff25_monthly.csv          Fama-French 25 size/value portfolios
output/                       Solution outputs (CSV tables and PNG figures)
  problem1_solution.csv       Autocorrelation statistics (daily/monthly/annual)
  problem2_solution.csv       ML model forecasting performance comparison
  problem3abc_4_asset_results.csv   Extreme portfolio analysis results
  problem3a_sml_4_asset_model.png   Security Market Line plot
  problem3d_25_asset_weights_grid.csv   Full 25-asset tangency/log-utility weights
  problem4_hj_bound.png       Hansen-Jagannathan bound visualization
  problem5_equity_premium_puzzle.png  Risk aversion vs. equity premium grid
.gitignore                    Ignore rules for Python, data, LaTeX aux files
```

---
## Quick Start

### Setup Python Environment
**PowerShell (Windows):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**bash (macOS/Linux):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run All Solutions
Execute all problems at once:
```powershell
python run_all.py
```

Or run individual problems:
```powershell
python src\hw1\problem1.py
python src\hw1\problem2.py
python src\hw1\problem3.py
python src\hw1\problem4.py
python src\hw1\problem5.py
```

Each script generates its corresponding outputs in the `output/` directory.

### Generate LaTeX Tables
Convert solution CSVs to formatted LaTeX tables:
```powershell
python src\hw1\to_latex.py
```

Tables are saved in `tex/out/` and referenced by the main solution document.

### Compile LaTeX Solution
```powershell
cd tex
pdflatex "Homework 1 solution.tex"
pdflatex "Homework 1 solution.tex"  # repeat for cross-references
```

---
## Problem Summaries

### Problem 1: Return Autocorrelation and Small-Sample Bias
- Computes sample autocorrelations for CRSP value-weighted returns at daily, monthly, and annual frequencies
- Estimates long-run return predictability coefficient δ(5) using overlapping 5-period regressions
- Runs Monte Carlo simulation (1000 draws) under null hypothesis of zero autocorrelation
- Calculates asymptotic and simulation-based standard errors
- **Output:** `output/problem1_solution.csv` containing estimates, standard errors, and t-statistics

### Problem 2: Machine Learning for Return Forecasting
- Constructs linear and non-linear feature sets from lagged returns
- Implements and compares forecasting models:
  - LASSO (L1 regularization)
  - Ridge (L2 regularization)  
  - Elastic Net (combined L1/L2)
  - Neural Network (MLP with single hidden layer, ReLU activation)
- Uses time-series cross-validation to select hyperparameters
- Evaluates out-of-sample R² performance
- **Output:** `output/problem2_solution.csv` with model comparison table

### Problem 3: Fama-French 25 Portfolio Analysis
- **(a)** Constructs tangency portfolio from 4 extreme FF25 portfolios (small/large × value/growth)
- **(b)** Tests Security Market Line pricing relation; plots expected returns vs. tangency betas
- **(c)** Computes log-utility optimal portfolio weights
- **(d)** Extends analysis to full 25-portfolio universe for tangency and log-utility solutions
- **Outputs:** 
  - `output/problem3abc_4_asset_results.csv` (Parts a-c summary)
  - `output/problem3a_sml_4_asset_model.png` (SML plot)
  - `output/problem3d_25_asset_weights_grid.csv` (Full 25-asset weights)

### Problem 4: Hansen-Jagannathan Bound
- Computes HJ bound using quarterly market returns and risk-free rate
- Evaluates CRRA stochastic discount factor: m = β(C_{t+1}/C_t)^{-γ}
- Tests range of risk aversion parameters γ ∈ [1, 425]
- Visualizes which (mean, std) pairs satisfy the HJ bound
- **Output:** `output/problem4_hj_bound.png` showing feasible region and CRRA SDFs

### Problem 5: Equity Premium Puzzle (Mehra-Prescott)
- Replicates classic equity premium puzzle analysis
- Generates grid of theoretical equity premiums for different (β, γ) combinations
- Overlays observed market equity premium from historical data
- Demonstrates implausibly high risk aversion needed to match observed premium
- **Output:** `output/problem5_equity_premium_puzzle.png` with contour plot

---
## Data Sources
- **Market Returns:** CRSP value-weighted and equal-weighted index returns (daily, monthly, annual)
- **Fama-French 25:** Monthly returns on 25 size/B-M sorted portfolios
- **Consumption & Risk-Free Rate:** Quarterly data from `Hw1p45.csv`

All raw data stored in `data/raw/`, with processed versions cached in `data/processed/`.

---
## Python Environment
**Key Dependencies:**
- `numpy`, `pandas` – Data manipulation
- `scipy`, `statsmodels` – Statistical analysis
- `scikit-learn` – Machine learning models (LASSO, Ridge, Elastic Net, MLP)
- `matplotlib`, `seaborn` – Visualization
- `tqdm` – Progress bars for simulations

All dependencies pinned in `requirements.txt`.

---
## Version Control Notes
- Raw data files (`data/raw/`) are not committed (except `.gitkeep`)
- Generated outputs (`output/`, `tex/out/`) are committed for reference
- LaTeX auxiliary files (`*.aux`, `*.log`) are gitignored
- Python cache (`__pycache__/`, `.pyc`) is gitignored

---
## Academic Integrity
This code implements solutions to assigned homework problems. Write-up must include economic interpretation and discussion, not just numerical results. All analysis should be original work; informal collaboration is allowed per course syllabus.

---
## Contact
**Course questions:** Contact instructor through official channels  
**Technical issues:** Open an issue or submit a pull request

---
## License
Academic use only. Not licensed for distribution outside course context.
