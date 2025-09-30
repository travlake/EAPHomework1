# Finance 395.4 – Empirical Methods in Asset Pricing

Repository for Homework 1 (and supportive lecture references) for Finance 395.4 "Empirical Methods in Asset Pricing" at the McCombs School of Business.

**Instructor:** Travis L. Johnson  
**Term:** Fall 2025  
**Assignment Focus:** Empirical methods, factor models, consumption-based asset pricing, and modern ML/AI tools applied to asset pricing research.

---
## Repository Layout
```
ref/                LaTeX sources + distributed PDFs for homework & lectures
src/hw1/            Python package with problem scaffolds & utilities
    problem1.py     Autocorrelation + small-sample bias simulation
    problem2.py     Return forecasting (LASSO / Ridge / Elastic Net / NN)
    problem3.py     FF25 portfolios: tangency, pricing relation, log utility
    problem4.py     Hansen–Jagannathan bound (placeholder implementation)
    problem5.py     Mehra–Prescott equity premium illustration (placeholder)
    data.py         Synthetic data loaders (replace with real data ingestion)
    portfolio.py    Portfolio math helpers (Sharpe, tangency)
    ml.py           Thin wrappers around scikit-learn models
    consumption.py  Consumption growth helpers
    paths.py        Centralized path constants
run_all.py          Simple driver script demonstrating module usage
requirements.txt    Python dependencies
data/
  raw/              (empty placeholder; put source files like Hw1p45.xlsx here)
  processed/        (empty placeholder for cleaned / intermediate datasets)
tex/                LaTeX auxiliary build artifacts (logs, aux, etc.)
.gitignore          Ignore rules for Python, data, LaTeX aux files
```

---
## Quick Start (Python Code)
PowerShell (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_all.py   # smoke test of all problem scaffolds
```
bash (macOS/Linux):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` prints brief sanity outputs (autocorr summary, feature matrix shape, tangency Sharpe, etc.). Replace synthetic data loaders in `data.py` with real downloads (e.g., WRDS, CSV exports) when ready.

---
## Module Intent Summary
- Problem 1: Compute and simulate autocorrelation; demonstrate small-sample bias.
- Problem 2: Build lag & aggregated-return feature sets; run penalized linear models and a single-hidden-layer MLP (optional if scikit-learn installed).
- Problem 3: Use (synthetic) FF25 to examine tangency portfolio, pricing relation, log-utility weights, and full 25-asset tangency.
- Problem 4: Placeholder Hansen–Jagannathan bound summary (replace with real quarterly consumption + returns from Hw1p45.xlsx).
- Problem 5: Placeholder Mehra–Prescott style grid of risk aversion vs implied premium; add observed point from data.

Replace synthetic generators in `data.py` with real series: market returns, FF25, consumption, T‑bill returns. Keep interfaces consistent so downstream code runs unchanged.

---
## Building the LaTeX Documents
Compile any `.tex` sources in `ref/`:
```powershell
cd ref
pdflatex "Homework 1.tex"
pdflatex "Homework 1.tex"  # repeat or use latexmk
```
(Extend with `bibtex` if/when a bibliography is added.)

---
## Data Handling Guidance
Place original spreadsheets (e.g., Hw1p45.xlsx) in `data/raw/` (ignored by default except for a `.gitkeep`). Write cleaning scripts/notebooks to output processed parquet/CSV to `data/processed/`. Do not commit licensed / restricted raw data.

---
## Python Environment Notes
Key dependencies: numpy, pandas, scipy, statsmodels, scikit-learn (optional ML), matplotlib, seaborn, tqdm, pytest. All pinned loosely in `requirements.txt` for flexibility. If you do not need ML yet, you can comment out scikit-learn to speed environment creation.

---
## Version Control / Contribution Notes
- Keep raw data out of version control unless publicly distributable.
- Commit generated PDFs in `ref/` only if they serve as distributed artifacts.
- Avoid committing large intermediate datasets; prefer reproducible scripts.

---
## Suggested Next Enhancements
1. Replace synthetic loaders with real data ingestion.
2. Add unit tests for portfolio math and regression feature construction.
3. Add notebook(s) demonstrating full workflow & plots.
4. Implement real Hansen–Jagannathan frontier and Mehra–Prescott replication.
5. Add plotting utilities (matplotlib / seaborn) for factor and SDF diagnostics.

---
## Academic Integrity
Write-up must include interpretation, not just numerical results. Code and analysis should be your own; informal discussion allowed per syllabus.

---
## Contact
Course questions: Instructor (official channels). Repository issues: open an issue or annotate in commit messages.

---
## License
No license declared yet. Add one (MIT / BSD / Academic) if sharing beyond course.
