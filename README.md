_# Finance 395.4 – Empirical Methods in Asset Pricing

Repository for Homework 1 (and supportive lecture references) for Finance 395.4 "Empirical Methods in Asset Pricing" at the McCombs School of Business.

**Instructor:** Travis L. Johnson  
**Term:** Fall 2025  
**Assignment Focus:** Empirical methods, factor models, consumption-based asset pricing, and modern ML/AI tools applied to asset pricing research.

---
## Repository Layout
```
ref/        Source .tex and distributed PDF files for Homework 1 and lecture slide decks
src/        (Reserved) Python or data scripts for empirical exercises (currently empty)
tex/        LaTeX build / auxiliary artifacts (e.g., auxil/ for .aux, .log, etc.)
```
Current notable files:
- `ref/Homework 1.tex` – Main homework assignment write‑up (source)
- `ref/Lecture 1 - Overview of Empirical Asset Pricing.tex` (+ PDF)
- `ref/Lecture 2 - Consumption Based Asset Pricing.tex` (+ PDF)
- `ref/Lecture 3 - Machine Learning and AI.tex` (+ PDF)

Auxiliary logs (e.g., `tex/auxil/*.log`, `*.aux`) are separated to keep the root clean.

---
## Building the LaTeX Documents
You can compile any of the `.tex` sources in `ref/` into PDFs locally.

### Recommended Toolchain (Windows)
- Install a modern TeX distribution: [TeX Live](https://tug.org/texlive/) or [MiKTeX](https://miktex.org/)
- (Optional) Install `latexmk` for automated multi-pass builds
- Editor suggestions: VS Code with LaTeX Workshop extension, PyCharm with TeXiFy IDEA, or Cursor with LaTeX Workshop

### Quick Compile (single file)
From the repository root (PowerShell):
```powershell
cd ref
pdflatex "Homework 1.tex"
bibtex   "Homework 1"   # Only if bibliography is added later
pdflatex "Homework 1.tex"
pdflatex "Homework 1.tex"
```

### Output Location
By default PDFs will land in `ref/`. If you prefer isolating outputs, you can redirect using latexmk rc config or a build script later.

---
## Python Environment (Planned for Empirical Exercises)
A `.venv/` folder is present (local virtual environment). To (re)create from scratch:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # (Add this file when empirical code is introduced)
```
Currently no Python dependencies are tracked (no `requirements.txt` yet).

Planned usage of `src/`:
- Data ingestion & cleaning (CRSP/Compustat/WRDS style placeholders)
- Factor construction (Fama-French style, anomalies, characteristics)
- Regression or ML experiments (cross-sectional return prediction)

---
## Version Control / Contribution Notes
- Keep generated PDFs committed only if they are distributed artifacts (current PDFs in `ref/` are acceptable). Avoid committing transient `.log`, `.aux`, `.out`, `.synctex.gz` unless diagnosing build issues.
- Consider a `.gitignore` entry for common LaTeX aux files (future enhancement).

---
## Suggested Next Enhancements
1. Add a `requirements.txt` once empirical scripts begin.
2. Introduce a Makefile or simple PowerShell script to batch build all lecture/homework PDFs.
3. Add a `.gitignore` tuned for LaTeX + Python.
4. Provide data schema documentation for any datasets used in analysis.
5. Include reproducible notebooks for factor construction or ML demonstrations.

---
## Academic Integrity
This repository is intended for instructional and solution documentation purposes. If solutions are distributed, clarify permitted use for enrolled students. Avoid posting proprietary or restricted-access dataset contents.

---
## Contact
Questions or clarifications: reach out to Professor Travis L. Johnson via official course communication channels.

---
## License
No explicit license specified yet. Add one (e.g., MIT or an academic license) if broader distribution is intended.

