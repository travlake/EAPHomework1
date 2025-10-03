"""
Converts all .csv files in the output directory to .tex files.
"""
import sys
from pathlib import Path
import pandas as pd

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hw1.paths import OUTPUT_DIR

def problem1_to_latex(file_path):
    """Converts the problem 1 solution CSV to a LaTeX tabular matching the layout.
    - Columns: Measure across rho(1..5) | spacer | Q(5), VR(5), delta(5)
    - Rows grouped by Panel and Frequency; Panel is a full-width multicolumn row.
    - Only outputs a tabular environment (no table/caption/label).
    """
    # Load data
    df = pd.read_csv(file_path, index_col=[0, 1, 2])

    # Define ordering
    panels = [idx for idx in df.index.get_level_values(0).unique()]
    # Prefer Panel A then Panel B if present
    panels_sorted = sorted(panels)
    freq_order = ['Daily', 'Monthly', 'Annual']
    measures = ['Estimate', 'Bias (Asy)', 'Bias (Sim)', 'SE (Asy)', 'SE (Sim)']
    rho_stats = [f'rho({k})' for k in range(1, 6)]
    other_stats = ['Q(5)', 'VR(5)', 'delta(5)']

    # Helper to fetch cell
    def get_val(panel, freq, stat, measure):
        try:
            return df.loc[(panel, freq, stat), measure]
        except KeyError:
            return float('nan')

    # Map measures for other_stats: for Daily, SE rows correspond to p-values; otherwise keep as-is
    def map_other_measure(freq: str, measure: str) -> str:
        if freq == 'Daily':
            if measure == 'SE (Asy)':
                return 'p-val (Asy)'
            if measure == 'SE (Sim)':
                return 'p-val (Sim)'
        return measure

    # Format numbers
    def fmt(x):
        if pd.isna(x):
            return ''
        # use 4 decimals, allow scientific for very small p-values
        try:
            if abs(x) < 1e-4 and x != 0:
                return f"{x:.2e}"
            return f"{x:.4f}"
        except Exception:
            return str(x)

    # Build LaTeX tabular manually
    # 2 text columns (Frequency, Measure), 5 rho numeric, 1 spacer, 3 numeric = 11 columns
    # Use alignment: l l r r r r r c r r r
    lines = []
    lines.append("\\begin{tabular}{l l r r r r r c r r r}")
    lines.append("\\hline")
    # Header row
    header_stats = ["$\\rho(1)$", "$\\rho(2)$", "$\\rho(3)$", "$\\rho(4)$", "$\\rho(5)$", '', "$Q(5)$", "$VR(5)$", "$\\delta(5)$"]
    lines.append("  &  & " + " & ".join(header_stats) + " \\\\")
    lines.append("\\hline")

    for panel in panels_sorted:
        # Panel header spanning all columns
        lines.append(rf"\multicolumn{{11}}{{c}}{{{panel}}} \\\\")
        lines.append("\\hline")

        for fi, freq in enumerate(freq_order):
            # Skip frequency if not present under this panel
            if (panel, freq) not in df.index.droplevel(2).unique():
                continue
            for mi, measure in enumerate(measures):
                freq_label = freq if mi == 0 else ''
                # Collect cells
                rho_vals = [fmt(get_val(panel, freq, s, measure)) for s in rho_stats]
                other_vals = [fmt(get_val(panel, freq, s, map_other_measure(freq, measure))) for s in other_stats]
                row = [freq_label, measure] + rho_vals + [''] + other_vals
                lines.append("  " + " & ".join(row) + " \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    return "\n".join(lines)


def problem2_to_latex(file_path):
    """Converts the problem 2 solution CSV to a LaTeX tabular (no table/caption)."""
    df = pd.read_csv(file_path, index_col=0)
    return df.to_latex(index=True, na_rep='', float_format="%.4f")


def problem3abc_to_latex(file_path):
    """Converts the problem 3abc solution CSV to two LaTeX tabulars (no table/caption)."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Find the split between the two tables
    split_index = [i for i, line in enumerate(lines) if "Tangency Portfolio Summary" in line][0]
    # First table (weights)
    weights_df = pd.read_csv(file_path, index_col=0, skiprows=1, nrows=4)
    # Second table (Sharpe ratio)
    summary_df = pd.read_csv(file_path, skiprows=split_index + 1)
    weights_latex = weights_df.to_latex(float_format="%.4f")
    summary_latex = summary_df.to_latex(index=False, float_format="%.4f")
    return weights_latex + "\n\n" + summary_latex


def problem3d_to_latex(file_path):
    """Converts the problem 3d solution CSV to a LaTeX tabular (no table/caption)."""
    # Read the main grid, skipping the title row
    df = pd.read_csv(file_path, index_col=0, skiprows=1, nrows=5)
    df.columns.name = 'Book-to-Market'
    df.index.name = 'Size'
    grid_latex = df.to_latex(float_format="%.4f")
    # Read the summary Sharpe Ratio
    summary_line = pd.read_csv(file_path, skiprows=8, header=None).iloc[0]
    summary_str = f"\\textbf{{{summary_line[0]}: {summary_line[1]:.4f}}}"
    return grid_latex + "\n" + summary_str


def main():
    """
    Main function to find all CSVs and convert them to LaTeX.
    """
    # Mapping from CSV filename to conversion function
    file_handlers = {
        "problem1_solution.csv": problem1_to_latex,
        "problem2_solution.csv": problem2_to_latex,
        "problem3abc_4_asset_results.csv": problem3abc_to_latex,
        "problem3d_25_asset_weights_grid.csv": problem3d_to_latex,
    }

    # Find all csv files in the output directory
    csv_files = list(OUTPUT_DIR.glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files to convert.")

    for csv_file in csv_files:
        if csv_file.name in file_handlers:
            print(f"Processing {csv_file.name}...")

            # Get the appropriate handler and generate LaTeX
            handler = file_handlers[csv_file.name]
            latex_content = handler(csv_file)

            # Define the output path
            output_path = csv_file.with_suffix(".tex")

            # Write the LaTeX content to the new file
            with open(output_path, 'w') as f:
                f.write(latex_content)

            print(f"  -> Saved LaTeX to {output_path}")
        else:
            print(f"Skipping {csv_file.name}, no handler defined.")

if __name__ == "__main__":
    main()
