"""HW1 package containing problem-specific modules and shared utilities.

Each problem module exposes high-level functions that can be called from a notebook or
from a simple driver script (see run_all.py at project root once added).

Design goals:
- Keep raw data I/O separate (see data.py) so core logic is testable.
- Provide deterministic (seeded) simulation helpers for reproducibility.
- Favor pandas Series/DataFrame inputs/outputs for clarity.

Nothing here intentionally executes at import-time beyond lightweight version metadata.
"""
from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "problem1", "problem2", "problem3", "problem4", "problem5",
    "data", "portfolio", "ml", "consumption", "plotting", "paths"
]

try:
    __version__ = version("hw1")  # Will fail unless packaged; harmless in dev.
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0-dev"

