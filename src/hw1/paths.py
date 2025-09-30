"""Centralized path utilities for the HW1 project.

Avoids hard-coding relative paths across modules. Import and use these helpers
so tests / notebooks can relocate the project root cleanly.
"""
from __future__ import annotations
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"

for _p in (DATA_ROOT, RAW_DATA, PROCESSED_DATA, CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

__all__ = [
    "PROJECT_ROOT", "SRC_ROOT", "DATA_ROOT", "RAW_DATA", "PROCESSED_DATA", "CACHE_DIR"
]

