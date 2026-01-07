from __future__ import annotations

from pathlib import Path
import numpy as np


def path_relative_to(path: Path, root: Path) -> str:
    """Return a string path relative to root if possible, else absolute."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_two_column_numeric(path: Path, delimiter: str = ",") -> tuple[np.ndarray, np.ndarray]:
    """Load two numeric columns from a text/CSV file, handling a one-line header.

    Returns (y0, y1) where y0 is column 0 and y1 is column 1.
    """
    try:
        data = np.loadtxt(path, delimiter=delimiter)
    except Exception:
        data = np.loadtxt(path, delimiter=delimiter, skiprows=1)
    return data[:, 0], data[:, 1]
