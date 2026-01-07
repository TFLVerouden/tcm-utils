from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
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


def save_metadata_json(
    metadata: dict[str, Any],
    output_path: Path,
    indent: int = 2,
) -> Path:
    """Save metadata dictionary as a JSON file.

    Parameters
    ----------
    metadata : dict
        Dictionary containing metadata to save
    output_path : Path
        Path where the JSON file should be saved
    indent : int
        Indentation level for JSON formatting (default: 2)

    Returns
    -------
    Path
        Path to the saved metadata file
    """
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=indent)
    return output_path


def move_to_raw_subfolder(
    file_path: Path,
    output_folder: Path,
    raw_subfolder_name: str = "raw_data",
) -> Path:
    """Move a file to a raw data subfolder within the output folder.

    Parameters
    ----------
    file_path : Path
        Path to the file to move
    output_folder : Path
        Parent folder where the raw subfolder should be created
    raw_subfolder_name : str
        Name of the raw data subfolder (default: "raw_data")

    Returns
    -------
    Path
        Path to the moved file
    """
    raw_dir = output_folder / raw_subfolder_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    moved_path = raw_dir / file_path.name

    if file_path.resolve() != moved_path.resolve():
        shutil.move(str(file_path), str(moved_path))

    return moved_path


def create_timestamped_filename(
    base_name: str,
    timestamp: str,
    suffix: str,
    extension: str,
) -> str:
    """Create a filename with timestamp and suffix.

    Parameters
    ----------
    base_name : str
        Base filename (without extension)
    timestamp : str
        Timestamp string to include
    suffix : str
        Suffix to add (e.g., "metadata", "plot", "calibration")
    extension : str
        File extension (with or without leading dot)

    Returns
    -------
    str
        Formatted filename

    Examples
    --------
    >>> create_timestamped_filename("test", "20260107_123456", "metadata", "json")
    'test_20260107_123456_metadata.json'
    """
    if not extension.startswith("."):
        extension = f".{extension}"
    return f"{base_name}_{timestamp}_{suffix}{extension}"
