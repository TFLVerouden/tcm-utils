from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Callable, Sequence, Literal

import cv2 as cv
import numpy as np
import tifffile
from tqdm import tqdm

from tcm_utils.file_dialogs import ask_directory, ask_open_file


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt the user for a yes/no response."""
    answer = input(f"{prompt}] ").strip().lower()
    if answer == "":
        return default
    return answer in {"y", "yes"}


def prompt_input(
    prompt: str,
    *,
    value_type: Literal["string", "float", "int"] = "string",
    allow_empty: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
) -> str | float | int | None:
    """Prompt the user for input with basic type and range validation.

    Parameters
    ----------
    prompt : str
        Prompt shown to the user.
    value_type : {"string", "float", "int"}
        Expected type. Numbers are parsed to the chosen type.
    allow_empty : bool
        If True, empty input returns ``None`` instead of re-prompting.
    min_value : float | None
        Minimum allowed value (inclusive by default).
    max_value : float | None
        Maximum allowed value (inclusive by default).
    exclusive_min : bool
        If True, ``min_value`` is treated as an exclusive bound.
    exclusive_max : bool
        If True, ``max_value`` is treated as an exclusive bound.

    Returns
    -------
    str | float | int | None
        Parsed value, or ``None`` when empty input is allowed and received.
    """

    expected = value_type.lower()
    if expected not in {"string", "float", "int"}:
        raise ValueError("value_type must be 'string', 'float', or 'int'")

    while True:
        raw = input(prompt).strip()

        if raw == "":
            if allow_empty:
                return None
            print("Input cannot be empty.")
            continue

        if expected == "string":
            return raw

        try:
            value_num: float | int
            if expected == "int":
                value_num = int(raw)
            else:
                value_num = float(raw)
        except ValueError:
            print("Invalid number. Please enter a numeric value.")
            continue

        if min_value is not None:
            if exclusive_min and value_num <= min_value:
                print(f"Please enter a value greater than {min_value}.")
                continue
            if not exclusive_min and value_num < min_value:
                print(f"Please enter a value of at least {min_value}.")
                continue

        if max_value is not None:
            if exclusive_max and value_num >= max_value:
                print(f"Please enter a value less than {max_value}.")
                continue
            if not exclusive_max and value_num > max_value:
                print(f"Please enter a value of at most {max_value}.")
                continue

        return value_num


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


def copy_file_to_raw_subfolder(
    file_path: Path,
    output_folder: Path,
    raw_subfolder_name: str = "raw_data",
) -> Path:
    """Copy a file to a raw data subfolder within the output folder.

    Parameters
    ----------
    file_path : Path
        Path to the file to copy (source is left untouched)
    output_folder : Path
        Parent folder where the raw subfolder should be created
    raw_subfolder_name : str
        Name of the raw data subfolder (default: "raw_data")

    Returns
    -------
    Path
        Path to the copied file in the raw_data folder
    """
    raw_dir = output_folder / raw_subfolder_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied_path = raw_dir / file_path.name

    if file_path.resolve() != copied_path.resolve():
        shutil.copy2(file_path, copied_path)

    return copied_path


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


def load_json_key(path: Path, key: str, default: Any | None = None) -> Any | None:
    """Load a JSON file and return a top-level key value.

    Parameters
    ----------
    path : Path
        Path to the JSON file.
    key : str
        Top-level key to retrieve.
    default : Any, optional
        Value to return if the key is missing (default: None).

    Returns
    -------
    Any | None
        The value for the key if present, otherwise ``default``.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get(key, default)


def ensure_path(
    value: str | Path | None,
    key: str,
    title: str | None = None,
    default_dir: Path | None = None,
) -> str | None:
    """Return a usable path string, prompting the user if missing.

    The caller supplies the current value (possibly empty/None/0). If it is
    missing, the user is asked to pick a directory. The dialog title includes
    the provided key to make the prompt clear.
    """

    is_missing = value is None or value == "" or value == 0
    if is_missing:
        selected = ask_directory(
            key=key,
            title=title or f"Select {key}",
            default_dir=default_dir,
        )
        if selected is None:
            print(f"WARNING: No {key} selected. Using default parameters.")
            return None
        return str(selected)

    return str(Path(value).expanduser())


def resolve_existing_path(value: str | Path | None) -> Path | None:
    """Expand and resolve a path-like value if it exists, else return None."""

    if value is None or value == "":
        return None

    candidate = Path(value).expanduser().resolve()
    return candidate if candidate.exists() else None


def find_latest_in_directory(
    folder: Path,
    pattern: str | Iterable[str],
) -> Path | None:
    """Return the most recently modified match for given glob pattern(s).

    Only direct children are considered; subdirectories are not searched.
    Accepts a single pattern (string) or an iterable of patterns.
    """

    if not folder.is_dir():
        return None

    patterns = (pattern,) if isinstance(pattern, str) else tuple(pattern)
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(folder.glob(pat))

    if not matches:
        return None

    return max(matches, key=lambda p: p.stat().st_mtime)


def ensure_processed_artifact(
    *,
    input_path: str | Path | None,
    output_dir: str | Path | None,
    metadata_pattern: str,
    source_patterns: Sequence[str],
    output_dir_key: str,
    output_dir_title: str,
    default_output_dir: Path,
    run_processor: Callable[[Path, Path], Any],
    prompt_key: str,
    prompt_title: str,
    prompt_filetypes: list[tuple[str, str]],
    start_path: Path,
) -> Path | None:
    """Return metadata path, running processing if needed.

    Resolution order (no subfolder scanning):
    1) If ``input_path`` is a ``*_metadata.json`` file, return it.
    2) If ``input_path`` is a folder containing ``*_metadata.json``, return the latest one.
    3) If ``input_path`` is a matching source file, process it to ``output_dir`` (or prompt) and return the created metadata JSON.
    4) If ``input_path`` is a folder containing a matching source file, process that file and return the created metadata JSON.
    5) Otherwise, prompt the user to select a metadata JSON or source file.
    """

    def _latest_metadata(folder: Path) -> Path | None:
        return find_latest_in_directory(folder, metadata_pattern)

    def _latest_source(folder: Path) -> Path | None:
        return find_latest_in_directory(folder, source_patterns)

    def _resolve_output_folder(default_dir: Path) -> Path | None:
        chosen = ensure_path(
            value=output_dir,
            key=output_dir_key,
            title=output_dir_title,
            default_dir=default_dir,
        )
        if chosen is None:
            print("No output directory selected.")
            return None
        dest = Path(chosen).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def _run_and_collect(source_path: Path, dest: Path) -> Path | None:
        run_processor(source_path, dest)
        metadata_path = _latest_metadata(dest)
        if metadata_path:
            copy_target = source_path.parent / metadata_path.name
            if metadata_path.resolve() != copy_target.resolve():
                shutil.copy2(metadata_path, copy_target)
        return metadata_path

    def _handle_candidate(path: Path | None) -> Path | None:
        if path is None:
            return None

        if path.is_file() and path.name.endswith("_metadata.json"):
            return path

        if path.is_dir():
            existing = _latest_metadata(path)
            if existing:
                return existing

        if path.is_file() and any(path.match(pat) for pat in source_patterns):
            output_folder = _resolve_output_folder(path.parent)
            if output_folder is None:
                return None
            return _run_and_collect(path, output_folder)

        if path.is_dir():
            source_in_dir = _latest_source(path)
            if source_in_dir:
                output_folder = _resolve_output_folder(path)
                if output_folder is None:
                    return None
                return _run_and_collect(source_in_dir, output_folder)

        return None

    candidate_path = resolve_existing_path(input_path)
    result = _handle_candidate(candidate_path)
    if result is not None:
        return result

    selection = ask_open_file(
        key=prompt_key,
        title=prompt_title,
        filetypes=prompt_filetypes,
        default_dir=default_output_dir,
        start=start_path,
    )
    if not selection:
        print("No file selected.")
        return None

    selection_path = resolve_existing_path(selection)
    return _handle_candidate(selection_path)


def load_image(path: Path) -> np.ndarray:
    """Read a single image file using OpenCV."""

    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def load_images(
    image_paths: Sequence[str | Path],
    *,
    n_jobs: int | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Load a list of image files into a 3D array.

    Parameters
    ----------
    image_paths : Sequence[str | Path]
        Iterable of image paths (e.g. from ``init_config._get_image_list``).
    n_jobs : int | None
        Max workers for ``ThreadPoolExecutor``. Defaults to ``os.cpu_count()``.
    show_progress : bool
        If True, wrap the loader in a tqdm progress bar.

    Returns
    -------
    np.ndarray
        Array of shape (n_images, y, x) with dtype ``uint8``.
    """

    if not image_paths:
        raise ValueError("image_paths must not be empty")

    resolved_paths = [Path(p).expanduser() for p in image_paths]
    max_workers = n_jobs or (os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator = executor.map(load_image, resolved_paths)
        if show_progress:
            iterator = tqdm(iterator, total=len(resolved_paths),
                            desc="Loading images", leave=False)
            # TODO: I don't think tqdm is working here
        images = list(iterator)

    return np.stack(images, axis=0)


def load_metadata(filepath):
    """
    Load previously saved CIHX metadata from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to the JSON file containing saved metadata

    Returns
    -------
    dict
        Dictionary containing the loaded metadata
    """
    filepath = Path(filepath)

    # Ensure file has correct extension
    if not filepath.suffix == '.json':
        filepath = filepath.with_suffix('.json')

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Metadata file not found: {filepath}")

    # Load the data
    with filepath.open('r', encoding='utf-8') as fh:
        loaded_data = json.load(fh)

    print(f"Loaded metadata from {filepath}")
    return loaded_data
