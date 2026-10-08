from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Callable, Sequence, Literal, overload

import cv2 as cv
import numpy as np
import tifffile
from tqdm import tqdm

from tcm_utils.file_dialogs import ask_directory, ask_open_file, find_repo_root


def beep(frequency_Hz: int = 1000, duration_ms: int = 200):
    """Play a beep sound using the system's default sound player.

    Parameters
    ----------
    frequency_Hz : int
        Frequency of the beep in Hertz (default: 1000).
    duration_ms : int
        Duration of the beep in milliseconds (default: 200).
    """
    try:
        import winsound
        winsound.Beep(frequency_Hz, duration_ms)
    except ImportError:
        # For non-Windows systems, use the 'beep' command if available
        # os.system(f'beep -f {frequency_Hz} -l {duration_ms}')
        print("\a", end="", flush=True)


def countdown_beep(
    frequency_Hz: int = 1000,
    duration_ms: int = 400,
) -> None:
    """Play a fixed ``3 2 1 BEEP`` countdown with precise 1 s cadence.

    Start times are scheduled exactly 1.0 second apart (1 -> 2 -> 3 -> BEEP).
    """

    frequency_Hz = max(37, int(frequency_Hz))
    duration_ms = max(1, int(duration_ms))
    countdown_frequency_Hz = max(37, frequency_Hz // 2)
    countdown_duration_ms = max(1, duration_ms // 2)

    start_time = time.perf_counter()
    for idx, (freq, dur) in enumerate(
        (
            (countdown_frequency_Hz, countdown_duration_ms),
            (countdown_frequency_Hz, countdown_duration_ms),
            (countdown_frequency_Hz, countdown_duration_ms),
            (frequency_Hz, duration_ms),
        )
    ):
        target_start = start_time + idx * 1.0
        sleep_s = target_start - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        beep(freq, dur)


def make_minimal_progress_bar(
    *,
    total: int | float,
    label: str,
    unit_label: str,
    bar_width: int = 16,
    postfix_width: int = 0,
    leave: bool = True,
) -> tqdm:
    """Create a minimal tqdm bar with constant bar width across labels."""
    count_width = max(1, len(str(int(math.ceil(total)))))
    bar_format = (
        f"{label}: {{bar}}| "
        f"{{n:>{count_width}.0f}}/{{total:.0f}} {unit_label}"
        f" {{postfix}}"
    )

    fixed_tail = f"| {0:>{count_width}d}/{total} {unit_label}"
    ncols = len(f"{label}: ") + bar_width + len(fixed_tail) + postfix_width

    return tqdm(
        total=total,
        unit=unit_label,
        ncols=ncols,
        leave=leave,
        bar_format=bar_format,
    )


def wait_with_progress(
    wait_s: float,
    *,
    label: str = "Waiting before starting next run",
    bar_width: int = 16,
    tick_s: float = 0.05,
) -> None:
    """Wait for ``wait_s`` with precise timing and integer-second bar updates."""
    if wait_s <= 0:
        return

    total_s = int(math.ceil(wait_s))
    start_time = time.perf_counter()
    next_tick = start_time

    with make_minimal_progress_bar(
        total=total_s,
        label=label,
        unit_label="s",
        bar_width=bar_width,
    ) as pbar:
        while (time.perf_counter() - start_time) < wait_s:
            next_tick += tick_s
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

            elapsed_whole_s = int(
                min(total_s, time.perf_counter() - start_time))
            if elapsed_whole_s > pbar.n:
                pbar.update(elapsed_whole_s - pbar.n)

        if pbar.n < total_s:
            pbar.update(total_s - pbar.n)


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt the user for a yes/no response."""
    answer = input(f"{prompt} ").strip().lower()
    if answer == "":
        return default
    return answer in {"y", "yes"}


@overload
def prompt_input(
    prompt: str,
    *,
    value_type: Literal["string"] = "string",
    allow_empty: Literal[False] = False,
    min_value: float | None = None,
    max_value: float | None = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
) -> str:
    ...


@overload
def prompt_input(
    prompt: str,
    *,
    value_type: Literal["string"] = "string",
    allow_empty: Literal[True],
    min_value: float | None = None,
    max_value: float | None = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
) -> str | None:
    ...


@overload
def prompt_input(
    prompt: str,
    *,
    value_type: Literal["float"],
    allow_empty: Literal[False] = False,
    min_value: float | None = None,
    max_value: float | None = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
) -> float:
    ...


@overload
def prompt_input(
    prompt: str,
    *,
    value_type: Literal["float"],
    allow_empty: Literal[True],
    min_value: float | None = None,
    max_value: float | None = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
) -> float | None:
    ...


@overload
def prompt_input(
    prompt: str,
    *,
    value_type: Literal["int"],
    allow_empty: Literal[False] = False,
    min_value: float | None = None,
    max_value: float | None = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
) -> int:
    ...


@overload
def prompt_input(
    prompt: str,
    *,
    value_type: Literal["int"],
    allow_empty: Literal[True],
    min_value: float | None = None,
    max_value: float | None = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
) -> int | None:
    ...


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


def ensure_non_empty_text(
    value: str | None,
    *,
    prompt: str,
    empty_error: str = "Input cannot be empty.",
) -> str:
    """Return a non-empty text value, prompting user when missing."""
    if value is not None:
        normalized = str(value).strip()
        if normalized:
            return normalized

    while True:
        prompted = prompt_input(prompt, value_type="string", allow_empty=True)
        normalized = "" if prompted is None else str(prompted).strip()
        if normalized:
            return normalized
        print(empty_error)


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
    temporary_output_dir: Path | None = None,
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

    When ``output_dir`` is omitted, processor outputs are isolated in
    ``temporary_output_dir`` until the destination is selected.

    Resolution order (no subfolder scanning):
    1) If ``input_path`` is a ``*_metadata.json`` file, return it.
    2) If ``input_path`` is a folder containing ``*_metadata.json``, return the latest one.
    3) If ``input_path`` is a matching source file, process it to ``output_dir`` or stage it before prompting for a destination.
    4) If ``input_path`` is a folder containing a matching source file, process it the same way.
    5) Otherwise, prompt the user to select a metadata JSON or source file.
    """

    def _latest_metadata(folder: Path) -> Path | None:
        return find_latest_in_directory(folder, metadata_pattern)

    def _latest_source(folder: Path) -> Path | None:
        return find_latest_in_directory(folder, source_patterns)

    def _move_staged_artifacts(
        staging_dir: Path,
        final_dir: Path,
        metadata_path: Path,
    ) -> Path:
        final_dir.mkdir(parents=True, exist_ok=True)
        metadata_relative = metadata_path.relative_to(staging_dir)
        for source_path in staging_dir.rglob("*"):
            if not source_path.is_file():
                continue
            relative_path = source_path.relative_to(staging_dir)
            destination_path = final_dir / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(destination_path))

        final_metadata_path = final_dir / metadata_relative
        with final_metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)

        repo_root = find_repo_root(Path(__file__))
        staging_root = staging_dir.resolve()
        final_root = final_dir.resolve()

        def _relocate_path_values(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: _relocate_path_values(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [_relocate_path_values(item) for item in value]
            if not isinstance(value, str):
                return value

            stored_path = Path(value)
            candidate = (
                stored_path if stored_path.is_absolute()
                else repo_root / stored_path
            ).resolve()
            try:
                relative_path = candidate.relative_to(staging_root)
            except ValueError:
                return value
            return path_relative_to(final_root / relative_path, repo_root)

        metadata = _relocate_path_values(metadata)
        save_metadata_json(metadata, final_metadata_path)

        staged_directories = sorted(
            (path for path in staging_dir.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in staged_directories:
            if not any(directory.iterdir()):
                directory.rmdir()
        if not any(staging_dir.iterdir()):
            staging_dir.rmdir()
        return final_metadata_path

    def _run_and_collect(source_path: Path, default_dir: Path) -> Path | None:
        if output_dir is None:
            temporary_root = (
                temporary_output_dir.expanduser().resolve()
                if temporary_output_dir is not None
                else (
                    find_repo_root(Path(__file__))
                    / ".temp"
                    / "processed_artifacts"
                )
            )
            temporary_root.mkdir(parents=True, exist_ok=True)
            dest = Path(tempfile.mkdtemp(prefix="run-", dir=temporary_root))
        else:
            dest = Path(output_dir).expanduser().resolve()
            dest.mkdir(parents=True, exist_ok=True)

        run_processor(source_path, dest)
        metadata_path = _latest_metadata(dest)
        if metadata_path is None:
            raise FileNotFoundError(
                f"Processor did not create {metadata_pattern} in {dest}"
            )

        if output_dir is None:
            chosen = ask_directory(
                key=output_dir_key,
                title=output_dir_title,
                default_dir=default_dir,
                start=Path(__file__),
            )
            if chosen is None:
                print(f"Outputs remain in {dest}")
                return metadata_path

            final_dir = Path(chosen).expanduser().resolve()
            if final_dir != dest.resolve():
                try:
                    final_dir.relative_to(dest.resolve())
                except ValueError:
                    pass
                else:
                    raise ValueError(
                        "The output directory cannot be inside the temporary "
                        f"staging directory: {dest}"
                    )
                metadata_path = _move_staged_artifacts(
                    dest, final_dir, metadata_path
                )
            print(f"Outputs saved to {metadata_path.parent}")

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
            return _run_and_collect(path, path.parent)

        if path.is_dir():
            source_in_dir = _latest_source(path)
            if source_in_dir:
                return _run_and_collect(source_in_dir, path)

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


def auto_brightness(
    image: np.ndarray,
    min_contrast: float = 40.0,
    target_brightness: float = 128.0,
    *,
    percentile_stretch: bool = False,
    low_percentile: float = 0.5,
    high_percentile: float = 99.95,
    limits: tuple[float, float] | None = None,
) -> np.ndarray:
    """Adjust image contrast using conditional brightening or percentile stretch.

    Processing steps
    ----------------
    The default conditional mode:
    1. Determine the intensity range (including 12-bit uint16 images).
    2. Measure contrast between the 2nd and 98th percentiles.
    3. Return the original image if it is flat or already has enough contrast.
    4. Otherwise, expand contrast to ``min_contrast``, shift the mean toward
       ``target_brightness``, clip to the intensity range, and preserve dtype.

    With ``percentile_stretch=True``:
    1. Validate the low and high percentiles.
    2. Use ``limits`` if supplied, or calculate limits from this image.
    3. Map the selected intensity range linearly to 0-255, clip values outside
       it, and return uint8. Flat input ranges become all-zero images.

    Shared ``limits`` let a sequence of images use consistent brightness.

    Parameters
    ----------
    image : np.ndarray
        Image data to adjust.
    min_contrast : float
        Minimum 2nd-to-98th percentile contrast for conditional brightening,
        expressed on an 8-bit scale.
    target_brightness : float
        Mean intensity target for conditional brightening, expressed on an
        8-bit scale.
    percentile_stretch : bool
        If True, use percentile stretching and return uint8 data instead.
    low_percentile : float
        Lower percentile used for stretching, from 0 to 100.
    high_percentile : float
        Upper percentile used for stretching, from 0 to 100.
    limits : tuple[float, float] | None
        Optional precomputed low and high intensity limits for stretching.

    Returns
    -------
    np.ndarray
        Adjusted image. The conditional path preserves unsigned image dtypes;
        percentile stretching returns uint8.
    """
    if percentile_stretch:
        if not 0 <= low_percentile < high_percentile <= 100:
            raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")

        array = np.asarray(image, dtype=np.float64)
        low, high = limits or tuple(
            float(value)
            for value in np.percentile(array, [low_percentile, high_percentile])
        )
        if high <= low:
            return np.zeros(array.shape, dtype=np.uint8)
        stretched = np.clip((array - low) / (high - low), 0.0, 1.0) * 255.0
        return np.rint(stretched).astype(np.uint8)

    if not 0 < min_contrast <= 255:
        raise ValueError("min_contrast must be in the range (0, 255]")
    if not 0 <= target_brightness <= 255:
        raise ValueError("target_brightness must be in the range [0, 255]")

    if image.dtype.kind == "u":
        image_max = float(np.iinfo(image.dtype).max)
        if image.dtype == np.uint16 and int(np.max(image)) <= 4095:
            image_max = 4095.0
    else:
        image_max = 255.0

    intensity_scale = image_max / 255.0
    effective_min_contrast = min_contrast * intensity_scale
    effective_target_brightness = target_brightness * intensity_scale

    low, high = np.percentile(image, (2, 98))
    contrast = float(high - low)
    if contrast == 0 or contrast >= effective_min_contrast:
        return image

    print(f"Auto-brightening image: contrast {contrast:.2f} "
          f"< {effective_min_contrast:.2f}, mean {np.mean(image):.2f} "
          f"-> target {effective_target_brightness:.2f}")
    scale = effective_min_contrast / contrast
    offset = effective_target_brightness - scale * float(np.mean(image))
    if image.dtype.kind != "u":
        return cv.convertScaleAbs(image, alpha=scale, beta=offset)
    adjusted = np.abs(image.astype(np.float64) * scale + offset)
    return np.rint(np.clip(adjusted, 0, image_max)).astype(image.dtype)


def load_image(path: Path) -> np.ndarray:
    """Read a single image file using OpenCV."""

    # If image is tif, use tifffile to read it, as OpenCV does not handle
    # 12-bit TIFF files correctly
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(path)
    else:
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
        Array of shape (n_images, y, x), preserving the source image dtype.
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


if __name__ == "__main__":
    # Example usage of the functions in this module
    directory = ask_directory(key="example_directory",
                              title="Select a directory")
    images = load_images(
        [directory / f for f in os.listdir(directory) if f.endswith('.tif')])

    print(f"Intensity range of loaded images: {images.min()} - {images.max()}")
    print(
        f"Average max intensity +- std across images: {images.max(axis=(1, 2)).mean()} +- {images.max(axis=(1, 2)).std()}")
