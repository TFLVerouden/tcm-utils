"""Create Keynote-friendly H.264 videos from numbered TIFF frames."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from decimal import Decimal
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from tcm_utils.file_dialogs import ask_directory, find_repo_root
from tcm_utils.io_utils import load_image, load_metadata, prompt_input, prompt_yes_no
from tcm_utils.read_cihx import extract_cihx_metadata, recursive_search

_FRAME_NUMBER = re.compile(r"(\d+)\.(?:tif|tiff)$", re.IGNORECASE)
_FRAME_RATE_KEYS = {
    "recordrate",
    "framerate",
    "frame_rate",
    "framerate_hz",
    "recordingframerate",
    "recording_frame_rate",
    "recording_fps",
    "fps",
}
_TIME_FACTORS = {"s": 1.0, "ms": 1_000.0, "us": 1_000_000.0}
_LABEL_LOCATIONS = {
    "upper left": (0.0, 0.0),
    "upper center": (0.5, 0.0),
    "upper right": (1.0, 0.0),
    "center left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "center right": (1.0, 0.5),
    "lower left": (0.0, 1.0),
    "lower center": (0.5, 1.0),
    "lower right": (1.0, 1.0),
    "right": (1.0, 0.5),
}


def _frame_number(path: Path) -> int:
    match = _FRAME_NUMBER.search(path.name)
    if match is None:
        raise ValueError(
            f"TIFF filename must end in a frame number before its extension: {path.name}"
        )
    return int(match.group(1))


def _select_frame_paths(
    frames_dir: Path,
    frames_range: tuple[int, int] | None,
) -> list[tuple[int, Path]]:
    if not frames_dir.is_dir():
        raise NotADirectoryError(
            f"Frames directory does not exist: {frames_dir}")

    all_paths = [
        path
        for path in frames_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
        and not path.name.startswith(".")
    ]
    if not all_paths:
        raise FileNotFoundError(f"No TIFF frame files found in {frames_dir}")

    numbered_paths = [(_frame_number(path), path) for path in all_paths]
    numbered_paths.sort(key=lambda item: (item[0], item[1].name))
    frame_numbers = [number for number, _ in numbered_paths]
    if len(frame_numbers) != len(set(frame_numbers)):
        raise ValueError(
            f"Multiple TIFF files have the same trailing frame number in {frames_dir}"
        )

    if frames_range is not None:
        start, end = frames_range
        if start > end:
            raise ValueError(
                "frames_range start must be less than or equal to end")
        numbered_paths = [
            item for item in numbered_paths if start <= item[0] <= end
        ]
        if not numbered_paths:
            raise FileNotFoundError(
                f"No TIFF frames found for inclusive frame range {start}–{end}"
            )

    return numbered_paths


def _numeric_rate(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(rate) and rate > 0:
        return rate
    return None


def _find_frame_rate(metadata: object) -> float | None:
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if str(key).lower() in _FRAME_RATE_KEYS:
                rate = _numeric_rate(value)
                if rate is not None:
                    return rate
        for value in metadata.values():
            rate = _find_frame_rate(value)
            if rate is not None:
                return rate
    elif isinstance(metadata, list):
        for value in metadata:
            rate = _find_frame_rate(value)
            if rate is not None:
                return rate
    return None


def _get_recording_frame_rate(frames_dir: Path) -> float:
    metadata_paths = sorted(
        (
            path
            for path in frames_dir.glob("*.json")
            if "metadata" in path.name.lower() or "camera" in path.name.lower()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for metadata_path in metadata_paths:
        rate = _find_frame_rate(load_metadata(metadata_path))
        if rate is not None:
            return rate

    cihx_paths = sorted(
        (*frames_dir.glob("*.cihx"), *frames_dir.glob("*.cih")),
        key=lambda path: path.name.lower(),
    )
    for cihx_path in cihx_paths:
        metadata = extract_cihx_metadata(
            cihx_path,
            output_folder=frames_dir,
            save=False,
            verbose=False,
            copy_raw=False,
        )
        rate = _numeric_rate(recursive_search(metadata, "recordRate"))
        if rate is not None:
            return rate

    rate = prompt_input(
        "Recording frame rate in frames per second:",
        value_type="float",
        min_value=0,
        exclusive_min=True,
    )
    return float(rate)


def _iter_loaded_frames(
    numbered_paths: list[tuple[int, Path]],
    n_jobs: int | None,
) -> Iterator[tuple[int, Path, np.ndarray]]:
    if n_jobs is not None and (
        isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1
    ):
        raise ValueError("n_jobs must be a positive integer or None")

    worker_count = n_jobs or min(4, os.cpu_count() or 1)
    worker_count = min(worker_count, len(numbered_paths))
    pending_limit = worker_count * 2
    path_iterator = iter(numbered_paths)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending: deque[tuple[int, Path, Future[np.ndarray]]] = deque()
        for _ in range(pending_limit):
            try:
                number, path = next(path_iterator)
            except StopIteration:
                break
            pending.append((number, path, executor.submit(load_image, path)))

        while pending:
            number, path, future = pending.popleft()
            yield number, path, future.result()
            try:
                next_number, next_path = next(path_iterator)
            except StopIteration:
                continue
            pending.append(
                (next_number, next_path, executor.submit(load_image, next_path))
            )


def _validate_frame(
    image: np.ndarray,
    path: Path,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(
            f"Expected one grayscale image per TIFF; got shape {image.shape} in {path}"
        )
    if image.dtype.kind != "u" or image.dtype.itemsize not in (1, 2):
        raise ValueError(
            f"Expected 8-bit or 16-bit unsigned grayscale data; got {image.dtype} in {path}"
        )
    if expected_shape is not None and image.shape != expected_shape:
        raise ValueError(
            f"Frame dimensions differ: expected {expected_shape}, got {image.shape} in {path}"
        )
    return image


def _crop_frame(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Crop one grayscale frame using (y_start, y_end, x_start, x_end)."""
    if roi is None:
        return image
    if (
        not isinstance(roi, tuple)
        or len(roi) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in roi
        )
    ):
        raise ValueError(
            "roi must be a tuple of four integers "
            "(y_start, y_end, x_start, x_end)"
        )

    height, width = image.shape
    y_start, y_end, x_start, x_end = (int(value) for value in roi)
    if y_start < 0:
        y_start += height
    if x_start < 0:
        x_start += width
    if y_end == 0:
        y_end = height
    elif y_end < 0:
        y_end += height
    if x_end == 0:
        x_end = width
    elif x_end < 0:
        x_end += width

    if not (
        0 <= y_start < height
        and 0 <= y_end <= height
        and 0 <= x_start < width
        and 0 <= x_end <= width
    ):
        raise ValueError(
            "roi coordinates are out of bounds of the image dimensions")
    if y_end <= y_start or x_end <= x_start:
        raise ValueError("roi must select a non-empty image region")
    return image[y_start:y_end, x_start:x_end]


def _sequence_contrast_limits(
    numbered_paths: list[tuple[int, Path]],
    n_jobs: int | None,
    low_percentile: float,
    high_percentile: float,
    roi: tuple[int, int, int, int] | None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[float, float]]:
    histogram = np.zeros(65_536, dtype=np.uint64)
    source_shape: tuple[int, int] | None = None
    cropped_shape: tuple[int, int] | None = None

    for _, path, loaded in tqdm(
        _iter_loaded_frames(numbered_paths, n_jobs),
        total=len(numbered_paths),
        desc="Analyzing TIFF frames",
        leave=False,
    ):
        image = _validate_frame(loaded, path, source_shape)
        if source_shape is None:
            source_shape = image.shape
        image = _crop_frame(image, roi)
        if cropped_shape is None:
            cropped_shape = image.shape
        counts = np.bincount(
            image.reshape(-1).astype(np.int64), minlength=65_536)
        histogram += counts.astype(np.uint64, copy=False)

    cumulative = np.cumsum(histogram)
    total_pixels = int(cumulative[-1])
    low_limit = float(
        np.searchsorted(cumulative, low_percentile * total_pixels / 100)
    )
    high_limit = float(
        np.searchsorted(cumulative, high_percentile * total_pixels / 100)
    )
    assert source_shape is not None and cropped_shape is not None
    return source_shape, cropped_shape, (low_limit, high_limit)


def _auto_brightness(
    image: np.ndarray,
    low_percentile: float = 0.5,
    high_percentile: float = 99.95,
    *,
    limits: tuple[float, float] | None = None,
) -> np.ndarray:
    """Stretch grayscale values to 8-bit, optionally using shared sequence limits."""
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


def _format_time(
    frame_number: int,
    recording_rate: float,
    unit: str,
    first_frame_number: int,
) -> str:
    seconds = (frame_number - first_frame_number) / recording_rate
    value = seconds * _TIME_FACTORS[unit]
    time_step = _TIME_FACTORS[unit] / recording_rate
    precision = max(
        0,
        -Decimal(format(time_step, ".3g")).as_tuple().exponent,
    )
    return f"{value:.{precision}f} {unit}"


def _resolve_label_color(
    color: str | int,
    parameter_name: str = "label_color",
) -> int:
    """Convert a grayscale name or intensity to an 8-bit gray value."""
    if isinstance(color, str):
        named_colors = {"black": 0, "white": 255, "gray": 128, "grey": 128}
        try:
            return named_colors[color.lower()]
        except KeyError as error:
            raise ValueError(
                f"{parameter_name} must be black, white, gray/grey, or an integer "
                "grayscale value from 0 to 255"
            ) from error

    if isinstance(color, bool) or not isinstance(color, int) or not 0 <= color <= 255:
        raise ValueError(
            f"{parameter_name} must be black, white, gray/grey, or an integer "
            "grayscale value from 0 to 255"
        )
    return color


def _resolve_label_location(
    location: str | tuple[float, float],
) -> tuple[float, float]:
    """Return normalized label placement, accepting legend-like names."""
    if isinstance(location, str):
        try:
            return _LABEL_LOCATIONS[location.lower()]
        except KeyError as error:
            choices = ", ".join(_LABEL_LOCATIONS)
            raise ValueError(
                f"Unknown label_location {location!r}; choose one of {choices} "
                "or pass an (x, y) pair from 0 to 1"
            ) from error

    if (
        not isinstance(location, tuple)
        or len(location) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in location
        )
    ):
        raise ValueError(
            "label_location must be a Matplotlib-style position name or "
            "an (x, y) pair from 0 to 1"
        )
    return float(location[0]), float(location[1])


def _add_time_label(
    image: np.ndarray,
    label: str,
    *,
    font: ImageFont.FreeTypeFont,
    color: int,
    stroke_color: int | None,
    location: tuple[float, float],
    font_size_px: int,
) -> np.ndarray:
    """Draw the timestamp in grayscale, anchored within the frame."""
    frame = Image.fromarray(image).convert("L")
    drawing = ImageDraw.Draw(frame)
    stroke_width = (
        max(1, round(font_size_px / 12))
        if stroke_color is not None
        else 0
    )
    stroke_bounds = drawing.textbbox(
        (0, 0),
        label,
        font=font,
        stroke_width=stroke_width,
    )
    text_width = stroke_bounds[2] - stroke_bounds[0]
    text_height = stroke_bounds[3] - stroke_bounds[1]

    # Named locations and normalized coordinates both select a point within
    # the available margin-to-margin space for the complete label.
    margin = max(8, round(font_size_px / 2))
    max_left = frame.width - text_width - margin
    max_top = frame.height - text_height - margin
    if max_left < margin or max_top < margin:
        raise ValueError(
            f"Time label at {font_size_px}px does not fit in "
            f"{frame.width}x{frame.height} frame"
        )
    left = round(margin + (max_left - margin) * location[0])
    top = round(margin + (max_top - margin) * location[1])

    drawing.text(
        (left - stroke_bounds[0], top - stroke_bounds[1]),
        label,
        font=font,
        fill=color,
        stroke_width=stroke_width,
        stroke_fill=stroke_color,
    )
    return np.asarray(frame)


def _show_frame_preview(
    image: np.ndarray,
    frame_path: Path,
    label: str,
    *,
    font: ImageFont.FreeTypeFont,
    color: int,
    stroke_color: int | None,
    location: tuple[float, float],
    font_size_px: int,
) -> bool:
    """Display one prepared frame so the user can review its appearance."""
    # Build the preview with the same overlay and contrast operation as output.
    preview = _add_time_label(
        _auto_brightness(image),
        label,
        font=font,
        color=color,
        stroke_color=stroke_color,
        location=location,
        font_size_px=font_size_px,
    )

    # Use Tk for both the preview and the later folder picker. On macOS,
    # Matplotlib's native backend and Tk can conflict in the same process.
    from tkinter import Tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = Tk()
    root.title(f"Preview: {frame_path.name}")
    figure = Figure(figsize=(10, 7))
    axis = figure.add_subplot(111)
    axis.imshow(preview, cmap="gray", vmin=0, vmax=255)
    axis.set_title(
        f"{frame_path.name}\n"
        "Preview contrast is based on this frame; the video uses one "
        "sequence-wide contrast range."
    )
    axis.axis("off")
    figure.tight_layout()
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Center on Tk's default display, then raise the static preview above other
    # windows before waiting for the terminal response.
    root.update_idletasks()
    x = max(0, (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2)
    y = max(0, (root.winfo_screenheight() - root.winfo_reqheight()) // 2)
    root.geometry(f"+{x}+{y}")
    root.lift()

    # Render the static preview without entering Tk's event loop, so the user
    # can answer in the terminal and Enter can close the preview immediately.
    root.update()
    try:
        return prompt_yes_no(
            "Does this preview look good? Press ENTER to continue.",
            default=True,
        )
    finally:
        root.destroy()


def make_video(
    frames_dir: str | Path | None = None,
    frames_range: tuple[int, int] | None = None,
    output_path: str | Path | None = None,
    time_stretch_s_per_s: float | None = 0.002,
    output_frame_rate: float | None = None,
    time_label_unit: str = "ms",
    recording_frame_rate: float | None = None,
    n_jobs: int | None = None,
    confirm: bool = True,
    show_preview: bool = True,
    label_font_path: str | Path | None = None,
    label_font_style: str = "regular",
    label_font_size_px: int = 48,
    label_color: str | int = "black",
    label_location: str | tuple[float, float] = "upper left",
    label_stroke_color: str | int | None = None,
    crop_roi: tuple[int, int, int, int] | None = None,
) -> Path | None:
    """Encode numbered grayscale TIFF frames as an H.264 MP4.

    ``frames_range`` uses the trailing filename number and includes both ends.
    The output rate defaults to recording rate times ``time_stretch_s_per_s``;
    the default stretch of 0.002 therefore makes 20,000-fps footage play at
    40 fps, retaining every selected frame. ``output_frame_rate`` overrides
    that derived rate. Frame labels use elapsed time from the first numbered
    TIFF in the folder, so that frame displays 0 and selected later frames
    retain their original offsets. The video is always encoded into ``<repo>/.temp``
    first and then moved to ``output_path``; when ``output_path`` is None, a
    folder picker is shown afterwards (cancelling leaves the video in .temp).
    ``crop_roi`` optionally crops every frame before preview and encoding, using
    ``(y_start, y_end, x_start, x_end)`` coordinates. Negative coordinates are
    offsets from the corresponding image edge, and zero end coordinates mean
    the full extent in that direction.
    With ``confirm=True`` (the default), ``show_preview=True`` displays the
    first selected frame with a timestamp and a quick per-frame contrast
    stretch before the sequence-wide contrast analysis begins.
    ``label_font_style`` specifies a fallback font style to use if the
    primary font is not available.
    ``label_font_path`` selects a TrueType/OpenType font (PT Sans by default),
    ``label_font_size_px`` sets its pixel size, ``label_color`` accepts black,
    white, gray/grey, or a grayscale integer from 0 to 255.
    ``label_stroke_color`` accepts the same colors; ``None`` (the default)
    disables the stroke. ``label_location`` accepts a common legend-style
    location or a normalized (x, y) tuple.
    """
    # Ask for the TIFF folder only when the caller did not supply one.
    if frames_dir is None:
        selected_dir = ask_directory(
            key="video_maker_input",
            title="Select the directory containing the frames",
        )
        if not selected_dir:
            print("No directory selected. Exiting.")
            return None
        frames_dir = selected_dir
    # Resolve the folder so later file and metadata lookups use a consistent path.
    frames_dir = Path(frames_dir).expanduser().resolve()

    # Reject invalid label units and playback rates before reading potentially
    # large image sequences.
    if time_label_unit not in _TIME_FACTORS:
        raise ValueError(
            f"time_label_unit must be one of {tuple(_TIME_FACTORS)}")
    if time_stretch_s_per_s is not None and (
        not math.isfinite(time_stretch_s_per_s) or time_stretch_s_per_s <= 0
    ):
        raise ValueError("time_stretch_s_per_s must be positive or None")
    if output_frame_rate is not None and (
        not math.isfinite(output_frame_rate) or output_frame_rate <= 0
    ):
        raise ValueError("output_frame_rate must be positive or None")
    if (
        isinstance(label_font_size_px, bool)
        or not isinstance(label_font_size_px, int)
        or label_font_size_px <= 0
    ):
        raise ValueError("label_font_size_px must be a positive integer")

    # Resolve the label style once and reuse its font during preview and encoding.
    if label_font_path is not None:
        font_path = Path(label_font_path).expanduser().resolve()
    elif label_font_style.lower() == "regular":
        font_path = Path(__file__).parent / "fonts" / "PTSans-Regular.ttf"
    elif label_font_style.lower() == "bold":
        font_path = Path(__file__).parent / "fonts" / "PTSans-Bold.ttf"
    elif label_font_style.lower() == "italic":
        font_path = Path(__file__).parent / "fonts" / "PTSans-Italic.ttf"
    elif label_font_style.lower() == "bold italic":
        font_path = Path(__file__).parent / "fonts" / "PTSans-BoldItalic.ttf"
    else:
        raise ValueError(
            "label_font_style must be one of 'regular', 'bold', 'italic', or 'bold italic'"
        )

    label_font = ImageFont.truetype(str(font_path), size=label_font_size_px)
    label_gray = _resolve_label_color(label_color)
    stroke_gray = None
    if label_stroke_color is not None:
        stroke_gray = _resolve_label_color(
            label_stroke_color,
            "label_stroke_color",
        )
    label_anchor = _resolve_label_location(label_location)

    # Select the requested TIFFs by their trailing frame numbers and determine
    # the camera's recording rate from metadata, or accept a caller-supplied rate.
    numbered_paths = _select_frame_paths(frames_dir, frames_range)
    # Use the earliest numbered TIFF in the folder as the time origin so a
    # selection starting later keeps its offset from the recording's first frame.
    first_frame_number = min(
        _frame_number(path)
        for path in frames_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
        and not path.name.startswith(".")
    )
    if recording_frame_rate is None:
        recording_frame_rate = _get_recording_frame_rate(frames_dir)
    elif not math.isfinite(recording_frame_rate) or recording_frame_rate <= 0:
        raise ValueError("recording_frame_rate must be positive")

    # Derive the playback rate from the requested stretch unless the caller
    # explicitly overrides the output rate.
    stretch = 1.0 if time_stretch_s_per_s is None else time_stretch_s_per_s
    frame_rate = output_frame_rate or recording_frame_rate * stretch
    frame_rate_text = format(frame_rate, ".12g")

    # The video is always encoded into the repo's .temp folder first. The final
    # destination is the caller's output_path, or (when None) a folder the user
    # is asked to pick once encoding has finished. This function writes H.264
    # only into an MP4 container.
    temp_dir = find_repo_root(Path(__file__)) / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    final_path: Path | None = None
    if output_path is not None:
        final_path = Path(output_path).expanduser().resolve()
        if final_path.suffix.lower() != ".mp4":
            raise ValueError("H.264 video output must use the .mp4 extension")
    temp_video_path = temp_dir / \
        (final_path.name if final_path else "video.mp4")

    # Preview a single selected TIFF before loading the whole sequence for its
    # shared contrast range; this gives quick visual feedback on the edits.
    if confirm and show_preview:
        preview_number, preview_path = numbered_paths[0]
        preview_image = _validate_frame(load_image(preview_path), preview_path)
        preview_image = _crop_frame(preview_image, crop_roi)
        if preview_image.shape[0] % 2 or preview_image.shape[1] % 2:
            raise ValueError(
                "H.264 4:2:0 requires even frame dimensions; "
                f"got {preview_image.shape[1]}x{preview_image.shape[0]}"
            )
        preview_label = _format_time(
            preview_number,
            recording_frame_rate,
            time_label_unit,
            first_frame_number,
        )
        print(
            "Reviewing one-frame preview before analyzing all selected TIFFs. "
            "The preview uses per-frame contrast; final contrast is computed "
            "across the sequence."
        )
        if not _show_frame_preview(
            preview_image,
            preview_path,
            preview_label,
            font=label_font,
            color=label_gray,
            stroke_color=stroke_gray,
            location=label_anchor,
            font_size_px=label_font_size_px,
        ):
            return None

    # Inspect the sequence once to ensure all frames have matching dimensions
    # and to calculate one shared contrast range, preventing brightness flicker.
    source_shape, expected_shape, contrast_limits = _sequence_contrast_limits(
        numbered_paths, n_jobs, 0.5, 99.95, crop_roi
    )
    height, width = expected_shape
    # The selected H.264 pixel format requires even dimensions, so fail rather
    # than silently resizing or padding source frames.
    if width % 2 or height % 2:
        raise ValueError(
            f"H.264 4:2:0 requires even frame dimensions; got {width}x{height}"
        )

    # Show the user the expected encoding settings and allow cancellation before
    # starting the more expensive encode.
    duration_s = len(numbered_paths) / frame_rate
    destination_text = (
        ", saved to " + str(final_path) if final_path else ""
    )
    video_description = "Original video" if crop_roi is None else "Cropped video"
    summary = (
        f"{video_description}: {len(numbered_paths)} .TIFF images ({width}x{height}) at {recording_frame_rate:g} fps.\n"
        f"Create an H.264 MP4 video that plays back at {frame_rate:g} fps "
        f"({duration_s:.3f} s){destination_text}?"
    )
    if confirm and not prompt_yes_no(summary, default=True):
        return None
    # Fail early for an explicit destination that cannot be overwritten, instead
    # of after a long encode. The check is repeated when the video is delivered.
    if final_path is not None and final_path.exists() and not confirm:
        raise FileExistsError(f"Output video already exists: {final_path}")

    # Locate FFmpeg explicitly so a missing encoder produces an actionable error.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError(
            "FFmpeg is required to create H.264 MP4 files; install FFmpeg and retry."
        )

    # Encode to a scratch file inside .temp so a failed encode cannot leave a
    # partial video at the temp video path or the final destination.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{temp_video_path.stem}.",
        suffix=".tmp.mp4",
        dir=temp_dir,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    # Configure FFmpeg to read raw grayscale frames from stdin, retain their
    # dimensions and square pixels, and encode an H.264 MP4 with fast-start layout.
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "gray",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        frame_rate_text,
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "format=yuv420p,setsar=1",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]

    # Start the encoder and stream frames to it in order rather than buffering
    # the whole video in memory.
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    completed = False
    try:
        assert process.stdin is not None
        # Load TIFFs with bounded threaded prefetch, apply the shared contrast
        # mapping, draw each source-time label, and send each frame to FFmpeg.
        for frame_number, frame_path, loaded in tqdm(
            _iter_loaded_frames(numbered_paths, n_jobs),
            total=len(numbered_paths),
            desc="Encoding video",
        ):
            frame = _validate_frame(loaded, frame_path, source_shape)
            frame = _crop_frame(frame, crop_roi)
            frame = _validate_frame(frame, frame_path, expected_shape)
            frame = _auto_brightness(frame, limits=contrast_limits)
            label = _format_time(
                frame_number,
                recording_frame_rate,
                time_label_unit,
                first_frame_number,
            )
            frame = _add_time_label(
                frame,
                label,
                font=label_font,
                color=label_gray,
                stroke_color=stroke_gray,
                location=label_anchor,
                font_size_px=label_font_size_px,
            )
            process.stdin.write(np.ascontiguousarray(frame).tobytes())

        # Signal the end of raw video input, collect FFmpeg's result, and only
        # promote the scratch file to the temp video when the encode succeeded.
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
        if return_code:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg failed to encode the video: {message}")
        os.replace(temporary_path, temp_video_path)
        completed = True
    except BrokenPipeError as error:
        # Include FFmpeg's diagnostic if it exited before accepting all frames.
        stderr = process.stderr.read() if process.stderr is not None else b""
        process.wait()
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"FFmpeg stopped while encoding the video: {message}") from error
    finally:
        # Ensure the subprocess and temporary file are cleaned up on every
        # failure path while leaving a successfully encoded video untouched.
        if process.poll() is None:
            process.kill()
            process.wait()
        if not completed:
            temporary_path.unlink(missing_ok=True)

    # Without an explicit destination, ask the user where the finished video
    # should go; cancelling leaves it in .temp.
    if final_path is None:
        chosen_dir = ask_directory(
            key=None,
            title="Select the folder to save the video in",
            default_dir=frames_dir,
        )
        if not chosen_dir:
            print(f"No folder selected. Video left at {temp_video_path}")
            return temp_video_path
        final_path = Path(chosen_dir).expanduser(
        ).resolve() / temp_video_path.name

    # Move the finished video from .temp to its destination, asking before
    # replacing an existing file.
    if final_path.exists():
        if not confirm:
            raise FileExistsError(
                f"Output video  already exists: {final_path}")
        if not prompt_yes_no(
            f"Overwrite existing video {final_path}? Press ENTER to cancel.", default=False
        ):
            print(f"Video left at {temp_video_path}")
            return temp_video_path
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_video_path), str(final_path))

    # Report and return the final video path for convenient programmatic use.
    print(f"Video saved to {final_path}")
    return final_path


if __name__ == "__main__":
    make_video(frames_dir="/Users/tommieverouden/Documents/Data/Droplet atomisation/260415_needlesize_timing_test/Ga26/timing_Ga26_59.5ms_newtube_P-001_20000fps_16700 nsec",
               time_stretch_s_per_s=0.002, crop_roi=(10, 0, 0, 0),
               frames_range=(1, 100), show_preview=True, label_font_style="bold")
