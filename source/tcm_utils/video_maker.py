"""Create Keynote-friendly H.264 videos from numbered TIFF frames."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from decimal import Decimal
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterator, Protocol, runtime_checkable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from tcm_utils.camera_calibration import ensure_calibration
from tcm_utils.file_dialogs import ask_directory, find_repo_root
from tcm_utils.io_utils import (
    auto_brightness,
    load_image,
    load_metadata,
    prompt_input,
    prompt_yes_no,
)
from tcm_utils.read_cihx import extract_cihx_metadata, recursive_search

_FRAME_NUMBER = re.compile(r"(\d{6})\.(?:tif|tiff)$", re.IGNORECASE)
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
_LENGTH_FACTORS_M = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "um": 0.000001,
    "µm": 0.000001,
    "nm": 0.000000001,
}


class FrameProcessor(Protocol):
    """Transform frames and save processor results alongside the output video."""

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        *,
        is_preview: bool,
    ) -> np.ndarray:
        """Process a frame and return its same-sized grayscale image."""
        ...

    def finish(self, video_path: Path) -> None:
        """Persist any processor output beside the completed video."""
        ...


@runtime_checkable
class VideoContextProcessor(Protocol):
    """Optional capability for processors that need video metadata."""

    def set_video_context(
        self,
        *,
        recording_frame_rate: float,
        scale_m_per_px: float | None,
    ) -> None:
        """Provide metadata resolved by ``make_video`` before processing."""
        ...


def _process_frame(
    frame: np.ndarray,
    frame_number: int,
    processor: FrameProcessor | None,
    *,
    is_preview: bool,
) -> np.ndarray:
    if processor is None:
        return frame

    processed = processor.process_frame(
        frame,
        frame_number,
        is_preview=is_preview,
    )
    if not isinstance(processed, np.ndarray):
        raise TypeError("frame processor must return a NumPy array")
    if (
        processed.ndim != 2
        or processed.dtype != np.uint8
        or processed.shape != frame.shape
    ):
        raise ValueError(
            "frame processor must return a same-sized 8-bit grayscale image"
        )
    return processed


def _finish_frame_processor(
    processor: FrameProcessor | None,
    video_path: Path,
) -> Path:
    if processor is not None:
        processor.finish(video_path)
    return video_path


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
    frames_dirs: Sequence[Path],
    frames_range: tuple[int, int] | None,
    *,
    require_contiguous: bool = False,
) -> tuple[list[tuple[int, Path]], int]:
    for frames_dir in frames_dirs:
        if not frames_dir.is_dir():
            raise NotADirectoryError(
                f"Frames directory does not exist: {frames_dir}")

    all_paths = [
        path
        for frames_dir in frames_dirs
        for path in frames_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
        and not path.name.startswith(".")
    ]
    if not all_paths:
        raise FileNotFoundError(
            f"No TIFF frame files found in {', '.join(map(str, frames_dirs))}"
        )

    numbered_paths = [(_frame_number(path), path) for path in all_paths]
    numbered_paths.sort(key=lambda item: (item[0], item[1].name))
    frame_numbers = [number for number, _ in numbered_paths]
    if len(frame_numbers) != len(set(frame_numbers)):
        raise ValueError(
            "Multiple TIFF files have the same trailing frame number across "
            f"the supplied directories: {', '.join(map(str, frames_dirs))}"
        )
    if require_contiguous:
        for previous_number, number in zip(frame_numbers, frame_numbers[1:]):
            if number != previous_number + 1:
                raise ValueError(
                    "Frame numbers across multiple frames_dir paths must be "
                    f"continuous; found {previous_number} followed by {number}"
                )
    first_frame_number = numbered_paths[0][0]
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

    return numbered_paths, first_frame_number


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


def _get_recording_frame_rate(frames_dirs: Sequence[Path]) -> float:
    def find_frame_rate(metadata: object) -> float | None:
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if str(key).lower() in _FRAME_RATE_KEYS:
                    rate = _numeric_rate(value)
                    if rate is not None:
                        return rate
            for value in metadata.values():
                rate = find_frame_rate(value)
                if rate is not None:
                    return rate
        elif isinstance(metadata, list):
            for value in metadata:
                rate = find_frame_rate(value)
                if rate is not None:
                    return rate
        return None

    metadata_paths = sorted(
        (
            path
            for frames_dir in frames_dirs
            for path in frames_dir.glob("*.json")
            if "metadata" in path.name.lower() or "camera" in path.name.lower()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for metadata_path in metadata_paths:
        rate = find_frame_rate(load_metadata(metadata_path))
        if rate is not None:
            return rate

    cihx_paths = sorted(
        (
            path
            for frames_dir in frames_dirs
            for pattern in ("*.cihx", "*.cih")
            for path in frames_dir.glob(pattern)
        ),
        key=lambda path: path.name.lower(),
    )
    for cihx_path in cihx_paths:
        metadata = extract_cihx_metadata(
            cihx_path,
            output_folder=cihx_path.parent,
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


def _flip_frame(
    image: np.ndarray,
    flip: str | None,
) -> np.ndarray:
    if flip is None:
        return image
    if flip == "vertical":
        return np.flip(image, axis=0)
    if flip == "horizontal":
        return np.flip(image, axis=1)
    if flip == "both":
        return np.flip(image, axis=(0, 1))
    raise ValueError("flip must be 'vertical', 'horizontal', 'both', or None")


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


def _resolve_label_location_offset(
    offset: tuple[float, float],
    parameter_name: str,
) -> tuple[float, float]:
    """Validate a pixel offset in (dy, dx) order."""
    if (
        not isinstance(offset, tuple)
        or len(offset) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in offset
        )
    ):
        raise ValueError(f"{parameter_name} must be a finite (dy, dx) pair")
    return float(offset[0]), float(offset[1])


def _add_frame_labels(
    image: np.ndarray,
    time_label: str,
    *,
    font: ImageFont.FreeTypeFont,
    color: int,
    stroke_color: int | None,
    time_location: tuple[float, float],
    time_location_offset: tuple[float, float],
    font_size_px: int,
    scale_bar: tuple[str, int, tuple[float, float]] | None,
    scale_bar_location_offset: tuple[float, float],
    scale_bar_height_px: int,
    stacked_label_margin: int,
    label_stacking_mode: str,
) -> np.ndarray:
    """Draw the timestamp and optional scale bar in grayscale."""
    frame = Image.fromarray(image).convert("L")
    drawing = ImageDraw.Draw(frame)
    stroke_width = (
        max(1, round(font_size_px / 12))
        if stroke_color is not None
        else 0
    )
    margin = max(8, round(font_size_px / 2))

    def label_dimensions(
        label: str,
        bar_length_px: int | None = None,
    ) -> tuple[tuple[float, float, float, float], int, int, int, int]:
        bounds = drawing.textbbox(
            (0, 0), label, font=font, stroke_width=stroke_width
        )
        text_width = math.ceil(bounds[2] - bounds[0])
        text_height = math.ceil(bounds[3] - bounds[1])
        gap = max(4, round(font_size_px / 4)) if bar_length_px else 0
        group_width = max(text_width, bar_length_px or 0)
        group_height = text_height + gap + (
            scale_bar_height_px if bar_length_px else 0
        )
        max_left = frame.width - group_width - margin
        max_top = frame.height - group_height - margin
        if max_left < margin or max_top < margin:
            overlay = "Scale bar" if bar_length_px else "Time label"
            raise ValueError(
                f"{overlay} at {font_size_px}px does not fit in "
                f"{frame.width}x{frame.height} frame"
            )
        return bounds, text_width, text_height, group_width, group_height

    def position(
        location: tuple[float, float],
        group_width: int,
        group_height: int,
    ) -> tuple[int, int]:
        max_left = frame.width - group_width - margin
        max_top = frame.height - group_height - margin
        return (
            round(margin + (max_left - margin) * location[0]),
            round(margin + (max_top - margin) * location[1]),
        )

    time_bounds, time_width, _, time_group_width, time_group_height = (
        label_dimensions(time_label)
    )
    time_left, time_top = position(
        time_location,
        time_group_width,
        time_group_height,
    )

    scale_geometry: tuple[
        str,
        int,
        tuple[float, float],
        tuple[float, float, float, float],
        int,
        int,
        int,
        int,
    ] | None = None
    if scale_bar is not None:
        scale_label, length_px, location = scale_bar
        (
            scale_bounds,
            scale_text_width,
            scale_text_height,
            scale_group_width,
            scale_group_height,
        ) = label_dimensions(scale_label, length_px)
        scale_left, scale_top = position(
            location,
            scale_group_width,
            scale_group_height,
        )
        scale_geometry = (
            scale_label,
            length_px,
            location,
            scale_bounds,
            scale_text_width,
            scale_text_height,
            scale_group_width,
            scale_group_height,
        )

        if time_location == location:
            if label_stacking_mode.startswith("vertical"):
                stack_width = max(time_group_width, scale_group_width)
                stack_height = (
                    time_group_height
                    + stacked_label_margin
                    + scale_group_height
                )
            else:
                stack_width = (
                    time_group_width
                    + stacked_label_margin
                    + scale_group_width
                )
                stack_height = max(time_group_height, scale_group_height)

            max_left = frame.width - stack_width - margin
            max_top = frame.height - stack_height - margin
            if max_left < margin or max_top < margin:
                raise ValueError(
                    f"Labels in {label_stacking_mode!r} mode with a "
                    f"{stacked_label_margin}px margin do not fit in "
                    f"{frame.width}x{frame.height} frame"
                )
            stack_left = round(
                margin + (max_left - margin) * time_location[0]
            )
            stack_top = round(
                margin + (max_top - margin) * time_location[1]
            )

            def alignment_offset(
                available: int,
                content: int,
                anchor: float,
            ) -> int:
                if anchor <= 0:
                    return 0
                if anchor >= 1:
                    return available - content
                return (available - content) // 2

            if label_stacking_mode == "vertical":
                time_left = stack_left + alignment_offset(
                    stack_width, time_group_width, time_location[0]
                )
                scale_left = stack_left + alignment_offset(
                    stack_width, scale_group_width, time_location[0]
                )
                time_top = stack_top
                scale_top = stack_top + time_group_height + stacked_label_margin
            elif label_stacking_mode == "vertical reversed":
                scale_left = stack_left + alignment_offset(
                    stack_width, scale_group_width, time_location[0]
                )
                time_left = stack_left + alignment_offset(
                    stack_width, time_group_width, time_location[0]
                )
                scale_top = stack_top
                time_top = stack_top + scale_group_height + stacked_label_margin
            elif label_stacking_mode == "horizontal":
                time_left = stack_left
                scale_left = stack_left + time_group_width + stacked_label_margin
                time_top = stack_top + alignment_offset(
                    stack_height, time_group_height, time_location[1]
                )
                scale_top = stack_top + alignment_offset(
                    stack_height, scale_group_height, time_location[1]
                )
            else:
                scale_left = stack_left
                time_left = stack_left + scale_group_width + stacked_label_margin
                scale_top = stack_top + alignment_offset(
                    stack_height, scale_group_height, time_location[1]
                )
                time_top = stack_top + alignment_offset(
                    stack_height, time_group_height, time_location[1]
                )

    time_left = round(time_left + time_location_offset[1])
    time_top = round(time_top + time_location_offset[0])
    if scale_geometry is not None:
        scale_left = round(scale_left + scale_bar_location_offset[1])
        scale_top = round(scale_top + scale_bar_location_offset[0])

    def draw_label(
        label: str,
        bounds: tuple[float, float, float, float],
        text_width: int,
        left: int,
        top: int,
        bar_length_px: int | None = None,
    ) -> None:
        group_width = max(text_width, bar_length_px or 0)
        gap = max(4, round(font_size_px / 4)) if bar_length_px else 0
        if bar_length_px:
            bar_left = left + (group_width - bar_length_px) // 2
            drawing.rectangle(
                (bar_left, top, bar_left + bar_length_px - 1,
                 top + scale_bar_height_px - 1),
                fill=color,
            )
            text_top = top + scale_bar_height_px + gap
            text_left = bar_left + (bar_length_px - text_width) // 2
        else:
            text_left, text_top = left, top
        drawing.text(
            (text_left - bounds[0], text_top - bounds[1]),
            label,
            font=font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )

    draw_label(
        time_label,
        time_bounds,
        time_width,
        time_left,
        time_top,
    )
    if scale_geometry is not None:
        (
            scale_label,
            length_px,
            _,
            scale_bounds,
            scale_text_width,
            _,
            _,
            _,
        ) = scale_geometry
        draw_label(
            scale_label,
            scale_bounds,
            scale_text_width,
            scale_left,
            scale_top,
            length_px,
        )
    return np.asarray(frame)


def _show_frame_preview(
    image: np.ndarray,
    frame_path: Path,
    label: str,
    *,
    frame_number: int,
    processor: FrameProcessor | None,
    font: ImageFont.FreeTypeFont,
    color: int,
    stroke_color: int | None,
    location: tuple[float, float],
    location_offset: tuple[float, float],
    font_size_px: int,
    scale_bar: tuple[str, int, tuple[float, float]] | None,
    scale_bar_location_offset: tuple[float, float],
    scale_bar_height_px: int,
    stacked_label_margin: int,
    label_stacking_mode: str,
) -> bool:
    """Display one prepared frame so the user can review its appearance."""
    # Build the preview with the same overlay and contrast operation as output.
    preview_frame = _process_frame(
        auto_brightness(image, percentile_stretch=True),
        frame_number,
        processor,
        is_preview=True,
    )
    preview = _add_frame_labels(
        preview_frame,
        label,
        font=font,
        color=color,
        stroke_color=stroke_color,
        time_location=location,
        time_location_offset=location_offset,
        font_size_px=font_size_px,
        scale_bar=scale_bar,
        scale_bar_location_offset=scale_bar_location_offset,
        scale_bar_height_px=scale_bar_height_px,
        stacked_label_margin=stacked_label_margin,
        label_stacking_mode=label_stacking_mode,
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
        try:
            canvas.get_tk_widget().destroy()
            # FigureCanvasTkAgg has no public close method for its Tk image.
            del canvas._tkphoto
        finally:
            root.destroy()


def make_video(
    frames_dir: str | Path | Sequence[str | Path] | None = None,
    frames_range: tuple[int, int] | None = None,
    recording_frame_rate: float | None = None,
    time_stretch_s_per_s: float | None = 0.002,
    output_frame_rate: float | None = None,
    time_label_unit: str = "ms",
    crop_roi: tuple[int, int, int, int] | None = None,
    flip: str | None = None,
    n_jobs: int | None = None,
    label_font_path: str | Path | None = None,
    label_font_style: str = "regular",
    label_font_size_px: int = 48,
    label_color: str | int = "black",
    label_location: str | tuple[float, float] = "upper left",
    label_location_offset: tuple[float, float] = (0.0, 0.0),
    label_stroke_color: str | int | None = None,
    show_scale_bar: bool = False,
    scale_bar_calibration_path: str | Path | None = None,
    scale_bar_length: float = 5.0,
    scale_bar_unit: str = "mm",
    scale_bar_location: str | tuple[float, float] = "lower right",
    scale_bar_location_offset: tuple[float, float] = (0.0, 0.0),
    scale_bar_height_px: int = 8,
    stacked_label_mode: str = "vertical",
    stacked_label_margin: int = 20,
    output_path: str | Path | None = None,
    output_filename: str | Path | None = None,
    confirm: bool = True,
    show_preview: bool = True,
    processor: FrameProcessor | None = None,
) -> Path | None:
    """Encode numbered grayscale TIFF frames as an H.264 MP4.

    Args:
        frames_dir: Directory containing numbered TIFF frames, or an ordered
            sequence of directories whose numbered frames form one continuous
            sequence. If omitted, a folder picker is shown.
        frames_range: Optional inclusive (start, end) range using the trailing
            frame number in each filename.
        recording_frame_rate: Camera recording rate in frames per second. If
            omitted, it is read from metadata or requested interactively.
        time_stretch_s_per_s: Playback seconds per second of recorded time. The
            default 0.002 makes 20,000-fps footage play at 40 fps while
            retaining every selected frame. ``None`` means real-time playback.
        output_frame_rate: Explicit playback rate, overriding the rate derived
            from the recording rate and time stretch.
        time_label_unit: Unit for timestamps: ``"s"``, ``"ms"``, or ``"us"``.
            Labels are elapsed time from the first numbered TIFF in the folder,
            so that frame displays 0 and selected later frames retain their
            original offsets.
        crop_roi: Optional crop applied before preview and encoding, given as
            ``(y_start, y_end, x_start, x_end)``. Negative coordinates are
            offsets from the corresponding image edge; zero end coordinates
            mean the full extent in that direction.
        flip: Optional frame flip: ``"vertical"``, ``"horizontal"``, or
            ``"both"``. Flips are applied after cropping and before labels.
        n_jobs: Number of TIFF-loading workers, or ``None`` to choose a
            default.
        label_font_path: Optional TrueType/OpenType font file. By default, the
            built-in PT Sans font matching ``label_font_style`` is used.
        label_font_style: Built-in PT Sans style: ``"regular"``, ``"bold"``,
            ``"italic"``, or ``"bold italic"``. Ignored when
            ``label_font_path`` is provided.
        label_font_size_px: Timestamp and scale-bar text size in pixels.
        label_color: Text and scale-bar color: black, white, gray/grey, or an
            integer grayscale value from 0 to 255.
        label_location: Timestamp position, as a common legend-style location
            or a normalized (x, y) pair.
        label_location_offset: Timestamp offset in pixels as a ``(dy, dx)``
            pair, applied after any automatic label stacking. Positive values
            move down and right; offsets may move the label off-screen.
        stacked_label_mode: Arrangement used when the timestamp and scale bar
            share a location: ``"vertical"`` (timestamp above), ``"horizontal"``
            (timestamp left), ``"horizontal reversed"`` (scale bar left), or
            ``"vertical reversed"`` (scale bar above).
        label_stroke_color: Text outline color, accepting the same values as
            ``label_color``. ``None`` disables the outline.
        show_scale_bar: Whether to draw a calibrated scale bar on every frame
            and in the preview. Supplying ``scale_bar_calibration_path`` also
            enables the scale bar.
        scale_bar_calibration_path: Calibration metadata path. If omitted when
            a scale bar is enabled, a picker can select metadata or an image to
            calibrate.
        scale_bar_length: Physical scale-bar length.
        scale_bar_unit: Scale-bar length unit, such as ``"mm"`` or ``"um"``.
        scale_bar_location: Scale-bar position, as a common legend-style
            location or a normalized (x, y) pair.
        scale_bar_location_offset: Scale-bar offset in pixels as a
            ``(dy, dx)`` pair, applied after any automatic label stacking.
            Positive values move down and right; offsets may move the scale
            bar off-screen.
        stacked_label_margin: Gap in pixels between labels automatically
            arranged at the same location.
        scale_bar_height_px: Scale-bar thickness in pixels.
        output_path: Destination MP4 file or output directory. The video is
            encoded into ``<repo>/.temp`` first and then moved to this
            destination. If omitted, a folder picker appears after encoding;
            cancelling leaves the video in ``.temp``.
        output_filename: MP4 filename used when ``output_path`` is a directory
            or omitted. Defaults to the first selected TIFF's name without its
            trailing frame number. Do not combine this with an MP4
            ``output_path``.
        confirm: Whether to show the pre-encode confirmation and ask before
            overwriting an existing destination.
        show_preview: Whether to show the first selected frame with its
            timestamp and a per-frame contrast stretch before sequence-wide
            contrast analysis. The preview is shown only when ``confirm`` is
            true.
        processor: Optional object with a ``process_frame(frame,
            frame_number, *, is_preview)`` method. It receives each
            brightness-normalized 8-bit grayscale frame before labels are
            drawn and must return a same-sized 8-bit grayscale NumPy array.
            The preview is marked with ``is_preview=True``; encoded frames use
            ``False``. A processor is called serially and must implement
            ``finish(video_path)`` to save any results after the video is
            delivered. A processor may also implement
            ``set_video_context(*, recording_frame_rate, scale_m_per_px)`` to
            receive the resolved recording rate and calibration scale before
            preview or frame processing. The scale is ``None`` when no
            calibration was loaded. ``video_path`` is the path returned by
            this function.
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
    if isinstance(frames_dir, (str, Path)):
        frames_dirs = [Path(frames_dir).expanduser().resolve()]
    else:
        frames_dirs = [
            Path(path).expanduser().resolve() for path in frames_dir
        ]
        if not frames_dirs:
            raise ValueError("frames_dir must contain at least one directory")
    # Use the first folder as the default for later destination pickers.
    frames_dir = frames_dirs[0]
    if flip not in (None, "vertical", "horizontal", "both"):
        raise ValueError(
            "flip must be 'vertical', 'horizontal', 'both', or None")
    label_offset = _resolve_label_location_offset(
        label_location_offset,
        "label_location_offset",
    )
    scale_bar_offset = _resolve_label_location_offset(
        scale_bar_location_offset,
        "scale_bar_location_offset",
    )
    if (
        isinstance(stacked_label_margin, bool)
        or not isinstance(stacked_label_margin, int)
        or stacked_label_margin < 0
    ):
        raise ValueError("stacked_label_margin must be a non-negative integer")
    if stacked_label_mode not in (
        "horizontal",
        "vertical",
        "horizontal reversed",
        "vertical reversed",
    ):
        raise ValueError(
            "label_stacking_mode must be 'horizontal', 'vertical', "
            "'horizontal reversed', or 'vertical reversed'"
        )
    scale_bar_enabled = (
        show_scale_bar or scale_bar_calibration_path is not None
    )

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
    if scale_bar_enabled:
        if (
            isinstance(scale_bar_length, bool)
            or not isinstance(scale_bar_length, (int, float))
            or not math.isfinite(scale_bar_length)
            or scale_bar_length <= 0
        ):
            raise ValueError(
                "scale_bar_length must be a positive finite number")
        if scale_bar_unit not in _LENGTH_FACTORS_M:
            raise ValueError(
                f"scale_bar_unit must be one of {tuple(_LENGTH_FACTORS_M)}"
            )
        if (
            isinstance(scale_bar_height_px, bool)
            or not isinstance(scale_bar_height_px, int)
            or scale_bar_height_px <= 0
        ):
            raise ValueError("scale_bar_height_px must be a positive integer")

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
    scale_bar_config: tuple[str, int, tuple[float, float]] | None = None
    scale_m_per_px: float | None = None
    if scale_bar_enabled:
        calibration_path = ensure_calibration(
            input_path=(
                Path(scale_bar_calibration_path).expanduser()
                if scale_bar_calibration_path is not None
                else None
            )
        )
        if calibration_path is None:
            print("No calibration selected. Exiting without creating a video.")
            return None
        calibration_metadata = load_metadata(calibration_path)
        calibration = (
            calibration_metadata.get("calibration")
            if isinstance(calibration_metadata, dict)
            else None
        )
        scale_m_per_px = (
            calibration.get("scale_m_per_px")
            if isinstance(calibration, dict)
            else None
        )
        if (
            isinstance(scale_m_per_px, bool)
            or not isinstance(scale_m_per_px, (int, float))
            or not math.isfinite(scale_m_per_px)
            or scale_m_per_px <= 0
        ):
            raise ValueError(
                f"Calibration metadata at {calibration_path} must contain "
                "a positive finite calibration.scale_m_per_px"
            )
        length_m = scale_bar_length * _LENGTH_FACTORS_M[scale_bar_unit]
        scale_length_px = round(length_m / scale_m_per_px)
        if scale_length_px < 1:
            raise ValueError(
                "The requested scale bar is shorter than one image pixel"
            )
        scale_bar_config = (
            f"{scale_bar_length:g} {scale_bar_unit}",
            scale_length_px,
            _resolve_label_location(scale_bar_location),
        )

    # Select the requested TIFFs by their trailing frame numbers and determine
    # the camera's recording rate from metadata, or accept a caller-supplied rate.
    # Use the earliest numbered TIFF in the folder as the time origin so a
    # selection starting later keeps its offset from the recording's first frame.
    numbered_paths, first_frame_number = _select_frame_paths(
        frames_dirs,
        frames_range,
        require_contiguous=len(frames_dirs) > 1,
    )
    if recording_frame_rate is None:
        recording_frame_rate = _get_recording_frame_rate(frames_dirs)
    elif not math.isfinite(recording_frame_rate) or recording_frame_rate <= 0:
        raise ValueError("recording_frame_rate must be positive")

    # Derive the playback rate from the requested stretch unless the caller
    # explicitly overrides the output rate.
    stretch = 1.0 if time_stretch_s_per_s is None else time_stretch_s_per_s
    frame_rate = output_frame_rate or recording_frame_rate * stretch
    frame_rate_text = format(frame_rate, ".12g")

    if isinstance(processor, VideoContextProcessor):
        processor.set_video_context(
            recording_frame_rate=recording_frame_rate,
            scale_m_per_px=scale_m_per_px,
        )

    # The video is always encoded into the repo's .temp folder first. The final
    # destination is the caller's output_path, or (when None) a folder the user
    # is asked to pick once encoding has finished. This function writes H.264
    # only into an MP4 container.
    temp_dir = find_repo_root(Path(__file__)) / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    first_image_stem = numbered_paths[0][1].stem
    default_filename_stem = re.sub(r"[\s_-]*\d+$", "", first_image_stem)
    if not default_filename_stem:
        default_filename_stem = first_image_stem
    if output_filename is None:
        video_filename = f"{default_filename_stem}.mp4"
    else:
        video_filename = str(output_filename)
        if not video_filename or Path(video_filename).name != video_filename:
            raise ValueError("output_filename must be a filename, not a path")
        if not video_filename.lower().endswith(".mp4"):
            video_filename += ".mp4"

    final_path: Path | None = None
    if output_path is not None:
        requested_path = Path(output_path).expanduser().resolve()
        if requested_path.is_dir():
            final_path = requested_path / video_filename
        elif requested_path.exists() or requested_path.suffix.lower() == ".mp4":
            if output_filename is not None and requested_path.suffix.lower() == ".mp4":
                raise ValueError(
                    "Specify either output_filename or an MP4 output_path, not both"
                )
            if requested_path.suffix.lower() != ".mp4":
                raise ValueError(
                    "output_path must be an MP4 file or an output directory"
                )
            final_path = requested_path
        elif output_filename is not None:
            final_path = requested_path / video_filename
        else:
            raise ValueError("H.264 video output must use the .mp4 extension")
    temp_video_path = temp_dir / (
        final_path.name if final_path else video_filename
    )

    # Preview a single selected TIFF before loading the whole sequence for its
    # shared contrast range; this gives quick visual feedback on the edits.
    if confirm and show_preview:
        preview_number, preview_path = numbered_paths[0]
        preview_image = _validate_frame(load_image(preview_path), preview_path)
        preview_image = _flip_frame(preview_image, flip)
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
            frame_number=preview_number,
            processor=processor,
            font=label_font,
            color=label_gray,
            stroke_color=stroke_gray,
            location=label_anchor,
            location_offset=label_offset,
            font_size_px=label_font_size_px,
            scale_bar=scale_bar_config,
            scale_bar_location_offset=scale_bar_offset,
            scale_bar_height_px=scale_bar_height_px,
            stacked_label_margin=stacked_label_margin,
            label_stacking_mode=stacked_label_mode,
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
    if flip is not None:
        video_description += f" with {flip} flip"
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
            frame = _flip_frame(frame, flip)
            frame = _crop_frame(frame, crop_roi)
            frame = _validate_frame(frame, frame_path, expected_shape)
            frame = auto_brightness(
                frame,
                percentile_stretch=True,
                limits=contrast_limits,
            )
            frame = _process_frame(
                frame,
                frame_number,
                processor,
                is_preview=False,
            )
            label = _format_time(
                frame_number,
                recording_frame_rate,
                time_label_unit,
                first_frame_number,
            )
            frame = _add_frame_labels(
                frame,
                label,
                font=label_font,
                color=label_gray,
                stroke_color=stroke_gray,
                time_location=label_anchor,
                time_location_offset=label_offset,
                font_size_px=label_font_size_px,
                scale_bar=scale_bar_config,
                scale_bar_location_offset=scale_bar_offset,
                scale_bar_height_px=scale_bar_height_px,
                stacked_label_margin=stacked_label_margin,
                label_stacking_mode=stacked_label_mode,
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
            return _finish_frame_processor(processor, temp_video_path)
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
            return _finish_frame_processor(processor, temp_video_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_video_path), str(final_path))

    # Report and return the final video path for convenient programmatic use.
    print(f"Video saved to {final_path}")
    return _finish_frame_processor(processor, final_path)


if __name__ == "__main__":
    # from tcm_utils.video_analysis import EllipseSizer, process_ellipse_data

    # frames_dir = Path(
    #     "/Users/tommieverouden/Documents/Data/Droplet atomisation/260415_needlesize_timing_test/Ga26/timing_Ga26_59.5ms_newtube_P-001_20000fps_16700 nsec"
    # )
    # calibration_path = Path(
    #     "/Users/tommieverouden/Documents/Data/Droplet atomisation/260415_needlesize_timing_test/calibration/calibration_1mmspacing_afternewtubing_C001H001S0002_260507_132953_metadata.json"
    # )
    # ellipse_sizer = EllipseSizer(
    #     polarity="dark",
    #     roi=(20, 350, 560, 680),
    #     outline_color=255,
    #     outline_thickness=2,
    #     frame_range=(0, 228),
    # )
    # video_path = make_video(
    #     frames_dir=frames_dir,
    #     scale_bar_calibration_path=calibration_path,
    #     time_stretch_s_per_s=0.002,
    #     flip="horizontal",
    #     crop_roi=(10, 0, 0, 0),
    #     label_location="lower left",
    #     scale_bar_location="lower left",
    #     stacked_label_mode="vertical reversed",
    #     stacked_label_margin=48,
    #     scale_bar_length=5,
    #     scale_bar_unit="mm",
    #     label_location_offset=(0, 0),
    #     show_preview=True,
    #     processor=ellipse_sizer,
    # )
    # if video_path is not None and ellipse_sizer.csv_path is not None:
    #     process_ellipse_data(ellipse_sizer.csv_path)

    frames_dir = [
        Path("/Users/tommieverouden/Documents/Data/PIV/260820_piv/260828_163535_droplet_atomisation_campaign_reference/P-002a"),
        Path("/Users/tommieverouden/Documents/Data/PIV/260820_piv/260828_163535_droplet_atomisation_campaign_reference/P-002b"),
        Path("/Users/tommieverouden/Documents/Data/PIV/260820_piv/260828_163535_droplet_atomisation_campaign_reference/P-002c"),
        Path("/Users/tommieverouden/Documents/Data/PIV/260820_piv/260828_163535_droplet_atomisation_campaign_reference/P-002d"),
        Path("/Users/tommieverouden/Documents/Data/PIV/260820_piv/260828_163535_droplet_atomisation_campaign_reference/P-002e"),
    ]

    calibration_path = Path(
        "/Users/tommieverouden/Documents/Data/PIV/260820_piv/calibration/processed/calibration_500um_1000001_260828_164450_metadata.json")

    make_video(frames_dir=frames_dir,
               frames_range=(0, 100),
               scale_bar_calibration_path=calibration_path,
               time_stretch_s_per_s=0.05,)
