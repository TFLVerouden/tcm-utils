"""Create Keynote-friendly H.264 videos from numbered TIFF frames."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterator

import cv2 as cv
import numpy as np
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


def _sequence_contrast_limits(
    numbered_paths: list[tuple[int, Path]],
    n_jobs: int | None,
    low_percentile: float,
    high_percentile: float,
) -> tuple[tuple[int, int], tuple[float, float]]:
    histogram = np.zeros(65_536, dtype=np.uint64)
    expected_shape: tuple[int, int] | None = None

    for _, path, loaded in tqdm(
        _iter_loaded_frames(numbered_paths, n_jobs),
        total=len(numbered_paths),
        desc="Analyzing TIFF frames",
        leave=False,
    ):
        image = _validate_frame(loaded, path, expected_shape)
        if expected_shape is None:
            expected_shape = image.shape
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
    assert expected_shape is not None
    return expected_shape, (low_limit, high_limit)


def auto_brightness(
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


def _format_time(frame_number: int, recording_rate: float, unit: str) -> str:
    seconds = frame_number / recording_rate
    value = seconds * _TIME_FACTORS[unit]
    return f"{value:.3f} {unit}"


def _add_time_label(image: np.ndarray, label: str) -> np.ndarray:
    height, width = image.shape
    font_scale = max(0.45, min(width, height) / 900)
    thickness = max(1, round(font_scale * 2))
    position = (16, min(height - 12, 32 + round(font_scale * 8)))
    cv.putText(
        image, label, position, cv.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness + 2,
        cv.LINE_AA,
    )
    cv.putText(
        image, label, position, cv.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness,
        cv.LINE_AA,
    )
    return image


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
) -> Path | None:
    """Encode numbered grayscale TIFF frames as an H.264 MP4.

    ``frames_range`` uses the trailing filename number and includes both ends.
    The output rate defaults to recording rate times ``time_stretch_s_per_s``;
    the default stretch of 0.002 therefore makes 20,000-fps footage play at
    40 fps, retaining every selected frame. ``output_frame_rate`` overrides
    that derived rate. Frame labels use the original filename number divided
    by the recording rate. TODO: confirm whether frame numbering is zero- or
    one-based; currently frame 1 is labeled as 1 / recording rate.
    """
    # Ask for the TIFF folder only when the caller did not supply one.
    if frames_dir is None:
        selected_dir = ask_directory(
            key="video_maker",
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

    # Select the requested TIFFs by their trailing frame numbers and determine
    # the camera's recording rate from metadata, or accept a caller-supplied rate.
    numbered_paths = _select_frame_paths(frames_dir, frames_range)
    if recording_frame_rate is None:
        recording_frame_rate = _get_recording_frame_rate(frames_dir)
    elif not math.isfinite(recording_frame_rate) or recording_frame_rate <= 0:
        raise ValueError("recording_frame_rate must be positive")

    # Derive the playback rate from the requested stretch unless the caller
    # explicitly overrides the output rate.
    stretch = 1.0 if time_stretch_s_per_s is None else time_stretch_s_per_s
    frame_rate = output_frame_rate or recording_frame_rate * stretch
    frame_rate_text = format(frame_rate, ".12g")

    # Choose the destination and create its parent folder; this function writes
    # H.264 only into an MP4 container.
    output_path = (
        Path(output_path).expanduser()
        if output_path is not None
        else find_repo_root(Path(__file__)) / ".temp" / "video.mp4"
    )
    output_path = output_path.resolve()
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("H.264 video output must use the .mp4 extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Inspect the sequence once to ensure all frames have matching dimensions
    # and to calculate one shared contrast range, preventing brightness flicker.
    expected_shape, contrast_limits = _sequence_contrast_limits(
        numbered_paths, n_jobs, 0.5, 99.95
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
    summary = (
        f"Create {len(numbered_paths)}-frame H.264 MP4 at {frame_rate:g} fps "
        f"({duration_s:.3f} s), {width}x{height}, to {output_path}?"
    )
    if confirm and not prompt_yes_no(summary, default=True):
        return None
    if output_path.exists():
        if confirm:
            if not prompt_yes_no(
                f"Overwrite existing video {output_path}?", default=False
            ):
                return None
        else:
            raise FileExistsError(
                f"Output video already exists: {output_path}")

    # Locate FFmpeg explicitly so a missing encoder produces an actionable error.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError(
            "FFmpeg is required to create H.264 MP4 files; install FFmpeg and retry."
        )

    # Encode to a temporary file beside the destination so a failed encode
    # cannot leave a partial video at the requested output path.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tmp.mp4",
        dir=output_path.parent,
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
            frame = _validate_frame(loaded, frame_path, expected_shape)
            frame = auto_brightness(frame, limits=contrast_limits)
            label = _format_time(
                frame_number, recording_frame_rate, time_label_unit)
            frame = _add_time_label(frame, label)
            process.stdin.write(np.ascontiguousarray(frame).tobytes())

        # Signal the end of raw video input, collect FFmpeg's result, and only
        # publish the temporary output when the complete encode succeeded.
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
        if return_code:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg failed to encode the video: {message}")
        os.replace(temporary_path, output_path)
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
        # failure path while leaving a successfully published video untouched.
        if process.poll() is None:
            process.kill()
            process.wait()
        if not completed:
            temporary_path.unlink(missing_ok=True)

    # Report and return the final video path for convenient programmatic use.
    print(f"Video saved to {output_path}")
    return output_path


if __name__ == "__main__":
    make_video(frames_range=(1, 100))
