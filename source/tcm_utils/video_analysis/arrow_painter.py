"""Draw flow-rate labels and velocity arrows on video frames."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import cv2 as cv
import numpy as np
from PIL import Image, ImageFont

from tcm_utils.video_maker import (
    FrameProcessor,
    _resolve_label_font_path,
    _resolve_label_color,
    _resolve_label_location,
    _resolve_label_location_offset,
    draw_label,
)


class ArrowPainter(FrameProcessor):
    """Draw flow rate and per-window velocity vectors on grayscale frames.

    Flow-label typography and placement options follow ``make_video``.
    """

    def __init__(
        self,
        *,
        flow_rate_csv_path: Path,
        velocity_csv_path: Path,
        arrow_scale_px_per_m_s: float,
        label_font_path: str | Path | None = None,
        label_font_style: str = "regular",
        label_font_size_px: int = 48,
        label_color: str | int = "white",
        label_location: str | tuple[float, float] = "lower left",
        label_location_offset: tuple[float, float] = (0.0, 0.0),
        label_stroke_color: str | int | None = None,
    ) -> None:
        if (
            isinstance(arrow_scale_px_per_m_s, bool)
            or not isinstance(arrow_scale_px_per_m_s, (int, float))
            or not math.isfinite(arrow_scale_px_per_m_s)
            or arrow_scale_px_per_m_s <= 0
        ):
            raise ValueError(
                "arrow_scale_px_per_m_s must be a positive finite number"
            )
        if (
            isinstance(label_font_size_px, bool)
            or not isinstance(label_font_size_px, int)
            or label_font_size_px <= 0
        ):
            raise ValueError("label_font_size_px must be a positive integer")

        self.flow_times_s, self.flow_rates_l_s = read_flow_rate_csv(
            flow_rate_csv_path
        )
        velocity_data = read_velocity_csv(velocity_csv_path)
        self.velocity_windows = cast(
            dict[tuple[float, float], dict[str, np.ndarray]],
            velocity_data["windows"],
        )
        self.arrow_scale_px_per_m_s = float(arrow_scale_px_per_m_s)
        font_path = _resolve_label_font_path(
            label_font_path,
            label_font_style,
        )
        self.label_font = ImageFont.truetype(
            str(font_path), size=label_font_size_px
        )
        self.label_font_size_px = label_font_size_px
        self.label_color = _resolve_label_color(label_color)
        self.label_location = _resolve_label_location(label_location)
        self.label_location_offset = _resolve_label_location_offset(
            label_location_offset,
            "label_location_offset",
        )
        self.label_stroke_color = (
            _resolve_label_color(label_stroke_color, "label_stroke_color")
            if label_stroke_color is not None
            else None
        )
        self.recording_frame_rate: float | None = None
        self.scale_m_per_px: float | None = None
        self.first_frame_number: int | None = None
        self.time_offset_s = 0.0

    def set_video_context(self, **context: object) -> None:
        recording_frame_rate = context.get("recording_frame_rate")
        if (
            isinstance(recording_frame_rate, bool)
            or not isinstance(recording_frame_rate, (int, float))
            or not math.isfinite(recording_frame_rate)
            or recording_frame_rate <= 0
        ):
            raise ValueError(
                "ArrowPainter requires a positive finite recording_frame_rate"
            )

        scale_m_per_px = context.get("scale_m_per_px")
        if (
            isinstance(scale_m_per_px, bool)
            or not isinstance(scale_m_per_px, (int, float))
            or not math.isfinite(scale_m_per_px)
            or scale_m_per_px <= 0
        ):
            raise ValueError(
                "ArrowPainter requires a positive finite scale_m_per_px"
            )

        first_frame_number = context.get("first_frame_number")
        if (
            isinstance(first_frame_number, bool)
            or not isinstance(first_frame_number, int)
        ):
            raise ValueError(
                "ArrowPainter requires an integer first_frame_number"
            )

        time_offset_s = context.get("time_offset_s")
        if time_offset_s is None:
            time_offset_s = 0.0
        elif (
            isinstance(time_offset_s, bool)
            or not isinstance(time_offset_s, (int, float))
            or not math.isfinite(time_offset_s)
        ):
            raise ValueError(
                "ArrowPainter requires a finite time_offset_s or None"
            )

        self.recording_frame_rate = float(recording_frame_rate)
        self.scale_m_per_px = float(scale_m_per_px)
        self.first_frame_number = first_frame_number
        self.time_offset_s = float(time_offset_s)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        *,
        is_preview: bool,
    ) -> np.ndarray:
        del is_preview
        if frame.ndim != 2 or frame.dtype != np.uint8:
            raise ValueError("ArrowPainter requires an 8-bit grayscale frame")
        if (
            self.recording_frame_rate is None
            or self.scale_m_per_px is None
            or self.first_frame_number is None
        ):
            raise RuntimeError(
                "ArrowPainter requires set_video_context before processing"
            )

        time_s = (
            (frame_number - self.first_frame_number)
            / self.recording_frame_rate
            + self.time_offset_s
        )
        flow_index = _nearest_time_index(self.flow_times_s, time_s)
        if flow_index is not None:
            flow_rate = self.flow_rates_l_s[flow_index]
            flow_rate_text = (
                f"{flow_rate:.3g}" if math.isfinite(flow_rate) else "-"
            )
            flow_label = f"{flow_rate_text} L/s"

        pixels_per_mm = 0.001 / self.scale_m_per_px
        for window in self.velocity_windows.values():
            sample_index = _nearest_time_index(window["time_s"], time_s)
            if sample_index is None:
                continue

            velocity_x = window["vx_m_s"][sample_index]
            velocity_y = window["vy_m_s"][sample_index]
            if not (math.isfinite(velocity_x) and math.isfinite(velocity_y)):
                continue

            start_x = round(window["win_x_mm"][sample_index] * pixels_per_mm)
            start_y = round(
                (frame.shape[0] - 1) / 2
                - window["win_y_mm"][sample_index] * pixels_per_mm
            )
            if not (0 <= start_x < frame.shape[1] and 0 <= start_y < frame.shape[0]):
                continue

            end_x = start_x + round(
                velocity_x * self.arrow_scale_px_per_m_s
            )
            end_y = start_y - round(
                velocity_y * self.arrow_scale_px_per_m_s
            )
            if (end_x, end_y) == (start_x, start_y):
                continue

            cv.arrowedLine(
                frame,
                (start_x, start_y),
                (end_x, end_y),
                color=255,
                thickness=1,
                line_type=cv.LINE_AA,
                tipLength=0.2,
            )

        if flow_index is not None:
            image = Image.fromarray(frame)
            draw_label(
                image,
                flow_label,
                font=self.label_font,
                color=self.label_color,
                stroke_color=self.label_stroke_color,
                location=self.label_location,
                location_offset=self.label_location_offset,
                font_size_px=self.label_font_size_px,
            )
            frame = np.asarray(image)
        return frame

    def finish(self, video_path: Path) -> None:
        return None


def _nearest_time_index(times_s: np.ndarray, target_time_s: float) -> int | None:
    if target_time_s < times_s[0] or target_time_s > times_s[-1]:
        return None
    right_index = int(np.searchsorted(times_s, target_time_s, side="left"))
    if right_index == 0:
        return 0
    if right_index == len(times_s):
        return len(times_s) - 1
    left_index = right_index - 1
    if target_time_s - times_s[left_index] <= times_s[right_index] - target_time_s:
        return left_index
    return right_index


def read_flow_rate_csv(flow_rate_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(flow_rate_path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data found in {flow_rate_path}")
    column_names = data.dtype.names or ()
    required_columns = ("time_s", "flow_rate_L_s")
    missing_columns = [
        name for name in required_columns if name not in column_names
    ]
    if missing_columns:
        raise ValueError(
            f"flow-rate CSV is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    data = np.atleast_1d(data)
    times_s = np.asarray(data["time_s"], dtype=np.float64)
    flow_rates_l_s = np.asarray(data["flow_rate_L_s"], dtype=np.float64)
    if not np.all(np.isfinite(times_s)):
        raise ValueError(
            "flow-rate CSV column time_s must contain finite values")

    sort_order = np.argsort(times_s, kind="stable")
    return times_s[sort_order], flow_rates_l_s[sort_order]


def read_velocity_csv(velocity_path: Path) -> dict[str, object]:
    """Read a stitched velocity CSV into columns and per-window time series.

    The ``windows`` entry is keyed by ``(win_x, win_y)`` and contains one
    dictionary of arrays per PIV window. Each window's rows are sorted by
    ``time_s`` so its velocity components can be plotted directly against
    time.
    """
    data = np.genfromtxt(
        velocity_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    if data.size == 0:
        raise RuntimeError(f"No data found in {velocity_path}")
    data = np.atleast_1d(data)
    column_names = data.dtype.names or ()

    required_columns = (
        "source_subfolder",
        "source_pair_index",
        "stitched_pair_index",
        "time_s",
        "win_y",
        "win_y_mm",
        "win_x",
        "win_x_mm",
        "vy_m_s",
        "vx_m_s",
    )
    missing_columns = [
        name for name in required_columns if name not in column_names
    ]
    if missing_columns:
        raise ValueError(
            f"velocity.csv is missing required columns: {', '.join(missing_columns)}"
        )

    columns: dict[str, np.ndarray] = {
        name: np.asarray(data[name]).reshape(-1) for name in required_columns
    }
    numeric_columns = required_columns[1:]
    for name in numeric_columns:
        values = np.asarray(columns[name], dtype=np.float64)
        if name not in ("vx_m_s", "vy_m_s") and not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                f"velocity CSV column {name} must contain finite values"
            )
        columns[name] = values

    windows: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    window_coordinates = np.unique(
        np.column_stack((columns["win_x"], columns["win_y"])), axis=0
    )
    for win_x, win_y in window_coordinates:
        window_mask = (
            (columns["win_x"] == win_x) & (columns["win_y"] == win_y)
        )
        window_indices = np.flatnonzero(window_mask)
        sort_order = np.argsort(
            columns["time_s"][window_indices], kind="stable"
        )
        window_indices = window_indices[sort_order]
        key = (float(win_x), float(win_y))
        windows[key] = {
            name: values[window_indices] for name, values in columns.items()
        }

    return {**columns, "windows": windows}
