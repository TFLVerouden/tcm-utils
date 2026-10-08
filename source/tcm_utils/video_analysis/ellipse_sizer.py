"""Fit and annotate a droplet ellipse in each processed video frame."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Literal

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from tcm_utils.scientific_cmaps import get_color
from tcm_utils.video_maker import FrameProcessor
from tcm_utils.plot_style import use_tcm_poster_style, append_unit_to_last_ticklabel, set_grid
from tcm_utils.io_utils import pdf_to_png

Roi = tuple[int, int, int, int]


@dataclass(frozen=True)
class EllipseMeasurement:
    frame_number: int
    detected: bool
    time_ms: float
    center_x_px: float | None = None
    center_y_px: float | None = None
    center_x_m: float | None = None
    center_y_m: float | None = None
    radius_major_px: float | None = None
    radius_minor_px: float | None = None
    radius_major_m: float | None = None
    radius_minor_m: float | None = None
    angle_deg: float | None = None
    contour_points: int = 0
    fit_rmse_px: float | None = None
    fit_rmse_m: float | None = None
    projected_area_m2: float | None = None
    projected_area_um2: float | None = None


class EllipseSizer(FrameProcessor):
    """Detect, measure, and draw one ellipse in each grayscale frame.

    The largest contour above ``min_area_px`` is fitted using OpenCV's
    ``fitEllipse``. ROI coordinates use ``(y_start, y_end, x_start, x_end)``;
    negative coordinates are offsets from the image edge and zero end
    coordinates select the full extent. ``fit_rmse_px`` is a radial boundary
    residual, not a statistical uncertainty estimate. ``frame_range`` is an
    inclusive range of source frame numbers; encoded frames outside it are
    returned unchanged and omitted from the CSV.
    """

    csv_columns = (
        "frame_number",
        "time_ms",
        "detected",
        "center_x_px",
        "center_y_px",
        "center_x_m",
        "center_y_m",
        "radius_major_px",
        "radius_minor_px",
        "radius_major_m",
        "radius_minor_m",
        "angle_deg",
        "contour_points",
        "fit_rmse_px",
        "fit_rmse_m",
        "projected_area_m2",
        "projected_area_um2",
    )

    def __init__(
        self,
        *,
        frame_range: tuple[int, int] | None = None,
        roi: Roi | None = None,
        polarity: Literal["bright", "dark"] = "bright",
        threshold_value: int | None = None,
        min_area_px: float = 5.0,
        outline_color: int = 160,
        outline_thickness: int = 2,
    ) -> None:
        if frame_range is not None and (
            not isinstance(frame_range, tuple)
            or len(frame_range) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                for value in frame_range
            )
        ):
            raise ValueError(
                "frame_range must be an inclusive pair of integers")
        if frame_range is not None and frame_range[0] > frame_range[1]:
            raise ValueError("frame_range start must not exceed its end")
        if roi is not None and (
            not isinstance(roi, tuple)
            or len(roi) != 4
            or any(
                isinstance(value, bool) or not isinstance(
                    value, (int, np.integer))
                for value in roi
            )
        ):
            raise ValueError(
                "roi must be a tuple of four integers "
                "(y_start, y_end, x_start, x_end)"
            )
        if polarity not in ("bright", "dark"):
            raise ValueError("polarity must be 'bright' or 'dark'")
        if threshold_value is not None and (
            isinstance(threshold_value, bool)
            or not isinstance(threshold_value, int)
            or not 0 <= threshold_value <= 255
        ):
            raise ValueError(
                "threshold_value must be an integer from 0 to 255")
        if (
            isinstance(min_area_px, bool)
            or not isinstance(min_area_px, (int, float))
            or not math.isfinite(min_area_px)
            or min_area_px < 0
        ):
            raise ValueError(
                "min_area_px must be a finite non-negative number")
        if (
            isinstance(outline_color, bool)
            or not isinstance(outline_color, int)
            or not 0 <= outline_color <= 255
        ):
            raise ValueError("outline_color must be an integer from 0 to 255")
        if (
            isinstance(outline_thickness, bool)
            or not isinstance(outline_thickness, int)
            or outline_thickness < 1
        ):
            raise ValueError("outline_thickness must be a positive integer")

        self.frame_range = (
            (int(frame_range[0]), int(frame_range[1]))
            if frame_range is not None
            else None
        )
        self.scale_m_per_px: float | None = None
        self.recording_frame_rate: float | None = None
        self.first_frame_number = 0
        self.time_offset_s = 0.0
        self.roi = roi
        self.polarity = polarity
        self.threshold_value = threshold_value
        self.min_area_px = float(min_area_px)
        self.outline_color = outline_color
        self.outline_thickness = outline_thickness
        self.measurements: list[EllipseMeasurement] = []
        self.csv_path: Path | None = None

    def set_video_context(self, **context: object) -> None:
        recording_frame_rate = context.get("recording_frame_rate")
        if (
            isinstance(recording_frame_rate, bool)
            or not isinstance(recording_frame_rate, (int, float))
            or not math.isfinite(recording_frame_rate)
            or recording_frame_rate <= 0
        ):
            raise ValueError(
                "EllipseSizer requires a positive finite recording_frame_rate"
            )
        scale_m_per_px = context.get("scale_m_per_px")
        if (
            isinstance(scale_m_per_px, bool)
            or not isinstance(scale_m_per_px, (int, float))
            or not math.isfinite(scale_m_per_px)
            or scale_m_per_px <= 0
        ):
            raise ValueError(
                "EllipseSizer requires a positive finite scale_m_per_px"
            )
        first_frame_number = context.get("first_frame_number")
        if (
            isinstance(first_frame_number, bool)
            or not isinstance(first_frame_number, int)
        ):
            raise ValueError(
                "EllipseSizer requires an integer first_frame_number"
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
                "EllipseSizer requires a finite time_offset_s or None"
            )

        self.recording_frame_rate = float(recording_frame_rate)
        self.scale_m_per_px = float(scale_m_per_px)
        self.first_frame_number = first_frame_number
        self.time_offset_s = float(time_offset_s)

    def _resolve_roi(
        self,
        frame_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape
        if self.roi is None:
            return 0, height, 0, width

        y_start, y_end, x_start, x_end = (int(value) for value in self.roi)
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
            0 <= y_start < y_end <= height
            and 0 <= x_start < x_end <= width
        ):
            raise ValueError("roi is empty or outside the frame dimensions")
        return y_start, y_end, x_start, x_end

    def _detect(
        self,
        frame: np.ndarray,
        frame_number: int,
    ) -> tuple[
        EllipseMeasurement,
        tuple[tuple[int, int], tuple[int, int], float] | None,
    ]:
        if (
            self.recording_frame_rate is None
            or self.scale_m_per_px is None
        ):
            raise RuntimeError(
                "EllipseSizer must receive video metadata from make_video "
                "before processing frames"
            )
        time_ms = (
            (frame_number - self.first_frame_number)
            / self.recording_frame_rate
            + self.time_offset_s
        ) * 1000
        y_start, y_end, x_start, x_end = self._resolve_roi(frame.shape)
        roi_frame = frame[y_start:y_end, x_start:x_end]
        threshold_type = (
            cv.THRESH_BINARY
            if self.polarity == "bright"
            else cv.THRESH_BINARY_INV
        )
        if self.threshold_value is None:
            threshold_type |= cv.THRESH_OTSU
            threshold = 0
        else:
            threshold = self.threshold_value
        _, mask = cv.threshold(
            roi_frame,
            threshold,
            255,
            threshold_type,
        )
        contours, _ = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE,
        )
        valid_contours = [
            contour
            for contour in contours
            if (
                len(contour) >= 5
                and cv.contourArea(contour) > 0
                and cv.contourArea(contour) >= self.min_area_px
            )
        ]
        if not valid_contours:
            return (
                EllipseMeasurement(
                    frame_number=frame_number,
                    detected=False,
                    time_ms=time_ms,
                ),
                None,
            )

        contour = max(valid_contours, key=cv.contourArea)
        (center_x, center_y), (diameter_a, diameter_b), angle = cv.fitEllipse(
            contour
        )
        center_x += x_start
        center_y += y_start
        radius_major = max(diameter_a, diameter_b) / 2
        radius_minor = min(diameter_a, diameter_b) / 2
        major_angle = angle if diameter_a >= diameter_b else angle + 90
        major_angle %= 180

        points = contour[:, 0, :].astype(np.float64)
        theta = math.radians(angle)
        delta_x = points[:, 0] - (center_x - x_start)
        delta_y = points[:, 1] - (center_y - y_start)
        local_x = math.cos(theta) * delta_x + math.sin(theta) * delta_y
        local_y = -math.sin(theta) * delta_x + math.cos(theta) * delta_y
        polar_angle = np.arctan2(local_y, local_x)
        measured_radius = np.hypot(local_x, local_y)
        radius_a = diameter_a / 2
        radius_b = diameter_b / 2
        fitted_radius = 1 / np.sqrt(
            (np.cos(polar_angle) / radius_a) ** 2
            + (np.sin(polar_angle) / radius_b) ** 2
        )
        fit_rmse = float(
            np.sqrt(np.mean((measured_radius - fitted_radius) ** 2)))
        fit_rmse_m = fit_rmse * self.scale_m_per_px
        radius_major_m = radius_major * self.scale_m_per_px
        radius_minor_m = radius_minor * self.scale_m_per_px
        projected_area_m2 = math.pi * radius_major_m * radius_minor_m

        measurement = EllipseMeasurement(
            frame_number=frame_number,
            detected=True,
            time_ms=time_ms,
            center_x_px=center_x,
            center_y_px=center_y,
            center_x_m=center_x * self.scale_m_per_px,
            center_y_m=center_y * self.scale_m_per_px,
            radius_major_px=radius_major,
            radius_minor_px=radius_minor,
            radius_major_m=radius_major_m,
            radius_minor_m=radius_minor_m,
            angle_deg=major_angle,
            contour_points=len(contour),
            fit_rmse_px=fit_rmse,
            fit_rmse_m=fit_rmse_m,
            projected_area_m2=projected_area_m2,
            projected_area_um2=projected_area_m2 * 1e12,
        )
        draw_ellipse = (
            (int(round(center_x)), int(round(center_y))),
            (max(1, int(round(diameter_a))), max(1, int(round(diameter_b)))),
            float(angle),
        )
        return measurement, draw_ellipse

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        *,
        is_preview: bool,
    ) -> np.ndarray:
        if frame.ndim != 2 or frame.dtype != np.uint8:
            raise ValueError("EllipseSizer requires an 8-bit grayscale frame")
        if self.frame_range is not None and not (
            self.frame_range[0] <= frame_number <= self.frame_range[1]
        ):
            return frame
        measurement, ellipse = self._detect(frame, frame_number)
        if ellipse is not None:
            cv.ellipse(
                frame,
                ellipse,
                color=self.outline_color,
                thickness=self.outline_thickness,
                lineType=cv.LINE_AA,
            )
        if not is_preview:
            self.measurements.append(measurement)
        return frame

    def finish(self, video_path: Path) -> None:
        """Write measurements beside the completed video."""
        self.csv_path = video_path.with_name(
            f"{video_path.stem}_ellipse_data.csv"
        )
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_columns)
            writer.writeheader()
            writer.writerows(asdict(item) for item in self.measurements)


def process_ellipse_data(
    csv_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Plot projected ellipse area in square micrometres against time."""
    use_tcm_poster_style(paper_mode=True)
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_name(
            f"{csv_path.stem.removesuffix('_ellipse_data')}_ellipse_analysis.pdf"
        )
    output_path = Path(output_path)

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    time_ms = np.array([float(row["time_ms"]) for row in rows])
    area_um2 = np.array([
        float(row["projected_area_um2"])
        if row["projected_area_um2"]
        else float("nan")
        for row in rows
    ])
    radius_major_m = np.array([
        float(row["radius_major_m"])
        if row["radius_major_m"]
        else float("nan")
        for row in rows
    ])
    radius_minor_m = np.array([
        float(row["radius_minor_m"])
        if row["radius_minor_m"]
        else float("nan")
        for row in rows
    ])

    circle_of_equal_area_radius_m = np.sqrt(area_um2 / math.pi) * 1e-6
    mean_radius = np.mean(
        circle_of_equal_area_radius_m[np.isfinite(circle_of_equal_area_radius_m)])

    figure, axis = plt.subplots()
    axis.plot(time_ms, radius_major_m * 1e3, label="Major axis")
    axis.plot(time_ms, radius_minor_m * 1e3, label="Minor axis")
    axis.plot(time_ms, circle_of_equal_area_radius_m *
              1e3, label="Circle of equal area")
    axis.axhline(mean_radius * 1e3, color=get_color(2),
                 linestyle="--", label=f"Mean: {mean_radius*1e3:.2f} mm")

    axis.legend(loc="lower center", ncol=2)
    # ncol=axis.get_legend_handles_labels()[0].__len__())
    # axis.set_xlabel("Time (ms)")
    # axis.xaxis.set_tick_params(labelbottom=False)
    axis.set_ylabel("Radius (mm)")
    axis.set_title("Falling droplet")
    append_unit_to_last_ticklabel(axis, axis="x", unit="ms")

    # axis[1].plot(time_ms, area_um2 / 1e6)
    # axis[1].set_ylabel("Ellipse area (mm²)")

    # # CHange y tick format to 2 digits after decimal
    # axis[1].yaxis.set_major_formatter(
    #     plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    set_grid(axis, mode="both")
    # set_grid(axis[1], mode="both")
    axis.set_ylim((1.1, 1.4))
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    # plt.show()
    pdf_to_png(output_path, dpi=160)
    return output_path


if __name__ == "__main__":
    from tcm_utils.video_analysis import EllipseSizer, process_ellipse_data
    ellipse_sizer = EllipseSizer(
        polarity="dark",
        roi=(20, 350, 560, 680),
        outline_color=255,
        outline_thickness=2,
        frame_range=(0, 228),
    )
    ellipse_sizer.csv_path = Path(
        "/Users/tommieverouden/Developer/twente-cough-machine/utils/.temp/timing_Ga26_59.5ms_newtube_P-001_20000fps_16700 nsec_ellipse_data.csv")
    # if video_path is not None and ellipse_sizer.csv_path is not None:
    process_ellipse_data(ellipse_sizer.csv_path)
