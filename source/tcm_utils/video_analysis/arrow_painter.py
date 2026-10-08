"""Draw arrows and the current flow rate on a video frame."""

import csv
from pathlib import Path
import numpy as np
from tcm_utils.video_maker import FrameProcessor, draw_label


class ArrowPainter(FrameProcessor):
    """Draw arrows and the current flow rate on a video frame."""

    def __init__(
            self,
            *,

            flow_rate_csv_path: Path,
            velocity_csv_path: Path,
            arrow_scale_px_per_m_s: float,
    ) -> None:

        #

        return

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
    ) -> np.ndarray:
        # Implementation for processing each frame

        return frame


def read_flow_rate_csv(flow_rate_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(flow_rate_path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data found in {flow_rate_path}")
    if getattr(data, "ndim", 0) == 0:
        data = np.array([data], dtype=data.dtype)

    return (
        np.asarray(data["time_s"], dtype=np.float64),
        np.asarray(data["flow_rate_L_s"], dtype=np.float64),
    )


def read_velocity_csv(velocity_path: Path) -> dict[str, object]:
    """Read a stitched velocity CSV into columns and per-window time series.

    The ``windows`` entry is keyed by ``(win_x, win_y)`` and contains one
    dictionary of arrays per PIV window.  Each window's rows are sorted by
    ``time_s`` so its velocity components can be plotted directly against
    time.
    """
    data = np.genfromtxt(velocity_path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")
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
        columns[name] = np.asarray(columns[name], dtype=np.float64)

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
            columns["time_s"][window_indices], kind="stable")
        window_indices = window_indices[sort_order]
        key = (float(win_x), float(win_y))
        windows[key] = {
            name: values[window_indices] for name, values in columns.items()
        }

    return {**columns, "windows": windows}
