from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from tcm_utils.file_dialogs import ask_open_file, ask_directory, find_repo_root
from tcm_utils.time_utils import timestamp_str, timestamp_from_file
from tcm_utils.io_utils import (
    load_image_with_path,
    path_relative_to,
    save_metadata_json,
    copy_file_to_raw_subfolder,
    create_timestamped_filename,
    ensure_processed_artifact,
    prompt_input,
)


def detect_circle_centers(
    roi_img: np.ndarray,
    min_area: float = 3.0,
    max_area: float = 2000.0,
    invert: bool = True,
    use_adaptive: bool = False,
    auto_retry: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect circle centers using binary thresholding and contours.

    Returns (centers Nx2 array, binary image used).
    """
    if roi_img.ndim != 2:
        roi_img = cv.cvtColor(roi_img, cv.COLOR_BGR2GRAY)

    def _run(invert_flag: bool, adaptive_flag: bool) -> Tuple[np.ndarray, np.ndarray]:
        if adaptive_flag:
            binary_local = cv.adaptiveThreshold(
                roi_img,
                255,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY_INV if invert_flag else cv.THRESH_BINARY,
                11,
                2,
            )
        else:
            thresh_type = cv.THRESH_BINARY_INV if invert_flag else cv.THRESH_BINARY
            _, binary_local = cv.threshold(
                roi_img, 0, 255, thresh_type + cv.THRESH_OTSU
            )

        binary_local = binary_local.astype(np.uint8)
        # Light smoothing/closing to unify blobs
        binary_local = cv.medianBlur(binary_local, 3)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        binary_local = cv.morphologyEx(binary_local, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(
            binary_local, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        centers_local = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            M = cv.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centers_local.append([cx, cy])

        return np.array(centers_local, dtype=np.float64), binary_local

    attempts = [(invert, use_adaptive)]
    if auto_retry:
        attempts += [
            (not invert, use_adaptive),
            (invert, not use_adaptive),
            (not invert, not use_adaptive),
        ]

    best_centers: np.ndarray | None = None
    best_binary: np.ndarray | None = None
    for inv_flag, adap_flag in attempts:
        centers_arr, binary_used = _run(inv_flag, adap_flag)
        if best_centers is None or len(centers_arr) > len(best_centers):
            best_centers, best_binary = centers_arr, binary_used
        if len(centers_arr) > 0:
            break
    if best_centers is None:
        best_centers = np.empty((0, 2))
    if best_binary is None:
        best_binary = np.zeros_like(roi_img, dtype=np.uint8)
    return best_centers, best_binary


def _select_roi_colored(img_gray: np.ndarray, color=(255, 0, 255)) -> tuple[int, int, int, int]:
    """Custom ROI selector with high-contrast rectangle.

    Auto-confirms on mouse release. ESC cancels.
    Returns (x, y, w, h); (0,0,0,0) if cancelled.
    """
    display = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    drawing = False
    finished = False
    start_pt = (0, 0)
    rect = (0, 0, 0, 0)

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, start_pt, rect, finished
        temp = display.copy()
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
        elif event == cv.EVENT_MOUSEMOVE and drawing:
            cv.rectangle(temp, start_pt, (x, y), color, 2)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            rect = (
                min(start_pt[0], x),
                min(start_pt[1], y),
                abs(x - start_pt[0]),
                abs(y - start_pt[1]),
            )
            cv.rectangle(temp, start_pt, (x, y), color, 2)
            finished = True
        cv.imshow("Calibration ROI - press ESC to cancel", temp)

    cv.namedWindow("Calibration ROI - press ESC to cancel")
    cv.setMouseCallback("Calibration ROI - press ESC to cancel", on_mouse)
    cv.imshow("Calibration ROI - press ESC to cancel", display)
    while True:
        key = cv.waitKey(20) & 0xFF
        if key == 27:  # ESC cancels
            rect = (0, 0, 0, 0)
            break
        if finished:
            break

    # cv.destroyWindow("Calibration ROI")
    # TODO: Seems to be broken (might be due to version of opencv?)
    return rect


def _rotate_points(pts: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate Nx2 points by angle around origin."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ R.T


def _estimate_grid_angle(centers: np.ndarray) -> float:
    """Estimate grid axis angle from local neighbor-vector orientations.

    Uses vectors to nearby neighbors, keeps near-first-neighbor distances,
    folds angles modulo 90 degrees, and selects the dominant orientation bin.
    """
    n_pts = len(centers)
    if n_pts < 2:
        return 0.0

    k = min(8, n_pts - 1)
    if k <= 0:
        return 0.0

    vectors: list[np.ndarray] = []
    nearest_dists: list[float] = []

    for i in range(n_pts):
        delta = centers - centers[i]
        d = np.linalg.norm(delta, axis=1)
        d[i] = np.inf

        nn_idx = np.argpartition(d, k)[:k]
        nn_d = d[nn_idx]
        if len(nn_d) == 0:
            continue
        nearest_dists.append(float(np.min(nn_d)))

        for j in nn_idx:
            if np.isfinite(d[j]) and d[j] > 0:
                vectors.append(delta[j])

    if not vectors or not nearest_dists:
        return 0.0

    nn_med = float(np.median(nearest_dists))
    if nn_med <= 0:
        return 0.0

    vec_arr = np.array(vectors, dtype=np.float64)
    vec_dist = np.linalg.norm(vec_arr, axis=1)

    # Keep vectors near one lattice step to suppress diagonal and long-range pairs.
    keep = (vec_dist >= 0.5 * nn_med) & (vec_dist <= 1.35 * nn_med)
    if not np.any(keep):
        return 0.0

    angles = np.arctan2(vec_arr[keep, 1], vec_arr[keep, 0])
    # Fold to the 90-degree periodic orientation range.
    folded = ((angles + np.pi / 4.0) % (np.pi / 2.0)) - np.pi / 4.0

    n_bins = 180
    hist, edges = np.histogram(
        folded,
        bins=n_bins,
        range=(-np.pi / 4.0, np.pi / 4.0),
    )
    i_peak = int(np.argmax(hist))
    bin_width = edges[1] - edges[0]
    center = 0.5 * (edges[i_peak] + edges[i_peak + 1])

    # Refine around the peak with a robust median.
    peak_mask = np.abs(folded - center) <= 2.0 * bin_width
    if np.any(peak_mask):
        return float(np.median(folded[peak_mask]))
    return float(center)


def _estimate_axis_spacings(rot_centers: np.ndarray, k_neighbors: int = 6) -> Tuple[float, float]:
    """Estimate lattice spacings (dx, dy) in approximately axis-aligned coordinates."""
    n_pts = len(rot_centers)
    if n_pts < 2:
        return 0.0, 0.0

    k = min(max(k_neighbors, 2), n_pts - 1)
    dx_candidates: list[float] = []
    dy_candidates: list[float] = []
    nn_candidates: list[float] = []

    for i in range(n_pts):
        delta = rot_centers - rot_centers[i]
        d = np.linalg.norm(delta, axis=1)
        d[i] = np.inf
        nn_idx = np.argpartition(d, k)[:k]

        for j in nn_idx:
            vx, vy = delta[j]
            ax, ay = abs(vx), abs(vy)
            dist = float(np.hypot(vx, vy))
            if dist <= 0:
                continue
            nn_candidates.append(dist)
            # Keep mostly axis-parallel neighbor vectors.
            if ax >= 1.5 * ay and ax > 0:
                dx_candidates.append(ax)
            elif ay >= 1.5 * ax and ay > 0:
                dy_candidates.append(ay)

    nn_med = float(np.median(nn_candidates)) if nn_candidates else 0.0
    dx = float(np.median(dx_candidates)) if dx_candidates else nn_med
    dy = float(np.median(dy_candidates)) if dy_candidates else nn_med
    return dx, dy


def _quantize_axis_with_phase(coords: np.ndarray, spacing: float) -> Tuple[np.ndarray, float, int]:
    """Quantize 1D coordinates to lattice indices using an estimated phase.

    Returns:
        idx_shifted: integer indices shifted to start at 0
        phase: phase in lattice units (fractional offset)
        k_min: pre-shift minimum integer lattice index
    """
    if spacing <= 0:
        return np.zeros(len(coords), dtype=int), 0.0, 0

    u = coords / spacing

    # Brute-force phase search in [0, 1) to find the best lattice offset.
    # This avoids anchoring on edge points, which can be incomplete.
    phase_grid = np.linspace(0.0, 1.0, 256, endpoint=False)
    best_phase = 0.0
    best_cost = np.inf
    for phase in phase_grid:
        snapped = np.rint(u - phase) + phase
        cost = float(np.median(np.abs(u - snapped)))
        if cost < best_cost:
            best_cost = cost
            best_phase = float(phase)

    k = np.rint(u - best_phase).astype(int)
    k_min = int(np.min(k)) if len(k) else 0
    return k - k_min, best_phase, k_min


def _merge_sparse_edge_bins(indices: np.ndarray, min_ratio: float = 0.6, min_count: int = 3) -> np.ndarray:
    """Merge sparse edge bins into adjacent bins to avoid spurious outer rows/cols."""
    if len(indices) == 0:
        return indices

    idx = indices.astype(int).copy()
    idx -= int(np.min(idx))

    while True:
        counts = np.bincount(idx)
        if len(counts) <= 1:
            break

        ref = float(np.median(counts[counts > 0])
                    ) if np.any(counts > 0) else 0.0
        threshold = max(min_count, int(np.ceil(ref * min_ratio)))
        changed = False

        # If only the outer edge bin is weak, treat it as clipped and merge inward.
        if counts[0] < threshold and len(counts) > 1:
            idx[idx == 0] = 1
            idx -= 1
            changed = True

        counts = np.bincount(idx)
        # Right edge too sparse: merge into neighbor bin.
        max_bin = len(counts) - 1
        if max_bin >= 1 and counts[max_bin] < threshold:
            idx[idx == max_bin] = max_bin - 1
            changed = True

        if not changed:
            break

    idx -= int(np.min(idx))
    return idx


def infer_grid_geometry(centers: np.ndarray) -> Tuple[int, int, np.ndarray, np.ndarray, float, float, float, float, float]:
    """Infer grid geometry from centers, allowing partial rows/cols.

    Pipeline:
    1) Estimate global grid angle and rotate points to near-axis alignment.
    2) Estimate dx/dy from mostly axis-parallel neighbor vectors.
    3) Quantize rotated coordinates with a learned lattice phase.
    4) Merge sparse edge bins to avoid overcounting clipped border rows/cols.

    Returns:
        rows, cols, row_indices, col_indices, theta, dx, dy, x0_rot, y0_rot
    """
    if len(centers) == 0:
        raise ValueError("No centers available for grid inference")

    theta = _estimate_grid_angle(centers)
    rot = _rotate_points(centers, -theta)
    dx, dy = _estimate_axis_spacings(rot)

    # Fallback spacing based on nearest-neighbor distances.
    if dx <= 0 or dy <= 0:
        n_pts = len(rot)
        nn = []
        for i in range(n_pts):
            d = np.linalg.norm(rot - rot[i], axis=1)
            d[i] = np.inf
            nn.append(float(np.min(d)))
        nn_med = float(np.median(nn)) if nn else 1.0
        if dx <= 0:
            dx = nn_med
        if dy <= 0:
            dy = nn_med

    # Quantize onto integer lattice coordinates with phase compensation.
    col_indices, x_phase, x_k_min = _quantize_axis_with_phase(rot[:, 0], dx)
    row_indices, y_phase, y_k_min = _quantize_axis_with_phase(rot[:, 1], dy)

    # Remove edge-only bins caused by cropped top/bottom/left/right rows.
    col_indices = _merge_sparse_edge_bins(col_indices)
    row_indices = _merge_sparse_edge_bins(row_indices)

    rows = int(np.max(row_indices)) + 1
    cols = int(np.max(col_indices)) + 1

    # Keep reporting format consistent with rows <= cols.
    if rows > cols:
        row_indices, col_indices = col_indices.copy(), row_indices.copy()
        rows, cols = cols, rows
        dx, dy = dy, dx
        theta += np.pi / 2.0
        x0_rot = float((y_k_min + y_phase) * dy)
        y0_rot = float((x_k_min + x_phase) * dx)
    else:
        x0_rot = float((x_k_min + x_phase) * dx)
        y0_rot = float((y_k_min + y_phase) * dy)

    return rows, cols, row_indices, col_indices, theta, dx, dy, x0_rot, y0_rot


def run_calibration(
    input_path: Path | None = None,
    distance_mm: float | None = None,
    invert: bool = True,
    adaptive: bool = False,
    min_area: float = 3.0,
    max_area: float = 2000.0,
    timestamp_source: str = "file",
    output_dir: Path | None = None,
    roi: tuple[int, int, int, int] | None = None,
) -> float:
    repo_root = find_repo_root(Path(__file__))

    # Select input image
    if input_path is not None:
        data_file: Path | None = Path(input_path).expanduser().resolve()
    else:
        data_file = ask_open_file(
            key="camera_calibration",
            title="Select calibration image",
            filetypes=[
                ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
            default_dir=repo_root,
            start=Path(__file__),
        )

    if data_file is None:
        print("No file selected. Exiting.")
        return 1

    data_file = Path(data_file).expanduser().resolve()
    if not data_file.exists():
        raise FileNotFoundError(f"Input file not found: {data_file}")

    # Load image (may use converted TIFF path)
    img, loaded_image_path = load_image_with_path(data_file)
    img_h, img_w = img.shape[:2]

    # ROI selection
    if roi is None:
        print("Please select the ROI containing the calibration circle grid (press ESC to cancel)")
        r = _select_roi_colored(img)
        if r == (0, 0, 0, 0):
            print("ROI selection cancelled.")
            return 1
        x, y, w, h = map(int, r)
    else:
        if len(roi) != 4:
            raise ValueError("roi must be a 4-tuple: (x, y, width, height)")

        x, y, w, h = map(int, roi)
        if w <= 0 or h <= 0:
            raise ValueError("roi width and height must be > 0")

        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))

    roi_img = img[y: y + h, x: x + w]
    centers_roi, binary = detect_circle_centers(
        roi_img,
        min_area=min_area,
        max_area=max_area,
        invert=invert,
        use_adaptive=adaptive,
    )

    # Offset to full image coordinates
    centers = centers_roi.copy()
    if len(centers) == 0:
        print("No circles detected in ROI.")
        return 1
    centers[:, 0] += x
    centers[:, 1] += y

    # Infer rotated lattice geometry before row/column clustering.
    # This keeps indexing stable when the ROI clips partial border rows.
    rows, cols, row_labels, col_indices, theta, dx, dy, x0_rot, y0_rot = infer_grid_geometry(
        centers
    )
    theta_deg = float(np.rad2deg(theta))
    # Grid orientation is 90-degree periodic, so report the canonical small tilt.
    theta_deg = ((theta_deg + 45.0) % 90.0) - 45.0
    print(
        f"Detected dot grid size (rotated by {theta_deg:.1f}°): {cols}x{rows}")

    spacing_candidates = [v for v in (dx, dy) if v > 0]
    if spacing_candidates:
        spacing_px = float(np.median(spacing_candidates))
    else:
        spacing_px = float(np.linalg.norm(
            np.ptp(centers, axis=0))) / max(cols - 1, 1)

    if distance_mm is None or distance_mm == "" or (isinstance(distance_mm, (int, float)) and distance_mm <= 0):
        spacing_input = prompt_input(
            "Enter the spacing between dots in millimeters (leave empty to cancel): ",
            value_type="float",
            allow_empty=True,
            min_value=0.0,
            exclusive_min=True,
        )
        if spacing_input is None:
            print("Calibration cancelled: no spacing provided.")
            return 1
        distance_mm = float(spacing_input)

    mm_per_px = float(distance_mm) / spacing_px

    # Build visualization
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for (cxp, cyp) in centers:
        center_pt = (int(round(cxp)), int(round(cyp)))
        cv.circle(vis, center_pt, 3, (0, 0, 255), -1)
        # Add a cross marker for clearer center indication
        cv.drawMarker(vis, center_pt, (0, 255, 255),
                      markerType=cv.MARKER_CROSS, markerSize=10, thickness=1)
    # Use a high-contrast rectangle color (magenta) so it remains visible on light backgrounds
    cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(vis[..., ::-1])
    plt.title(
        f"Circle grid: {cols}x{rows}\nSpacing ~ {spacing_px:.3f} px | Scale {mm_per_px:.6f} mm/px"
    )
    plt.axis("off")
    plt.tight_layout()

    base_filename = Path(data_file).stem
    if timestamp_source == "file":
        timestamp = timestamp_from_file(data_file, prefer_creation=True)
        timestamp_source_description = "file_creation_time"
    else:
        timestamp = timestamp_str()
        timestamp_source_description = "current_time"

    if output_dir is not None:
        output_folder = Path(output_dir).expanduser().resolve()
    else:
        selected_output_dir = ask_directory(
            key="camera_calibration_output",
            title="Select output directory for calibration results",
            default_dir=data_file.parent,
            start=Path(__file__),
        )
        if selected_output_dir is None:
            print("Calibration cancelled: no output directory selected.")
            return 1
        output_folder = selected_output_dir

    output_folder.mkdir(parents=True, exist_ok=True)

    # Outputs
    output_plot = output_folder / create_timestamped_filename(
        base_filename, timestamp, "calibration_plot", "pdf"
    )
    plt.savefig(output_plot)

    # Prepare CSV with centers and angle-aware lattice fit residuals.
    predicted_rot_x = x0_rot + col_indices * dx
    predicted_rot_y = y0_rot + row_labels * dy
    predicted_rot = np.column_stack((predicted_rot_x, predicted_rot_y))
    predicted_xy = _rotate_points(predicted_rot, theta)
    predicted_x = predicted_xy[:, 0]
    predicted_y = predicted_xy[:, 1]
    residuals = np.sqrt((centers[:, 0] - predicted_x)
                        ** 2 + (centers[:, 1] - predicted_y) ** 2)

    output_csv = output_folder / create_timestamped_filename(
        base_filename, timestamp, "calibration", "csv"
    )
    csv_header = "center_x_px,center_y_px,row_index,col_index,pred_x_px,pred_y_px,residual_px"
    csv_data = np.column_stack(
        (centers[:, 0], centers[:, 1], row_labels,
         col_indices, predicted_x, predicted_y, residuals)
    )
    np.savetxt(output_csv, csv_data, delimiter=",",
               header=csv_header, comments="")

    # Copy the actual loaded input file to raw_data (converted TIFF when used)
    moved_raw = copy_file_to_raw_subfolder(loaded_image_path, output_folder)

    # Metadata JSON
    metadata = {
        "timestamp": timestamp,
        "timestamp_source": timestamp_source_description,
        "analysis_run_time": timestamp_str(),
        "input_file_original": path_relative_to(Path(data_file), repo_root),
        "input_file_used": path_relative_to(Path(loaded_image_path), repo_root),
        "raw_data_path": path_relative_to(moved_raw, repo_root),
        "output_files": {
            "plot_pdf": path_relative_to(output_plot, repo_root),
            "calibration_csv": path_relative_to(output_csv, repo_root),
        },
        "calibration": {
            "rows": int(rows),
            "cols": int(cols),
            "grid_rotation_deg": float(theta_deg),
            "roi": {"x": x, "y": y, "width": w, "height": h},
            "spacing_px": float(spacing_px),
            "scale_mm_per_px": float(mm_per_px),
            "scale_m_per_px": float(mm_per_px) / 1000.0,
            "distance_mm_input": float(distance_mm),
            "image_size_px": {"width": int(img_w), "height": int(img_h)},
            "image_size_m": {"width": float(img_w) * mm_per_px / 1000.0,
                             "height": float(img_h) * mm_per_px / 1000.0},
        },
    }

    metadata_path = output_folder / create_timestamped_filename(
        base_filename, timestamp, "metadata", "json"
    )
    save_metadata_json(metadata, metadata_path)

    print(f"- Plot written to {output_plot}")
    print(f"- CSV written to {output_csv}")
    print(f"- Metadata written to {metadata_path}")
    print(
        f"Estimated scale: {mm_per_px:.6f} mm/px (spacing {spacing_px:.3f} px)")
    return mm_per_px


def ensure_calibration(
    input_path: Path | None = None,
    distance_mm: float | None = None,
    invert: bool = True,
    adaptive: bool = False,
    min_area: float = 3.0,
    max_area: float = 2000.0,
    timestamp_source: str = "file",
    output_dir: Path | None = None,
) -> Path | None:
    """Return calibration metadata path or run calibration to create it.

    Resolution order (no subfolder scanning):
    1) If ``input_path`` is a ``*_metadata.json`` file, return it.
    2) If ``input_path`` is a folder containing ``*_metadata.json``, return the latest one.
    3) If ``input_path`` is a ``.tif/.tiff`` file, run calibration on it (prompt for output dir when not provided) and return the created metadata JSON.
    4) If ``input_path`` is a folder containing a ``.tif/.tiff`` file, run calibration on that file and return the resulting metadata JSON.
    5) Otherwise, ask the user to select a metadata JSON or calibration image file.

    If ``distance_mm`` is omitted, the user will be prompted for the dot spacing.
    """

    repo_root = find_repo_root(Path(__file__))
    default_output = repo_root / "examples" / "calibration_demo"

    def _runner(image_path: Path, dest: Path) -> float:
        return run_calibration(
            input_path=image_path,
            distance_mm=distance_mm,
            invert=invert,
            adaptive=adaptive,
            min_area=min_area,
            max_area=max_area,
            timestamp_source=timestamp_source,
            output_dir=dest,
        )

    return ensure_processed_artifact(
        input_path=input_path,
        output_dir=output_dir,
        metadata_pattern="*_metadata.json",
        source_patterns=("*.tif", "*.tiff"),
        output_dir_key="camera_calibration_output",
        output_dir_title="Select output directory for calibration results",
        default_output_dir=default_output,
        run_processor=_runner,
        prompt_key="camera_calibration_metadata_or_image",
        prompt_title="Select calibration metadata JSON or calibration image",
        prompt_filetypes=[
            ("Calibration metadata", "*_metadata.json"),
            ("Image files", "*.tif *.tiff"),
            ("All files", "*.*"),
        ],
        start_path=Path(__file__),
    )


if __name__ == "__main__":
    run_calibration()
