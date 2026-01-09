from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import cv2 as cv
import tifffile
import matplotlib.pyplot as plt

from tcm_utils.file_dialogs import ask_open_file, find_repo_root
from tcm_utils.time_utils import timestamp_str, timestamp_from_file
from tcm_utils.io_utils import (
    path_relative_to,
    save_metadata_json,
    copy_file_to_raw_subfolder,
    create_timestamped_filename,
    ensure_processed_artifact,
)


def _load_grayscale_image(path: Path) -> np.ndarray:
    """Load image as uint8 grayscale.

    Tries `tifffile` for TIFFs, otherwise uses OpenCV.
    """
    ext = path.suffix.lower()
    img: np.ndarray
    if ext in {".tif", ".tiff"}:
        img = tifffile.imread(str(path))
        if img.ndim == 3:
            img = img[..., 0]
    else:
        loaded = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if loaded is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = loaded

    if img.dtype != np.uint8:
        # Normalize to 0-255
        img = img.astype(np.float32)
        maxv = img.max() if img.size else 1.0
        if maxv == 0:
            maxv = 1.0
        img = (img / maxv * 255.0).astype(np.uint8)
    return img


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

    cv.destroyWindow("Calibration ROI")
    return rect


def _cluster_rows_by_y(centers: np.ndarray, rows: int) -> np.ndarray:
    """Assign a row index to each center by y-clustering.

    Uses approximate uniform spacing assumption.
    """
    if len(centers) == 0:
        return np.array([], dtype=int)

    sorted_idx = np.argsort(centers[:, 1])
    centers_sorted = centers[sorted_idx]

    if rows <= 1:
        labels = np.zeros(len(centers), dtype=int)
        return labels

    row_height = (centers_sorted[-1, 1] -
                  centers_sorted[0, 1]) / max(rows - 1, 1)
    labels = np.zeros(len(centers), dtype=int)
    current_row = 0
    labels[sorted_idx[0]] = current_row
    for i in range(1, len(centers)):
        if centers_sorted[i, 1] - centers_sorted[i - 1, 1] > row_height * 0.5:
            current_row += 1
        labels[sorted_idx[i]] = min(current_row, rows - 1)
    return labels


def estimate_spacing(centers: np.ndarray, rows: int, cols: int) -> Tuple[float, float]:
    """Estimate horizontal and vertical spacing (dx, dy) using medians."""
    if len(centers) != rows * cols:
        raise ValueError("Number of centers does not match rows*cols")

    row_labels = _cluster_rows_by_y(centers, rows)

    dx_list: list[float] = []
    for r in range(rows):
        row_pts = centers[row_labels == r]
        if len(row_pts) < 2:
            continue
        row_pts_sorted = row_pts[np.argsort(row_pts[:, 0])]
        dx_row = np.diff(row_pts_sorted[:, 0])
        dx_list.extend(dx_row.tolist())
    dx = float(np.median(dx_list)) if dx_list else 0.0

    y_per_row = [np.median(centers[row_labels == r, 1]) for r in range(rows)]
    dy = float(np.median(np.diff(y_per_row))) if rows > 1 else 0.0
    return dx, dy


def infer_grid_size(n_points: int) -> Tuple[int, int]:
    """Infer rows and cols from number of points by factor pairs.

    Returns (rows, cols) with rows <= cols.
    """
    best_r, best_c = 1, n_points
    for i in range(1, int(np.sqrt(n_points)) + 1):
        if n_points % i == 0:
            r, c = i, n_points // i
            if r <= c:
                best_r, best_c = r, c
    return best_r, best_c


def run_calibration(
    input_path: Path | None = None,
    distance_mm: float = 0.5,
    invert: bool = True,
    adaptive: bool = False,
    min_area: float = 3.0,
    max_area: float = 2000.0,
    timestamp_source: str = "file",
    output_dir: Path | None = None,
) -> int:
    repo_root = find_repo_root(Path(__file__))
    default_output = repo_root / "examples" / "calibration_demo"
    output_folder = (output_dir or default_output)
    output_folder.mkdir(parents=True, exist_ok=True)

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
            default_dir=output_folder,
            start=Path(__file__),
        )

    if data_file is None:
        print("No file selected. Exiting.")
        return 1

    data_file = Path(data_file).expanduser().resolve()
    if not data_file.exists():
        raise FileNotFoundError(f"Input file not found: {data_file}")

    # Load image
    img = _load_grayscale_image(data_file)
    img_h, img_w = img.shape[:2]

    # ROI selection
    print("Please select the ROI containing the calibration circle grid")
    r = _select_roi_colored(img)
    if r == (0, 0, 0, 0):
        print("ROI selection cancelled.")
        return 1
    x, y, w, h = map(int, r)

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

    # Infer grid size automatically
    rows, cols = infer_grid_size(len(centers))
    if rows * cols != len(centers):
        print(
            f"Warning: detected {len(centers)} centers, but rows*cols={rows*cols}. Proceeding with estimation."
        )

    # Estimate spacing
    try:
        dx, dy = estimate_spacing(centers, rows, cols)
    except ValueError:
        # Fallback: approximate spacing by nearest neighbor median
        # optional; if missing, fallback below
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=2).fit(centers)
        distances, _ = nbrs.kneighbors(centers)
        nn = distances[:, 1]
        dx = float(np.median(nn))
        dy = dx

    spacing_px = float(dx) if dx > 0 else float(
        np.linalg.norm(np.ptp(centers, axis=0))) / max(cols - 1, 1)
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

    # Outputs
    output_plot = output_folder / create_timestamped_filename(
        base_filename, timestamp, "calibration_plot", "pdf"
    )
    plt.savefig(output_plot)

    # Prepare CSV with centers and simple fit residuals
    # Map centers to row/col indices
    row_labels = _cluster_rows_by_y(centers, rows)
    # Estimate origin as min per row/col
    origin_x = float(np.min(centers[:, 0]))
    origin_y = float(np.min(centers[:, 1]))
    # For columns per row, sort by x within each row to assign indices
    col_indices = np.zeros(len(centers), dtype=int)
    for r in range(rows):
        idx_r = np.where(row_labels == r)[0]
        if idx_r.size == 0:
            continue
        row_pts = centers[idx_r]
        order = np.argsort(row_pts[:, 0])
        col_indices[idx_r[order]] = np.arange(len(idx_r))

    predicted_x = origin_x + col_indices * spacing_px
    # If dy is unreliable for single row, keep row spacing 0
    predicted_y = origin_y + row_labels * (dy if dy > 0 else 0.0)
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

    # Copy raw file to raw_data subfolder
    moved_raw = copy_file_to_raw_subfolder(data_file, output_folder)

    # Metadata JSON
    metadata = {
        "timestamp": timestamp,
        "timestamp_source": timestamp_source_description,
        "analysis_run_time": timestamp_str(),
        "input_file_original": path_relative_to(Path(data_file), repo_root),
        "raw_data_path": path_relative_to(moved_raw, repo_root),
        "output_files": {
            "plot_pdf": path_relative_to(output_plot, repo_root),
            "calibration_csv": path_relative_to(output_csv, repo_root),
        },
        "calibration": {
            "rows": int(rows),
            "cols": int(cols),
            "roi": {"x": x, "y": y, "width": w, "height": h},
            "spacing_px": float(spacing_px),
            "scale_mm_per_px": float(mm_per_px),
            "distance_mm_input": float(distance_mm),
            "image_size_px": {"width": int(img_w), "height": int(img_h)},
        },
    }

    metadata_path = output_folder / create_timestamped_filename(
        base_filename, timestamp, "metadata", "json"
    )
    save_metadata_json(metadata, metadata_path)

    print(f"Plot written to {output_plot}")
    print(f"CSV written to {output_csv}")
    print(f"Metadata written to {metadata_path}")
    print(
        f"Estimated scale: {mm_per_px:.6f} mm/px (spacing {spacing_px:.3f} px)")
    return 0


def ensure_calibration(
    input_path: Path | None = None,
    distance_mm: float = 0.5,
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
    """

    repo_root = find_repo_root(Path(__file__))
    default_output = repo_root / "examples" / "calibration_demo"

    def _runner(image_path: Path, dest: Path) -> int:
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
