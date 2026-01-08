import os, glob
import numpy as np
import tifffile
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.cluster import DBSCAN
from scipy.optimize import minimize

"""
Calibration which seems to work pretty well. You select the area, where the circles are and confirm and off you go.
"""

def params_findcirclegrid():
    params = cv.SimpleBlobDetector_Params()

    # Filter by area (adjust to your circle size)
    params.filterByArea = True
    params.minArea = 5       # minimum area of a circle
    params.maxArea = 5000    # maximum area

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1  # 0 = any shape, 1 = perfect circle

    # Filter by convexity
    params.filterByConvexity = False

    # Filter by inertia (elongation)
    params.filterByInertia = True
    params.minInertiaRatio = 0.1  # 0 = elongated allowed, 1 = perfect circle

    # Create detector
    detector = cv.SimpleBlobDetector_create(params)
    return detector
detector = params_findcirclegrid()

def get_resolution(centers,rows,cols):
    sorted_centers = centers[np.lexsort((centers[:,0],centers[:,1]))]

    X = sorted_centers[:,0]
    Y = sorted_centers[:,1]

    # Reshape according to rows × cols
    X_g = X.reshape(rows, cols)
    Y_g = Y.reshape(rows, cols)

    # Sort each row by x
    ind = np.argsort(X_g, axis=1)
    X_g = np.take_along_axis(X_g, ind, axis=1)
    Y_g = np.take_along_axis(Y_g, ind, axis=1)

    # Normalize to top-left origin
    X_g -= X_g[0,0]
    Y_g -= Y_g[0,0]

    # --- Error functions with separate rows and cols ---
    def error_func_spacing(spacing):
        """
        spacing: scalar (assumes square spacing for simplicity)
        """
        X0_g, Y0_g = np.meshgrid(np.arange(cols)*spacing, np.arange(rows)*spacing)

        error = np.sum(np.abs(X0_g - X_g) + np.abs(Y0_g - Y_g))
        return error



    # --- Optimize spacing ---
    res = minimize(error_func_spacing, 40)
    return res
def get_contour_centers(roi_img, roi=None, min_area=3, max_area=2000, invert=True, use_adaptive=False):
    """
    Extract circle centers from a calibration image using contours.

    Args:
        tiff_path: Path to TIFF calibration image
        roi: Optional tuple (x, y, w, h). If None, use full image.
        min_area, max_area: Contour area thresholds in pixels
        invert: True if circles are dark on light background
        use_adaptive: True to use adaptive thresholding instead of Otsu

    Returns:
        centers: Nx2 numpy array of (x, y) coordinates
        roi_img: the ROI used for detection
    """
    

    # Binarize
    if use_adaptive:
        binary = cv.adaptiveThreshold(roi_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV if invert else cv.THRESH_BINARY, 11, 2)
    else:
        thresh_type = cv.THRESH_BINARY_INV if invert else cv.THRESH_BINARY
        _, binary = cv.threshold(roi_img, 0, 255, thresh_type + cv.THRESH_OTSU)
    binary =binary.astype(np.uint8)
    # Optional: morphological close to fill small gaps

    
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centers = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Compute centroid
        M = cv.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"]/M["m00"]
        cy = M["m01"]/M["m00"]
        centers.append([cx, cy])


    centers = np.array(centers)
 

    return centers, binary

def robust_estimate_grid(centers, rows_guess=None, cols_guess=None):
    """
    Estimate an ordered grid from unordered centers using residual fitting.
    Works for large grids even if points are slightly misaligned.

    Args:
        centers: Nx2 array of points
        rows_guess, cols_guess: optional initial guess

    Returns:
        (cols, rows), ordered_grid (rows x cols x 2)
    """
    N = len(centers)

    # --- 1. Guess rows/cols if not provided ---
    if rows_guess is None or cols_guess is None:
        # assume roughly square grid
        side = int(np.round(np.sqrt(N)))
        rows_guess = side
        cols_guess = side

    # --- 2. Create ideal grid coordinates ---
    gx, gy = np.meshgrid(np.arange(cols_guess), np.arange(rows_guess))
    ideal_grid = np.column_stack([gx.ravel(), gy.ravel()])

    # --- 3. Residual function for rotation+scale+translation ---
    def residual(params):
        dx, dy, sx, sy, theta = params
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        transformed = (ideal_grid * np.array([sx, sy]) @ R.T) + np.array([dx, dy])
        # find nearest neighbors
        if transformed.shape[0] != centers.shape[0]:
            # truncate to min size
            k = min(transformed.shape[0], centers.shape[0])
            return np.sum(np.linalg.norm(transformed[:k] - centers[:k], axis=1))
        return np.sum(np.linalg.norm(transformed - centers, axis=1))

    # --- 4. Optimize ---
    params0 = [0,0,1,1,0]
    res = minimize(residual, params0, method='Powell')
    dx, dy, sx, sy, theta = res.x

    # --- 5. Transform ideal grid to match centers ---
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    ordered_grid = (ideal_grid * np.array([sx, sy]) @ R.T) + np.array([dx, dy])
    ordered_grid = ordered_grid.reshape(rows_guess, cols_guess, 2)

    return ordered_grid
def fit_best_grid(centers):
    """
    Fit an ordered grid to centers when rows/cols unknown.
    Tries all factor pairs and returns the one with minimal residual.
    
    Args:
        centers: Nx2 array of points
    
    Returns:
        best_rows, best_cols, best_grid, best_residual
    """
    N = len(centers)
    
    # Generate factor pairs
    def factor_pairs(n):
        pairs = []
        for i in range(1, int(np.sqrt(n)) + 1):
            if n % i == 0:
                pairs.append((i, n // i))
        return pairs
    
    pairs = factor_pairs(N)
   

    best_res = np.inf
    best_grid = None
    best_rows, best_cols = None, None
    for r, c in factor_pairs(N):
        for rows, cols in [(r, c), (c, r)]:
            try:
                grid = robust_estimate_grid(centers, rows, cols)
                # Compute residual: sum of distances to nearest centers
                residual = np.sum(np.linalg.norm(grid.reshape(-1,2) - centers, axis=1))
                
                if residual < best_res:
                    best_res = residual
                    best_grid = grid
                    best_rows, best_cols = rows, cols
            except Exception as e:
                # ignore factor pairs that fail
                continue
 
    return best_rows, best_cols, best_grid, best_res
def estimate_spacing(centers, rows, cols):
    """
    Estimate horizontal and vertical spacing of a rectangular grid.
    
    Args:
        centers: Nx2 array of points
        rows, cols: known grid size

    Returns:
        dx, dy: average spacing in x and y directions
    """
    if len(centers) != rows*cols:
        raise ValueError("Number of centers does not match rows*cols")

    # --- 1. Reshape into rough grid by sorting ---
    # Sort by y (row-wise)
    sorted_idx = np.argsort(centers[:,1])
    centers_sorted = centers[sorted_idx]
    
    # Approximate row height
    row_height = (centers_sorted[-1,1] - centers_sorted[0,1]) / (rows-1)
    
    # Cluster points into rows based on y
    row_labels = np.zeros(len(centers), dtype=int)
    current_row = 0
    row_labels[sorted_idx[0]] = current_row
    for i in range(1, len(centers)):
        if centers_sorted[i,1] - centers_sorted[i-1,1] > row_height*0.5:  # half the average row height
            current_row += 1
        row_labels[sorted_idx[i]] = current_row

    # --- 2. Compute horizontal spacing per row ---
    dx_list = []
    for r in range(rows):
        row_pts = centers[row_labels==r]
        if len(row_pts) < 2:
            continue
        row_pts_sorted = row_pts[np.argsort(row_pts[:,0])]
        dx_row = np.diff(row_pts_sorted[:,0])
        dx_list.extend(dx_row)
    dx = np.median(dx_list)

    # --- 3. Compute vertical spacing between rows ---
    # Take median y per row
    y_per_row = [np.median(centers[row_labels==r,1]) for r in range(rows)]
    dy = np.median(np.diff(y_per_row))

    return dx, dy

def get_calibration(folder, save=True, distance_between = 0.5):
    """
    Calibration workflow:
    - Select ROI for calibration (OpenCV window)
    - Enter real-world width of ROI in mm
    - Select ROI for rotation (OpenCV window)
    - Save results in .npz
    """
    def save_backup(proc_path: str, file_name: str, **kwargs) -> bool:
        if not kwargs:
            print("⚠️ No variables provided for saving.")
            return False
        if not file_name.endswith(".npz"):
            file_name += ".npz"
        file_path = os.path.join(proc_path, file_name)
        np.savez(file_path, **kwargs)
        print(f"✅ Saved calibration data to {file_path}")
        return True

    # 1. Load existing calibration if available
    calibration_done = glob.glob(os.path.join(folder, "*calibration*.npz"))
    if calibration_done:
        with np.load(calibration_done[0]) as data:
            scale = data["resolution_mm_per_px"].item()
        print(f"📂 Loaded calibration from {calibration_done[0]}")
        return scale



    # 3. Otherwise, do manual calibration

    calibration_files = glob.glob(os.path.join(folder, "*calibration*.tif"))
    print(calibration_files)

    if not calibration_files:
        print("❌ No calibration TIFF found!")
        return None

    calibration_path = calibration_files[0]
    print(f"✅ Calibration file: {calibration_path}")

    # --- Load TIFF ---
    img = tifffile.imread(calibration_path)

    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    if img.ndim == 3:
        img = img[..., 0]  # take first channel if RGB
    y_length_pix , x_length_pix  =img.shape
    # --- Step 1: Select Calibration ROI ---
    print("👉 Select CALIBRATION ROI, press ENTER/SPACE to confirm, ESC to cancel.")
    r = cv.selectROI("Calibration ROI", img, fromCenter=False, showCrosshair=False)
    cv.destroyWindow("Calibration ROI")
    if r == (0, 0, 0, 0):
        print("❌ Calibration ROI selection cancelled.")
        return None
    x, y, w, h = map(int, r)



    print(f"📏 Calibration ROI size: {w} × {h} px")

    roi_img = img[y:y+h, x:x+w]


    centers,binary= get_contour_centers(roi_img)

    centers[:, 0] += x
    centers[:, 1] += y
        # Visualization
    vis = cv.cvtColor(roi_img, cv.COLOR_GRAY2BGR)
    for pt in centers:
        cv.circle(vis, (int(pt[0]- x), int(pt[1]- y)), 3, (0,0,255), -1)
    plt.figure(figsize=(6,5))
    plt.imshow(vis[..., ::-1])
    plt.title(f"Detected {len(centers)} blob centers")
    plt.show()



    best_rows,best_cols,_,__ = fit_best_grid(centers)
   


    # 2. Input number of columns and rows
    cols = int(input(f"Enter number of circles per row (columns),press ENTER for {best_cols} cols") or best_cols)
    rows = int(input(f"Enter number of circles per column (rows), press ENTER for {best_rows} rows") or best_rows)
   

    res= get_resolution(centers,rows,cols)
    

    # --- Compute residuals in pixels ---
    


    print(f"Optimal spacing: {res.x[0]:.3f} px")
    
    
    scale = distance_between/res.x[0]

    # --- Convert to mm using scale ---

    
    if distance_between is not None:
        print(f"Resolution: {scale:.5f} mm/pix")

    frame_distance_x_mm = x_length_pix * scale 
    print(f"✅ Circle grid detected: {cols} x {rows}")
    # --- Step 3: Save results ---
    if save:
        save_backup(proc_path=folder,
                    file_name="calibration.npz",
                    resolution_mm_per_px=scale,
                    calibration_file=calibration_path,
                    frame_distance_x_mm=frame_distance_x_mm,
                    roi_width_px=w,
                    roi_height_px=h)

    return scale

