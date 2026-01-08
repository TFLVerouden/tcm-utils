from __future__ import annotations

from pathlib import Path
import sys

from tcm_utils.camera_calibration import run_calibration

run_calibration(distance_mm=1, output_dir=Path(
    __file__).parent / "calibration_demo")
