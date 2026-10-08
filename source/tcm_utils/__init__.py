"""Twente Cough Machine utilities package."""

__version__ = "0.1.0"

from . import camera_calibration
from . import cough_model
from . import cvd_check
from . import io_utils
from . import plot_style
from . import tif_utils
from . import time_utils
from . import video_maker

__all__ = [
    "camera_calibration",
    "cough_model",
    "cvd_check",
    "io_utils",
    "plot_style",
    "scientific_cmaps",
    "tif_utils",
    "time_utils",
    "video_maker",
]
