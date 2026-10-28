from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt

from tcm_utils.video_analysis import EllipseSizer


video_path = Path(
    "/Users/tommieverouden/Library/CloudStorage/OneDrive-UniversityofTwente/"
    "Presentations/Nice videos/Dripping droplets/"
    "timing_test_26ga_59500us_0-3mlpmn_moveddetectioncloser.mov"
)
capture = cv.VideoCapture(str(video_path))
if not capture.isOpened():
    raise OSError(f"Could not open video: {video_path}")

recording_frame_rate = capture.get(cv.CAP_PROP_FPS)
if recording_frame_rate <= 0:
    capture.release()
    raise ValueError(f"Video has an invalid frame rate: {recording_frame_rate}")

ellipse_sizer = EllipseSizer(
    polarity="dark",
    roi=(20, 350, 560, 680),
)
ellipse_sizer.set_video_context(
    recording_frame_rate=recording_frame_rate,
    # Pixel axes are used; EllipseSizer requires a positive scale regardless.
    scale_m_per_px=1.0,
    first_frame_number=0,
)

frame_number = 0
while True:
    success, frame = capture.read()
    if not success:
        break

    grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ellipse_sizer.process_frame(
        grayscale_frame,
        frame_number,
        is_preview=False,
    )
    frame_number += 1

capture.release()
if frame_number == 0:
    raise RuntimeError(f"No readable frames found in video: {video_path}")

measurements = ellipse_sizer.measurements
experiment_numbers = [
    measurement.frame_number + 1 for measurement in measurements
]
major_axes_px = [
    2 * measurement.radius_major_px
    if measurement.radius_major_px is not None
    else float("nan")
    for measurement in measurements
]
minor_axes_px = [
    2 * measurement.radius_minor_px
    if measurement.radius_minor_px is not None
    else float("nan")
    for measurement in measurements
]

figure, axis = plt.subplots()
axis.plot(experiment_numbers, major_axes_px, marker=".", label="Major axis")
axis.plot(experiment_numbers, minor_axes_px, marker=".", label="Minor axis")
axis.set_xlabel("Experiment number (video frame)")
axis.set_ylabel("Ellipse axis length (px)")
axis.set_title("Ellipse size by experiment")
axis.grid(True)
axis.legend()
figure.tight_layout()
plt.show()
