# tcm-utils
Twente Cough Machine - utilities

How to install local editable version?
pip install -e ../tcm-utils

## TIFF video export

`tcm_utils.video_maker.make_video` converts individually numbered grayscale TIFF
frames to an H.264 MP4. FFmpeg must be installed and available on `PATH`.
Frame ranges refer to the trailing number before `.tif`/`.tiff` and include both
endpoints. By default, the output frame rate is the recording rate multiplied by
`time_stretch_s_per_s` (0.002 by default), and every selected frame is retained.
Before sequence-wide contrast analysis, the default interactive flow previews
the first selected frame with a timestamp and a quick per-frame contrast
stretch. Set `crop_roi=(y_start, y_end, x_start, x_end)` to crop every frame; the
preview, contrast analysis, and encoded video all use the cropped image. Negative
coordinates count from the corresponding image edge, while a zero end coordinate
means the full extent in that direction. The first numbered TIFF in the folder
is labeled 0; later timestamps retain their offset from it.
The timestamp uses the bundled PT Sans font by default. Set `label_font_path`
to a `.ttf` or `.otf` file, `label_font_size_px` to its pixel size, and
`label_color` to `"black"`, `"white"`, `"gray"`/`"grey"`, or an integer from 0
(black) to 255 (white). `label_stroke_color` accepts the same values and
defaults to `None`, which disables the stroke. `label_location` accepts
positions such as `"upper left"` and `"lower right"`, or normalized coordinates
from `(0, 0)` at the upper-left margin to `(1, 1)` at the lower-right margin.
Set `show_scale_bar=True` to draw a scale bar and its centered length under it
on every frame. Pass `scale_bar_calibration_path` as a calibration metadata JSON
or calibration image path; omitting it opens the calibration picker, which can
also run calibration on a selected image. The bar's default length is 5 mm.
Configure its physical length, unit (`"m"`, `"cm"`, `"mm"`, `"um"`/`"µm"`, or
`"nm"`), location, and rectangle height with `scale_bar_length`,
`scale_bar_unit`, `scale_bar_location`, and `scale_bar_height_px`. The scale
label shares the timestamp's font, size, color, and stroke settings.
The video is first encoded into `<repo>/.temp` and then moved to `output_path`;
if `output_path` is omitted, a folder picker asks where to save it afterwards.
For example:

```python
from tcm_utils.video_maker import make_video

make_video(
    frames_dir="/path/to/tiff-frames",
    frames_range=(1, 40),
    output_path="/path/to/export/first-40.mp4",
    label_font_size_px=32,
    label_color=220,
    label_stroke_color="black",
    label_location="upper right",
    crop_roi=(59, -59, 0, 0),
    show_scale_bar=True,
    scale_bar_calibration_path="/path/to/calibration_metadata.json",
    scale_bar_length=5,
    scale_bar_unit="mm",
    scale_bar_location="lower right",
)
```
