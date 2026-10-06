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
The video is first encoded into `<repo>/.temp` and then moved to `output_path`;
if `output_path` is omitted, a folder picker asks where to save it afterwards.
For example:

```python
from tcm_utils.video_maker import make_video

make_video(
    frames_dir="/path/to/tiff-frames",
    frames_range=(1, 40),
    output_path="/path/to/export/first-40.mp4",
)
```
