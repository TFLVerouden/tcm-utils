from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def timestamp_str(
    dt_value: datetime | None = None,
    include_us: bool = False,
) -> str:
    """Return a timestamp string for filenames.

    Defaults to local current time in ``YYMMDD_HHMMSS`` format.
    Optionally pass ``dt_value`` to format a specific datetime and set
    ``include_us=True`` to append ``_ffffff``.
    """
    dt = datetime.now() if dt_value is None else dt_value
    fmt = "%y%m%d_%H%M%S_%f" if include_us else "%y%m%d_%H%M%S"
    return dt.strftime(fmt)


def timestamp_from_file(path: str | os.PathLike, prefer_creation: bool = True) -> str:
    """Return timestamp string (YYMMDD_HHMMSS) based on file times.

    On platforms that support birth time (e.g., macOS), prefer it when
    prefer_creation=True; otherwise fall back to modification time.
    """
    p = Path(path)
    st = p.stat()
    dt: datetime
    if prefer_creation and hasattr(st, "st_birthtime"):
        dt = datetime.fromtimestamp(st.st_birthtime)
    else:
        dt = datetime.fromtimestamp(st.st_mtime)
    return dt.strftime("%y%m%d_%H%M%S")
