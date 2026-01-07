from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def timestamp_str() -> str:
    """Return a standard timestamp string (local time) for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_from_file(path: str | os.PathLike, prefer_creation: bool = True) -> str:
    """Return timestamp string based on file times.

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
    return dt.strftime("%Y%m%d_%H%M%S")
