from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image


def _is_12bit_like(image: np.ndarray) -> bool:
    """Return True when the image appears to use a 12-bit value range."""
    if image.dtype not in (np.uint16, np.int16):
        return False
    if image.size == 0:
        return False
    max_value = int(np.max(image))
    return max_value <= 4095


def get_tiff_bits_per_sample(image_path: Path) -> int | None:
    """Return TIFF bits-per-sample from metadata, when available."""
    src_path = Path(image_path).expanduser().resolve()
    if src_path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    try:
        with tifffile.TiffFile(src_path) as tif:
            page0: Any = tif.pages[0]
            bps = getattr(page0, "bitspersample", None)
            if bps is None:
                tags = getattr(page0, "tags", None)
                if tags is not None:
                    bits_tag = tags.get("BitsPerSample")
                    bps = getattr(bits_tag, "value", None)
            if isinstance(bps, tuple):
                return int(bps[0]) if bps else None
            return int(bps) if bps is not None else None
    except Exception:
        pass

    # Pillow metadata fallback for TIFF tag 258 (BitsPerSample).
    try:
        with Image.open(src_path) as pil_img:
            tags: Any = getattr(pil_img, "tag_v2", None)
            bps = tags.get(258) if tags is not None else None
            if isinstance(bps, tuple):
                return int(bps[0]) if bps else None
            return int(bps) if bps is not None else None
    except Exception:
        return None


def _read_tiff_with_fallback(src_path: Path) -> np.ndarray | None:
    """Read TIFF image with tifffile first, then Pillow as fallback.

    This function stays quiet when fallback succeeds and only reports when
    all decoding options fail.
    """
    tifffile_error: Exception | None = None
    try:
        return tifffile.imread(src_path)
    except Exception as exc:
        tifffile_error = exc

    pillow_error: Exception | None = None
    try:
        with Image.open(src_path) as pil_img:
            return np.array(pil_img)
    except Exception as exc:
        pillow_error = exc

    print(f"Failed to decode TIFF file: {src_path}")
    if tifffile_error is not None:
        print(f"tifffile error: {tifffile_error}")
    if pillow_error is not None:
        print(f"Pillow error: {pillow_error}")
    if tifffile_error is not None and "imagecodecs" in str(tifffile_error).lower():
        print(
            "Tip: install 'imagecodecs' for full TIFF support: "
            "python -m pip install imagecodecs"
        )
    return None


def convert_12bit_tiff_to_16bit(
    image_path: Path,
) -> Path | None:
    """Optionally convert a 12-bit TIFF to 16-bit and return the new file path.

    The converted file is written in the same directory with a ``_16bit`` suffix.
    Returns None when conversion is not applicable or fails.
    """
    src_path = Path(image_path).expanduser().resolve()

    if src_path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    image = _read_tiff_with_fallback(src_path)
    if image is None:
        return None

    if not _is_12bit_like(image):
        return None

    converted = (image.astype(np.uint16) << 4)
    dst_path = src_path.with_name(f"{src_path.stem}_16bit{src_path.suffix}")

    try:
        tifffile.imwrite(dst_path, converted)
    except Exception as exc:
        print(f"Failed to save converted TIFF: {exc}")
        return None

    print(f"Saved converted 16-bit TIFF: {dst_path}")
    return dst_path
