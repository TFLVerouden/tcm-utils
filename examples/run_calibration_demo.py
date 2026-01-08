from __future__ import annotations
from tcm_utils.camera_calibration import run_calibration

import sys
from pathlib import Path

# Ensure local src/ is on the path when running from the repo
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    input_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    sys.exit(run_calibration(input_path=input_path))
