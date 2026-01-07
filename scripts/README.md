# Scripts

This folder contains small runnable scripts for local demos and utilities.

- **`run_cvd_demo.py`**: Opens a file dialog to select an image, runs color vision deficiency (DaltonLens) simulation and a monochrome version, and saves comparison plots under `examples/cvd_demo_outputs/`.

- **`run_cihx_demo.py`**: Demonstrates the CIHX metadata extraction functionality. Opens a file dialog to select a .cihx file, extracts embedded XML metadata, saves it as JSON with timestamps, and moves the raw file to an organized subfolder under `examples/read_cihx_outputs/`.

## Usage

From the repository root (macOS/Linux):

```bash
# Run CVD demo
python scripts/run_cvd_demo.py

# Run CIHX metadata extraction demo
python scripts/run_cihx_demo.py
```

If you are using a virtual environment:

```bash
. .venv/bin/activate
python scripts/run_cvd_demo.py
# or
python scripts/run_cihx_demo.py
```

The script prepends `src/` to `PYTHONPATH` so it works without installing the package. If you prefer an installed environment, you can also run:

```bash
pip install -e .
 python scripts/run_cvd_demo.py
```
