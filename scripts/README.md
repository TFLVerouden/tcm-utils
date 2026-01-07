# Scripts

This folder contains small runnable scripts for local demos and utilities.

 `run_cvd_demo.py`: Opens a file dialog to select an image, runs color vision deficiency (DaltonLens) simulation and a monochrome version, and saves comparison plots under `scripts/outputs/`.

## Usage

From the repository root (macOS/Linux):

```bash
python scripts/run_cvd_demo.py
```

If you are using a virtual environment:

```bash
. .venv/bin/activate
python scripts/run_cvd_demo.py
```

The script prepends `src/` to `PYTHONPATH` so it works without installing the package. If you prefer an installed environment, you can also run:

```bash
pip install -e .
 python scripts/run_cvd_demo.py
```
