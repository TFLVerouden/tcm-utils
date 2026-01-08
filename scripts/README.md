# Scripts

This folder contains small runnable scripts for local demos and utilities.

- **`run_cvd_demo.py`**: Opens a file dialog to select an image, runs color vision deficiency (DaltonLens) simulation and a monochrome version, and saves comparison plots under `examples/cvd_demo_outputs/`.

- **`run_cihx_demo.py`**: Demonstrates the CIHX metadata extraction functionality. Opens a file dialog to select a .cihx file, extracts embedded XML metadata, saves it as JSON with timestamps, and moves the raw file to an organized subfolder under `examples/read_cihx_outputs/`.

- **`plot_cough_models.py`**: Plots cough flow-rate models and recorded examples.
	- Example coughs are loaded from preprocessed CSVs in `src/tcm_utils/data/Results_*.csv`.
	- Outputs comparison figures as PDFs to `examples/cough_model_outputs/`.
	- Includes a Gupta 2009 model section with five explicit parameter cases:
		- Male lower (CPFR=3.0 L/s, CEV=0.4 L, PVT=57 ms)
		- Male upper (CPFR=8.5 L/s, CEV=1.6 L, PVT=96 ms)
		- Female lower (CPFR=1.6 L/s, CEV=0.25 L, PVT=57 ms)
		- Female upper (CPFR=6.0 L/s, CEV=1.25 L, PVT=110 ms)
		- Test subject (Male, 70 kg, 1.93 m) using estimator for comparison.

## Usage

From the repository root (macOS/Linux):

```bash
# Run CVD demo
python scripts/run_cvd_demo.py

# Run CIHX metadata extraction demo
python scripts/run_cihx_demo.py

# Plot cough models
python scripts/plot_cough_models.py
```

If you are using a virtual environment:

```bash
. .venv/bin/activate
python scripts/run_cvd_demo.py
# or
python scripts/run_cihx_demo.py
# or
python scripts/plot_cough_models.py
```

The script prepends `src/` to `PYTHONPATH` so it works without installing the package. If you prefer an installed environment, you can also run:

```bash
pip install -e .
 python scripts/run_cvd_demo.py
 # or
 python scripts/plot_cough_models.py
```
