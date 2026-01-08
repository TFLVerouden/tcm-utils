# Examples and Demos

Run these utilities from the repository root (after activating your virtual environment if needed):

```bash
python examples/run_cvd_demo.py
python examples/run_cihx_demo.py
python examples/plot_cough_models.py
python examples/run_calibration_demo.py
```

- **run_cvd_demo.py**: Choose an image, simulate color-vision-deficiency variations, and save outputs under `examples/cvd_demo_outputs/`.
- **run_cihx_demo.py**: Extract CIHX metadata to JSON, copy the raw file into `examples/cihx_demo_outputs/raw_data/`, and save metadata alongside.
- **plot_cough_models.py**: Generate comparison plots for recorded coughs and Gupta 2009 model variants into `examples/cough_model_outputs/`.
- **run_calibration_demo.py**: Launch the circle-grid camera calibration workflow, producing plot/CSV/JSON outputs under `docs/calibration/` by default.

Each script prepends `src/` to `PYTHONPATH`, so editable installs (`pip install -e .`) are optional.
