"""Plot all cough models (Gupta parametric + recorded examples) in poster style.

- Gupta model: 1.9 m tall, 70 kg male with black dashed line (no markers).
- Example models: loaded from CSV files with cvd_friendly colors and markers.
- 1.5x larger figure size than test_tcm_poster_style.py.

Run:
    python examples/all_cough_models.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tcm_utils.cough_model import CoughModel
from tcm_utils.plot_style import (
    add_label,
    append_unit_to_last_ticklabel,
    raise_axis_frame,
    set_grid,
    use_tcm_poster_style,
    set_ticks_every
)

colors = use_tcm_poster_style(cvd_friendly=True, dark_mode=False)

# Save to output directory
out_dir = Path(__file__).parent / "cough_model_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# ========== Generate/load all models ==========
models = {}


def _legend_label(model: CoughModel, fallback: str) -> str:
    try:
        citation = model.citation()
    except KeyError:
        return fallback
    return citation.get("short") or fallback


# Gupta model: 70 kg / 1.9 m male
gupta_model = CoughModel.from_gupta("Male", weight_kg=70, height_m=1.9)

# Load all example models
available = CoughModel.available_examples()
for example_name in available:
    # Skip Results_Gupta
    if example_name.lower().endswith("gupta"):
        continue
    models[example_name] = CoughModel.from_example(example_name)

# ========== Plot all models ==========
fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5), constrained_layout=True)
set_grid(ax, mode="horizontal", on=False)

# Marker styles for different models
markers = ['o', 's', '^', 'v', 'd', 'p', '*', 'h', 'D', 'X']
marker_size = 7

# Plot example models with cvd_friendly colors and markers
color_idx = 0
for model_name, model in sorted(models.items()):
    t_s, q_lps = model.flow(dt=1e-3)
    color = colors[color_idx % len(colors)]
    marker = markers[color_idx % len(markers)]
    label = _legend_label(model, model_name)

    ax.plot(t_s, q_lps, color=color, linewidth=2, label=label)
    ax.plot(t_s[::10], q_lps[::10], marker=marker, markersize=marker_size,
            color=color, linestyle="none", alpha=0.7)

    color_idx += 1

# Plot Gupta model in black dashed, no markers
t_gupta, q_gupta = gupta_model.flow(dt=1e-3, duration_s=0.5)
gupta_label = _legend_label(gupta_model, "Gupta")
ax.plot(t_gupta, q_gupta, color="black",
        linewidth=2, linestyle="--", label=gupta_label)

ax.set_ylabel("Flow rate (L/s)")
ax.set_xlim(0, max(t_gupta[-1], max([model.flow()[0][-1]
            for model in models.values()])))
ax.set_ylim(0, None)
ax.legend(loc="upper right", fontsize=8)
set_ticks_every(ax, axis="y", step=1)
append_unit_to_last_ticklabel(ax, axis="x", unit="s", fmt="{x:.1f}")

raise_axis_frame(ax)

plt.show()
out_path = out_dir / "all_cough_models.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)
