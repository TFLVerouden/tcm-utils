"""Quick demo/test for the flow_rate_model Matplotlib style.

- Uses the bundled style (thick black spines, outward ticks, PT Sans).
- Plots the Gupta cough model for a 70 kg, 1.93 m male.
- Demonstrates replacing a legend with italic in-plot text.
- Includes a small continuous-bar demo subplot.

Run:
    python examples/test_flow_rate_model_style.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tcm_utils.cough_model import CoughModel
from tcm_utils.cvd_check import set_cvd_friendly_colors
from tcm_utils.plot_style import (
    add_label,
    append_unit_to_last_ticklabel,
    plot_binned_area,
    raise_axis_frame,
    set_log_axes,
    set_grid,
    use_tcm_poster_style,
    set_ticks_every
)

use_tcm_poster_style()
colors = set_cvd_friendly_colors()

# --- Model: 70 kg / 1.93 m male ---
model = CoughModel.from_gupta("Male", weight_kg=70, height_m=1.93)
t_s, q_lps = model.flow(dt=1e-3, duration_s=0.5)

# Save next to other example outputs
out_dir = Path(__file__).parent / "cough_model_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# ========== Demo 1: cough model (linear axes) ==========
fig1, ax1 = plt.subplots(1, 1, figsize=(4.0, 3.0), constrained_layout=True)
set_grid(ax1, mode="horizontal", on=False)
ax1.plot(t_s, q_lps, color=colors[0])
ax1.set_ylabel("Flow rate (L/s)")
ax1.set_xlim(0, 0.5)
ax1.set_ylim(0, max(1.0, float(np.nanmax(q_lps)) * 1.05))
add_label(ax1, "real cough", xy=(0.27, 3), coord_system="data")
set_ticks_every(ax1, axis="y", step=1)
append_unit_to_last_ticklabel(ax1, axis="x", unit="s", fmt="{x:.1f}")

raise_axis_frame(ax1)

out_path1 = out_dir / "tcm_poster_demo_cough.pdf"
fig1.savefig(out_path1, bbox_inches="tight")
print(f"Saved: {out_path1}")
plt.close(fig1)

# ========== Demo 2: binned area (log-x) ==========
fig2, ax2 = plt.subplots(1, 1, figsize=(4.0, 3.0), constrained_layout=True)

# Provide BIN CENTERS directly
bin_centers = np.geomspace(0.2, 200, 40)  # e.g. droplet diameter bins (µm)
heights = np.exp(-((np.log10(bin_centers) - np.log10(20))
                 ** 2) / (2 * 0.22**2))
heights = heights / heights.max()

# Bars in front of grid: grid below, fill above.
set_grid(ax2, mode="horizontal", on=True)
set_log_axes(ax2, x=True)

plot_binned_area(
    ax2,
    bin_centers,
    heights,
    x_mode="centers",
    edge_method="log",
    color=colors[1],
    alpha=0.22,
    outline=True,
    outline_color=colors[1],
    white_underlay=True,
    zorder_fill=6,
    zorder_outline=7,
)

ax2.set_ylabel("Normalized volume")
append_unit_to_last_ticklabel(ax2, axis="x", unit="µm", fmt="{x:.0f}")
add_label(ax2, "example", xy=(0.52, 0.5), ha="right", va="bottom")
raise_axis_frame(ax2)

out_path2 = out_dir / "tcm_poster_demo_binned_area.pdf"
fig2.savefig(out_path2, bbox_inches="tight")
print(f"Saved: {out_path2}")
plt.close(fig2)
