"""Plot cough models - both Gupta model and example coughs from literature.

Generates comparison plots saved to examples/cough_model_outputs/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tcm_utils.cough_model import CoughModel
from tcm_utils.cvd_check import set_cvd_friendly_colors


# Output directory
output_dir = Path(__file__).parent.parent / \
    "examples" / "cough_model_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Setup CVD-friendly colors
colors = set_cvd_friendly_colors()

# Get available example coughs
examples = CoughModel.available_examples()
print(f"Found {len(examples)} example coughs:")
for ex in examples:
    print(f"  - {ex}")

# ========== Plot 1: All example coughs with Gupta reference ==========
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Flow rate (L/s)")
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 12)
ax.grid(which="both", alpha=0.3)

# Plot each example cough (use native time grid)
for i, example_name in enumerate(sorted(examples)):
    # Determine line style and color
    if i < 6:
        linestyle = "-"
    else:
        linestyle = "--"
    color = colors[i % len(colors)]

    # Load and plot example (dt=None uses native grid)
    model = CoughModel.from_example(example_name)
    time_s, flow_lps = model.flow(dt=None)

    # Create friendly label (remove "Results_" prefix)
    label = example_name.replace("Results_", "").replace("_r", "")

    ax.plot(time_s, flow_lps, label=label,
            linestyle=linestyle, color=color, linewidth=1.5)

# Add Gupta model for comparison (typical adult male)
gupta_model = CoughModel.from_gupta("Male", weight_kg=70, height_m=1.90)
time_gupta, flow_gupta = gupta_model.flow(dt=1e-3, duration_s=0.5)
ax.plot(time_gupta, flow_gupta, label="Gupta model (70kg, 1.90m)",
        linestyle=":", color="black", linewidth=2.5, alpha=0.7)

# Add legend outside plot
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=True)

plt.tight_layout()

# Save figure as PDF only
output_path = output_dir / "all_cough_models.pdf"
fig.savefig(output_path, bbox_inches="tight")
print(f"\nSaved plot to: {output_path}")

plt.close()

# ========== Plot 2: Gupta model - five explicit parameter examples ==========
fig2, (ax2_lin, ax2_norm) = plt.subplots(1, 2, figsize=(14, 5))

# Linear plot
ax2_lin.set_xlabel("Time (s)")
ax2_lin.set_ylabel("Flow rate (L/s)")
ax2_lin.set_xlim(0, 0.5)
ax2_lin.set_ylim(0, 10)
ax2_lin.grid(which="both", alpha=0.3)
ax2_lin.set_title("Gupta 2009 Model - Dimensional")

# Normalized plot
ax2_norm.set_xlabel("Non-dimensional time (τ = t/PVT)")
ax2_norm.set_ylabel("Non-dimensional flow (M = Q/CPFR)")
ax2_norm.set_xlim(0, 3)
ax2_norm.set_ylim(-0.2, 1.2)
ax2_norm.grid(which="both", alpha=0.3)
ax2_norm.set_title("Gupta 2009 Model - Non-dimensional")

# Literature bounds (interpreting PVT ranges as milliseconds)
male_lower = dict(cpfr=3.0, cev=0.4, pvt_s=0.057)
male_upper = dict(cpfr=8.5, cev=1.6, pvt_s=0.096)
female_lower = dict(cpfr=1.6, cev=0.25, pvt_s=0.057)
female_upper = dict(cpfr=6.0, cev=1.25, pvt_s=0.110)

examples_params = [
    ("Male lower", male_lower, "steelblue", "-"),
    ("Male upper", male_upper, "steelblue", "--"),
    ("Female lower", female_lower, "salmon", "-"),
    ("Female upper", female_upper, "salmon", "--"),
]

# Plot four bound examples
for label, params, color, linestyle in examples_params:
    model = CoughModel.from_gupta(
        pvt_s=params["pvt_s"], cpfr_lps=params["cpfr"], cev_l=params["cev"], label=label
    )
    t_s, q_lps = model.flow(dt=1e-3, duration_s=0.5)
    tau, m = model.flow(dt=1e-3, duration_s=0.5, units="normalized")

    ax2_lin.plot(t_s, q_lps, label=label, linestyle=linestyle,
                 color=color, linewidth=2)
    ax2_norm.plot(tau, m, label=label, linestyle=linestyle,
                  color=color, linewidth=2)

# Test subject: 70 kg, 1.93 m male (uses estimator)
test_model = CoughModel.from_gupta("Male", weight_kg=70, height_m=1.93)
t_test, q_test = test_model.flow(dt=1e-3, duration_s=0.5)
tau_test, m_test = test_model.flow(dt=1e-3, duration_s=0.5, units="normalized")

ax2_lin.plot(t_test, q_test, label="Test: 70kg, 1.93m Male",
             linestyle=":", color="black", linewidth=2.5)
ax2_norm.plot(tau_test, m_test, label="Test: 70kg, 1.93m Male",
              linestyle=":", color="black", linewidth=2.5)

ax2_lin.legend(loc="best", frameon=True, fontsize=9)
ax2_norm.legend(loc="best", frameon=True, fontsize=9)

plt.tight_layout()

output_path2 = output_dir / "gupta_parameter_variations.pdf"
fig2.savefig(output_path2, bbox_inches="tight")
print(f"Saved plot to: {output_path2}")

plt.close()

print("\nAll plots generated successfully!")
