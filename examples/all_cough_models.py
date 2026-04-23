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
# from tcm_utils.cvd_check import set_cvd_friendly_colors

colors = use_tcm_poster_style(cvd_friendly=True, dark_mode=False)
# colors = set_cvd_friendly_colors()
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

# Marker styles for different models
markers = ['o', 's', '^', 'v', 'd', 'p', '*', 'h', 'D', 'X']
marker_size = 7

# Keep this hook for stage 1 only (Gupta-only slide).
# Add your future Gupta-specific annotations here.


def _add_stage1_annotations(ax_obj) -> None:
    pass


def _fixed_width_legend(ax_obj, *, width_axes: float = 0.44) -> None:
    """Draw legend in a fixed-width box (axes-fraction units)."""

    ax_obj.legend(
        loc="upper left",
        bbox_to_anchor=(1.0 - width_axes, 1.0, width_axes, 0.0),
        bbox_transform=ax_obj.transAxes,
        mode="expand",
        borderaxespad=0.0,
    )


stage2_examples = ["Results_King", "Results_Knudson", "Results_Mahajan"]
stage3_extra_examples = ["Results_Feinstein", "Results_Ross", "Results_Smith"]
presentation_order = stage2_examples + stage3_extra_examples

style_by_model_name = {
    name: (colors[i % len(colors)], markers[i % len(markers)])
    for i, name in enumerate(presentation_order)
}

# Fixed axis limits for all build-up slides, so overlays stay visually stable.
t_gupta, q_gupta = gupta_model.flow(dt=1e-3, duration_s=0.5)
global_x_max = max(t_gupta[-1], max(model.flow()[0][-1]
                   for model in models.values())) + 0.05
global_y_max = max(float(np.nanmax(q_gupta)), max(
    float(np.nanmax(model.flow(dt=None)[1])) for model in models.values()))

stages = [
    ("all_cough_models1_gupta_only", []),
    ("all_cough_models2_plus_king_knudson_mahajan", stage2_examples),
    (
        "all_cough_models3_plus_feinstein_ross_smith",
        stage2_examples + stage3_extra_examples,
    ),
]

for stage_name, visible_examples in stages:
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5), constrained_layout=True)
    set_grid(ax, mode="horizontal", on=False)

    gupta_label = _legend_label(gupta_model, "Gupta")
    ax.plot(
        t_gupta,
        q_gupta,
        color="black",
        linewidth=2,
        linestyle="--",
        label=gupta_label,
    )

    for example_name in visible_examples:
        model = models.get(example_name)
        if model is None:
            continue

        t_s, q_lps = model.flow(dt=None)
        color, marker = style_by_model_name.get(
            example_name,
            (colors[0], markers[0]),
        )
        label = _legend_label(model, example_name)

        ax.plot(
            t_s,
            q_lps,
            color=color,
            linewidth=2,
            label=label,
            marker=marker,
            markersize=marker_size,
            markeredgewidth=0,
            alpha=0.7,
        )

    if stage_name == "all_cough_models1_gupta_only":
        _add_stage1_annotations(ax)

    ax.set_ylabel("Flow rate (L/s)")
    ax.set_xlim(0, global_x_max)
    ax.set_ylim(0, global_y_max * 1.05)
    _fixed_width_legend(ax)
    set_ticks_every(ax, axis="y", step=1)
    set_ticks_every(ax, axis="x", step=0.1)
    append_unit_to_last_ticklabel(ax, axis="x", unit="s", fmt="{x:.1f}")
    raise_axis_frame(ax)

    out_path = out_dir / f"{stage_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

# Keep legacy filename for convenience (same content as stage 3).
final_path = out_dir / "all_cough_models.pdf"
stage3_path = out_dir / "all_cough_models3_plus_feinstein_ross_smith.pdf"
if stage3_path.exists():
    final_path.write_bytes(stage3_path.read_bytes())
    print(f"Saved: {final_path}")
