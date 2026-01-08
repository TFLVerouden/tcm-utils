from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# Ensure local src/ is on the path when running from the repo
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcm_utils.file_dialogs import ask_open_file, find_repo_root
from tcm_utils.cvd_check import (
    simulate_cvd_on_file,
    set_cvd_friendly_colors,
)


def plot_color_cycle(output_path: Path, title: str = "Colors in the property cycle") -> None:
    def f(xi, a):
        return 0.85 * a * (1 / (1 + np.exp(-xi)) + 0.2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.set_title(title)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    x = np.linspace(-4, 4, 200)

    tableau_colors = {color: name for name, color in TABLEAU_COLORS.items()}
    matched_colors = [tableau_colors.get(color, "unnamed") for color in colors]

    num_colors = len(colors)
    y_positions = np.linspace(num_colors / 2, -num_colors / 2, num_colors)

    for i, (color, color_name, pos) in enumerate(zip(colors, matched_colors, y_positions)):
        ax.plot(x, f(x, pos), color=color)
        ax.text(4.2, pos, f"'C{i}': '{color_name}'", color=color, va="center")
        ax.bar(9, 1, width=2, bottom=pos - 0.5, color=color)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    repo_root = find_repo_root(REPO_ROOT)
    output_dir = repo_root / "examples" / "cvd_demo_outputs"

    selected = ask_open_file(
        key="cvd_source_image",
        title="Select image for CVD simulation",
        filetypes=(("Image files", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")),
        default_dir=repo_root,
        start=repo_root,
    )

    if not selected:
        print("No file selected; exiting.")
    else:
        selected_path = Path(selected).expanduser().resolve()
        print(f"Selected image: {selected_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        cvd_image = simulate_cvd_on_file(selected_path, output_dir=output_dir)
        monochrome_image = simulate_cvd_on_file(selected_path,
                                                deficiency="MONOCHROME",
                                                suffix="_mono",
                                                output_dir=output_dir)
        print(f"Saved: {cvd_image}")
        print(f"Saved: {monochrome_image}")

        color_cycle_image = output_dir / "color_cycle.png"
        plot_color_cycle(color_cycle_image, title="Colors in the default property cycle")

        adjusted_image = output_dir / "color_cycles" / "removed_colors.png"
        set_cvd_friendly_colors(do_print=True)
        plot_color_cycle(adjusted_image, title="Colors in the adjusted property cycle")

        adjusted_image2 = output_dir / "color_cycles" / "tableau-colorblind10.png"
        set_cvd_friendly_colors(style="tableau-colorblind10", do_print=True)
        plot_color_cycle(adjusted_image2, title="Colors in the tableau_colorblind10 property cycle")

        set_cvd_friendly_colors(do_reset=True, do_print=True)
