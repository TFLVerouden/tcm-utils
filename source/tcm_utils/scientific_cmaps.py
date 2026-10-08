"""Matplotlib defaults using scientific colormaps from cmcrameri."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import cmcrameri.cm as cmc

_categorical_colors: tuple[str, ...] = ()


def set_scientific_colormap(name: str = "batlow") -> list[str]:
    """Set the default image colormap and matching categorical color cycle.

    ``name`` must be a cmcrameri colormap with a categorical ``S`` variant.
    The cmcrameri package only provides categorical variants for some maps.
    """
    if not isinstance(name, str):
        raise TypeError("Colormap name must be a string.")
    if name not in cmc.cmaps:
        raise ValueError(f"Unknown cmcrameri colormap: {name!r}.")

    categorical_name = f"{name}S"
    if categorical_name not in cmc.cmaps:
        available = sorted(
            cmap_name[:-1]
            for cmap_name in cmc.cmaps
            if cmap_name.endswith("S")
        )
        raise ValueError(
            f"Colormap {name!r} has no categorical variant. "
            f"Choose a map with an S variant: {', '.join(available)}."
        )

    global _categorical_colors
    _categorical_colors = tuple(
        to_hex(color) for color in cmc.cmaps[categorical_name].colors
    )
    plt.rcParams["image.cmap"] = f"cmc.{name}"
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=_categorical_colors)
    return list(_categorical_colors)


def get_color(index: int) -> str:
    """Return a categorical color by zero-based index, wrapping at the end."""
    return _categorical_colors[index % len(_categorical_colors)]


set_scientific_colormap()
