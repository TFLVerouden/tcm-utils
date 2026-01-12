"""Plot styling helpers.

This module provides a bundled Matplotlib style file ("tcm-poster") intended
to match the poster look: thick black spines, outward ticks, large PT Sans
fonts, white axes background, and transparent figure background.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def data_to_axes_frac(ax, xy: tuple[float, float]) -> tuple[float, float]:
    """Convert (x, y) from data coordinates to axes-fraction coordinates."""

    x_disp, y_disp = ax.transData.transform(xy)
    x_ax, y_ax = ax.transAxes.inverted().transform((x_disp, y_disp))
    return float(x_ax), float(y_ax)


def axes_frac_to_data(ax, xy: tuple[float, float]) -> tuple[float, float]:
    """Convert (x, y) from axes-fraction coordinates to data coordinates."""

    x_disp, y_disp = ax.transAxes.transform(xy)
    x_data, y_data = ax.transData.inverted().transform((x_disp, y_disp))
    return float(x_data), float(y_data)


def use_tcm_poster_style() -> None:
    """Activate the bundled Matplotlib style: "tcm-poster"."""

    import matplotlib.pyplot as plt

    try:
        from importlib.resources import files

        pkg_root = files("tcm_utils")
        style_path = pkg_root.joinpath("styles", "tcm-poster.mplstyle")

        # Ensure Matplotlib sees PT Sans italic/bold faces.
        # On macOS the system TTC sometimes exposes only Regular to Matplotlib,
        # so we ship the TTFs and register them at runtime.
        fonts_dir = pkg_root.joinpath("fonts")
        font_paths = []
        try:
            for name in [
                "PTSans-Regular.ttf",
                "PTSans-Italic.ttf",
                "PTSans-Bold.ttf",
                "PTSans-BoldItalic.ttf",
            ]:
                p = fonts_dir.joinpath(name)
                if p.is_file():
                    font_paths.append(str(p))
        except Exception:
            font_paths = []

        if font_paths:
            register_fonts(font_paths)

        plt.style.use(str(style_path))
    except Exception:
        # Fallback: if package resources aren't available for some reason.
        plt.style.use("default")


def use_flow_rate_model_style() -> None:
    """Backward-compatible alias for older scripts."""

    use_tcm_poster_style()


def add_label(
    ax,
    text: str,
    *,
    xy: tuple[float, float] = (0.02, 0.95),
    coord_system: Literal["axes", "data"] = "axes",
    ha: str = "left",
    va: str = "top",
    italic: bool = True,
    fontsize: float | None = None,
) -> None:
    """Add a text label inside the axes (legend replacement)."""

    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    if fontsize is None:
        # Match tick label size (not axis label size).
        fontsize = float(plt.rcParams.get("xtick.labelsize", 10))

    # Derive font family from the currently active style (rcParams) so callers
    # don't have to specify the font in multiple places.
    sans_list = plt.rcParams.get("font.sans-serif") or []
    family = sans_list[0] if len(sans_list) > 0 else "sans-serif"
    fp = FontProperties(family=family, style="italic" if italic else "normal")

    if coord_system == "axes":
        transform = ax.transAxes
        x, y = xy
    elif coord_system == "data":
        transform = ax.transData
        x, y = xy
    else:
        raise ValueError("coord_system must be 'axes' or 'data'")

    ax.text(
        x,
        y,
        text,
        transform=transform,
        ha=ha,
        va=va,
        fontproperties=fp,
        fontsize=fontsize,
    )


def set_log_axes(ax, *, x: bool = False, y: bool = False) -> None:
    """Convenience to enable log scaling on axes."""

    if x:
        ax.set_xscale("log")
    if y:
        ax.set_yscale("log")


def register_fonts(font_paths: list[str]) -> None:
    """Register font files with Matplotlib at runtime.

    This is useful on macOS when the system-provided TTC collection exposes only
    the Regular face to Matplotlib (e.g. PT Sans), but you want Italic/Bold.

    Example:
        register_fonts(["/path/to/PTSans-Italic.ttf", "/path/to/PTSans-Bold.ttf"])
    """

    from matplotlib import font_manager as fm

    for p in font_paths:
        fm.fontManager.addfont(p)


def set_grid(
    ax,
    *,
    mode: Literal["none", "horizontal", "vertical", "both"] = "horizontal",
    on: bool = True,
    which: str = "major",
    zorder: float = 0,
) -> None:
    """Grid helper with horizontal/vertical/both modes.

    - horizontal => y-gridlines only
    - vertical   => x-gridlines only
    - both       => both axes
    - none       => no grid

    Always forces gridlines to a low z-order.
    """

    ax.set_axisbelow(True)

    if mode == "none" or not on:
        ax.grid(False)
    elif mode == "horizontal":
        ax.grid(True, axis="y", which=which)
        ax.grid(False, axis="x")
    elif mode == "vertical":
        ax.grid(True, axis="x", which=which)
        ax.grid(False, axis="y")
    elif mode == "both":
        ax.grid(True, which=which)
    else:
        raise ValueError(
            "mode must be one of: none, horizontal, vertical, both")

    for gl in ax.get_xgridlines() + ax.get_ygridlines():
        gl.set_zorder(zorder)


def append_unit_to_last_ticklabel(
    ax,
    *,
    axis: Literal["x", "y"] = "x",
    unit: str = "",
    fmt: str = "{x:g}",
    space: bool = True,
) -> None:
    """Append a unit to only the final tick label.

    Note: this snapshots the current ticks; call after setting limits and
    (optional) tick locator.
    """

    from matplotlib.ticker import FixedFormatter, FixedLocator

    def _apply(ticks: list[float], vmin: float, vmax: float, set_locator, set_formatter):
        if not ticks:
            return
        # Append unit to the last tick that is actually visible within limits.
        lo, hi = (vmin, vmax) if vmin <= vmax else (vmax, vmin)
        eps = (hi - lo) * 1e-12 + 1e-12
        visible_idx = [i for i, t in enumerate(
            ticks) if (lo - eps) <= t <= (hi + eps)]
        labels = [fmt.format(x=t) for t in ticks]
        if visible_idx and unit:
            i = visible_idx[-1]
            labels[i] = labels[i] + (" " if space else "") + unit
        set_locator(FixedLocator(ticks))
        set_formatter(FixedFormatter(labels))

    if axis == "x":
        ticks = list(map(float, ax.get_xticks()))
        vmin, vmax = map(float, ax.get_xlim())
        _apply(ticks, vmin, vmax, ax.xaxis.set_major_locator,
               ax.xaxis.set_major_formatter)
        return

    ticks = list(map(float, ax.get_yticks()))
    vmin, vmax = map(float, ax.get_ylim())
    _apply(ticks, vmin, vmax, ax.yaxis.set_major_locator,
           ax.yaxis.set_major_formatter)


def set_ticks_every(ax, *, axis: Literal["x", "y"] = "x", step: float = 1.0) -> None:
    """Set ticks at every 'step' units on the specified axis."""

    from matplotlib.ticker import MultipleLocator

    locator = MultipleLocator(base=step)
    if axis == "x":
        ax.xaxis.set_major_locator(locator)
    else:
        ax.yaxis.set_major_locator(locator)


def raise_axis_frame(ax, *, zorder: float = 20) -> None:
    """Draw frame/ticks above plotted artists without lifting the grid."""

    for spine in ax.spines.values():
        spine.set_zorder(zorder)

    # Raise tick marks (not tick label text).
    for tick in ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks():
        tick.tick1line.set_zorder(zorder + 1)
        tick.tick2line.set_zorder(zorder + 1)
    for tick in ax.yaxis.get_major_ticks() + ax.yaxis.get_minor_ticks():
        tick.tick1line.set_zorder(zorder + 1)
        tick.tick2line.set_zorder(zorder + 1)

    # Re-assert grid at the very bottom (in case something else touched it).
    for gl in ax.get_xgridlines() + ax.get_ygridlines():
        gl.set_zorder(0)


def plot_binned_area(
    ax,
    x,
    heights,
    *,
    baseline: float = 0.0,
    color=None,
    alpha: float = 0.25,
    outline: bool = True,
    outline_color=None,
    outline_linewidth: float | None = None,
    zorder_fill: float = 3,
    zorder_outline: float = 4,
    white_underlay: bool = True,
    x_mode: Literal["edges", "centers"] = "edges",
    edge_method: Literal["auto", "linear", "log"] = "auto",
):
    """Plot a continuous filled area using bin edges (no internal bar edges).

    Uses Matplotlib "stairs" so only the outer edge is drawn.

    Args:
        x_edges: Bin edges, length N+1.
        heights: Bin heights, length N.
    """

    x_arr = np.asarray(x, dtype=float)
    heights_arr = np.asarray(heights, dtype=float)
    if x_arr.ndim != 1 or heights_arr.ndim != 1:
        raise ValueError("x (edges or centers) and heights must be 1D")

    if x_mode == "edges":
        x_edges_arr = x_arr
        if len(x_edges_arr) != len(heights_arr) + 1:
            raise ValueError(
                "When x_mode='edges', x must have length len(heights)+1")
    elif x_mode == "centers":
        centers = x_arr
        if len(centers) != len(heights_arr):
            raise ValueError(
                "When x_mode='centers', x must have length len(heights)")
        if len(centers) < 2:
            raise ValueError("Need at least 2 bin centers")
        if not np.all(np.diff(centers) > 0):
            raise ValueError("bin centers must be strictly increasing")

        method = edge_method
        if method == "auto":
            if np.all(centers > 0):
                ratios = centers[1:] / centers[:-1]
                method = "log" if (np.nanmax(ratios) /
                                   np.nanmin(ratios) < 1.15) else "linear"
            else:
                method = "linear"

        if method == "log":
            inner = np.sqrt(centers[:-1] * centers[1:])
            first = centers[0] ** 2 / inner[0]
            last = centers[-1] ** 2 / inner[-1]
            x_edges_arr = np.concatenate([[first], inner, [last]])
        else:
            inner = 0.5 * (centers[:-1] + centers[1:])
            first = centers[0] - (inner[0] - centers[0])
            last = centers[-1] + (centers[-1] - inner[-1])
            x_edges_arr = np.concatenate([[first], inner, [last]])
    else:
        raise ValueError("x_mode must be 'edges' or 'centers'")

    if outline_color is None:
        outline_color = color

    if outline_linewidth is None:
        # Default to current line width (usually > spine width).
        import matplotlib.pyplot as plt

        outline_linewidth = float(plt.rcParams.get("lines.linewidth", 2.0))

    # Optional underlay to simulate higher-opacity edges while keeping the fill
    # visually "transparent" over a grid.
    if white_underlay:
        ax.stairs(
            heights_arr + baseline,
            x_edges_arr,
            baseline=baseline,
            fill=True,
            facecolor="white",
            edgecolor="none",
            linewidth=0.0,
            zorder=zorder_fill,
        )

    ax.stairs(
        heights_arr + baseline,
        x_edges_arr,
        baseline=baseline,
        fill=True,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        linewidth=0.0,
        zorder=zorder_fill + 0.01,
    )

    if outline:
        # Keep outline below axis spines; raise_axis_frame(ax) can place spines above.
        return ax.stairs(
            heights_arr + baseline,
            x_edges_arr,
            baseline=baseline,
            fill=False,
            color=outline_color,
            linewidth=outline_linewidth,
            zorder=zorder_outline,
        )

    return None


def plot_continuous_bar(*args, **kwargs):
    """Backward-compatible alias; prefer plot_continuous_area."""

    return plot_binned_area(*args, **kwargs)


def plot_continuous_area(*args, **kwargs):
    """Backward-compatible alias; prefer plot_binned_area."""

    return plot_binned_area(*args, **kwargs)
