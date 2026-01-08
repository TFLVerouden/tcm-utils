"""Cough modelling utilities.

Provides a class to generate cough flow/velocity series from the
Gupta 2009 model and to load recorded example coughs stored in the repository.
Also exposes helpers to convert between flow rate and velocity fields for
velocimetry workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gamma as _gamma
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

# Path to example cough datasets (CSV) shipped in this repository.
_DEFAULT_EXAMPLE_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class GuptaParams:
    """Parameter set for the Gupta 2009 cough model."""

    gender: str
    weight_kg: float
    height_m: float
    pvt_s: float
    cpfr_lps: float
    cev_l: float


def estimate_gupta_parameters(gender: str, weight_kg: float, height_m: float) -> GuptaParams:
    """Estimate Gupta 2009 parameters from subject characteristics.

    Args:
        gender: "Male" or "Female" (case-insensitive).
        weight_kg: Body mass in kg.
        height_m: Body height in meters.

    Returns:
        GuptaParams with peak velocity time (s), cough peak flow rate (L/s),
        and cough expired volume (L).
    """

    gender_norm = gender.lower()
    if gender_norm == "male":
        cpfr = -8.890 + 6.3952 * height_m + 0.0346 * weight_kg
        cev = 0.138 * cpfr + 0.2983
        pvt = (1.360 * cpfr + 65.860) * 1e-3
    elif gender_norm == "female":
        cpfr = -3.9702 + 4.6265 * height_m
        cev = 0.0204 * cpfr - 0.043
        pvt = (3.152 * cpfr + 64.631) * 1e-3
    else:
        raise ValueError("gender must be 'Male' or 'Female'")

    return GuptaParams(gender=gender, weight_kg=weight_kg, height_m=height_m, pvt_s=pvt, cpfr_lps=cpfr, cev_l=cev)


def _gupta_dimensionless_flow(tau: np.ndarray, pvt_s: float, cpfr_lps: float, cev_l: float) -> np.ndarray:
    """Compute the non-dimensional flow m(tau) from Gupta 2009."""

    tau = np.asarray(tau, dtype=float)
    a1 = 1.680
    b1 = 3.338
    c1 = 0.428
    a2 = cev_l / (pvt_s * cpfr_lps) - a1
    b2 = -2.158 * cev_l / (pvt_s * cpfr_lps) + 10.457
    c2 = 1.8 / (b2 - 1)

    m = np.zeros_like(tau)
    mask = tau < 1.2

    m[mask] = a1 * tau[mask] ** (b1 - 1) * \
        np.exp(-tau[mask] / c1) / (_gamma(b1) * c1 ** b1)
    m[~mask] = (
        a1 * tau[~mask] ** (b1 - 1) * np.exp(-tau[~mask] /
                                             c1) / (_gamma(b1) * c1 ** b1)
        + a2
        * (tau[~mask] - 1.2) ** (b2 - 1)
        * np.exp(-(tau[~mask] - 1.2) / c2)
        / (_gamma(b2) * c2**b2)
    )

    return m


def velocity_to_flow(vel: np.ndarray | float, depth_m: float, width_m: float) -> np.ndarray | float:
    """Convert velocity to volumetric flow rate.

    Args:
        vel: Velocity in m/s. Can be a scalar, 1D array, or higher dimensional
            array. For multi-dimensional arrays, averages over all but the first axis.
        depth_m: Out-of-plane depth of the measurement field (m).
        width_m: In-plane width of the full field (m).

    Returns:
        Flow rate(s) in m^3/s with same shape as leading dimension of vel.
    """
    if depth_m <= 0 or width_m <= 0:
        raise ValueError("depth_m and width_m must be positive")

    vel_arr = np.asarray(vel)
    if vel_arr.ndim == 0:
        return float(vel_arr * depth_m * width_m)

    # Average over all dimensions except the first (time/frame dimension)
    if vel_arr.ndim > 1:
        vel_avg = np.nanmean(vel_arr.reshape(vel_arr.shape[0], -1), axis=1)
    else:
        vel_avg = vel_arr

    return vel_avg * depth_m * width_m


def flow_to_velocity(
    flow_rate_m3ps: np.ndarray | float, depth_m: float, width_m: float
) -> np.ndarray | float:
    """Convert volumetric flow rate to velocity.

    Args:
        flow_rate_m3ps: Flow rate in m^3/s (scalar or array).
        depth_m: Out-of-plane depth of the field (m).
        width_m: In-plane width of the field (m).

    Returns:
        Velocity in m/s with same shape as flow_rate_m3ps.
    """
    if depth_m <= 0 or width_m <= 0:
        raise ValueError("depth_m and width_m must be positive")

    flow = np.asarray(flow_rate_m3ps, dtype=float)
    area = depth_m * width_m
    vel = flow / area

    if np.isscalar(flow_rate_m3ps):
        return float(vel)
    return vel


class CoughModel:
    """Cough flow generator supporting modelled and recorded coughs."""

    def __init__(
        self,
        *,
        gupta_params: GuptaParams | None,
        source: str,
        example_name: str | None = None,
        example_time_s: np.ndarray | None = None,
        example_flow_lps: np.ndarray | None = None,
    ) -> None:
        self._gupta_params = gupta_params
        self._source = source
        self._example_name = example_name
        self._example_time_s = example_time_s
        self._example_flow_lps = example_flow_lps

    @classmethod
    def from_gupta(
        cls,
        gender: str | None = None,
        weight_kg: float | None = None,
        height_m: float | None = None,
        *,
        pvt_s: float | None = None,
        cpfr_lps: float | None = None,
        cev_l: float | None = None,
        label: str | None = None,
    ) -> "CoughModel":
        """Build a Gupta 2009 model.

        Two modes:
        1) Explicit parameters: provide pvt_s, cpfr_lps, and cev_l.
        2) Estimation: provide gender, weight_kg, and height_m to estimate parameters.
        """

        if pvt_s is not None and cpfr_lps is not None and cev_l is not None:
            params = GuptaParams(
                gender=label or "custom",
                weight_kg=weight_kg or 0.0,
                height_m=height_m or 0.0,
                pvt_s=pvt_s,
                cpfr_lps=cpfr_lps,
                cev_l=cev_l,
            )
        else:
            if gender is None or weight_kg is None or height_m is None:
                raise ValueError(
                    "Provide either explicit pvt/cpfr/cev or gender, weight_kg, height_m")
            params = estimate_gupta_parameters(gender, weight_kg, height_m)

        return cls(gupta_params=params, source="gupta2009")

    @classmethod
    def from_example(cls, name: str, *, data_dir: Path | None = None) -> "CoughModel":
        """Build a model from a recorded example cough (CSV or NPZ)."""

        time_s, flow_lps, resolved_name = load_example(name, data_dir=data_dir)
        return cls(
            gupta_params=None,
            source="example",
            example_name=resolved_name,
            example_time_s=time_s,
            example_flow_lps=flow_lps,
        )

    @staticmethod
    def available_examples(data_dir: Path | None = None) -> list[str]:
        """List available example cough names (case-insensitive)."""

        paths = _find_example_paths(data_dir)
        return sorted({p.stem for p in paths})

    def flow(
        self,
        *,
        time_s: np.ndarray | None = None,
        dt: float | None = None,
        duration_s: float | None = None,
        start_s: float = 0.0,
        tau_max: float | None = None,
        units: str = "L/s",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a flow rate series.

        Args:
            time_s: Explicit time vector to sample at. If provided, dt/duration
                are ignored.
            dt: Time step for automatic sampling (seconds). If None, uses native
                grid for examples or 1 ms for Gupta model.
            duration_s: Duration when generating the time vector (seconds).
            start_s: Start time for automatic sampling (seconds).
            tau_max: Optional cut-off in non-dimensional time; values beyond are
                returned as NaN (Gupta model only).
            units: "L/s" (default), "m^3/s", or "normalized" (time in PVT units,
                flow in CPFR units, Gupta model only).
        """

        if units not in {"L/s", "m^3/s", "normalized"}:
            raise ValueError("units must be 'L/s', 'm^3/s', or 'normalized'")

        if units == "normalized" and self._source != "gupta2009":
            raise ValueError(
                "Normalized output is only available for Gupta model")

        if self._source == "gupta2009":
            if self._gupta_params is None:
                raise RuntimeError(
                    "Gupta parameters missing for modelled cough")
            p = self._gupta_params
            if time_s is None:
                if dt is None:
                    dt = 1e-3  # default for Gupta
                if duration_s is None:
                    duration_s = 0.2
                time_s = np.arange(start_s, start_s + duration_s + 1e-12, dt)

            if units == "normalized":
                # Return non-dimensional time and flow
                tau = time_s / p.pvt_s
                m = _gupta_dimensionless_flow(
                    tau, p.pvt_s, p.cpfr_lps, p.cev_l)
                if tau_max is not None:
                    m = np.where(tau <= tau_max, m, np.nan)
                return tau, m
            else:
                # Return dimensional flow
                tau = time_s / p.pvt_s
                m = _gupta_dimensionless_flow(
                    tau, p.pvt_s, p.cpfr_lps, p.cev_l)
                if tau_max is not None:
                    m = np.where(tau <= tau_max, m, np.nan)
                flow_lps = m * p.cpfr_lps
        elif self._source == "example":
            if self._example_time_s is None or self._example_flow_lps is None:
                raise RuntimeError("Example data not loaded")
            if time_s is None:
                if dt is None:
                    # Use native time grid for examples
                    time_s = self._example_time_s
                else:
                    if duration_s is None:
                        duration_s = float(
                            self._example_time_s[-1] - self._example_time_s[0])
                    time_s = np.arange(start_s, start_s +
                                       duration_s + 1e-12, dt)
            flow_lps = np.interp(time_s, self._example_time_s,
                                 self._example_flow_lps, left=np.nan, right=np.nan)
        else:
            raise RuntimeError(f"Unknown cough source '{self._source}'")

        flow_out = flow_lps if units == "L/s" else flow_lps / 1000.0
        return time_s, flow_out

    def velocity(
        self,
        depth_m: float,
        width_m: float,
        *,
        time_s: np.ndarray | None = None,
        dt: float | None = None,
        duration_s: float | None = None,
        start_s: float = 0.0,
        tau_max: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | float]:
        """Generate velocity series corresponding to the cough flow.

        Returns uniform velocity (vx) derived from the flow rate.
        """
        time_s, flow_m3ps = self.flow(
            time_s=time_s,
            dt=dt,
            duration_s=duration_s,
            start_s=start_s,
            tau_max=tau_max,
            units="m^3/s",
        )
        vel = flow_to_velocity(flow_m3ps, depth_m=depth_m, width_m=width_m)
        return time_s, vel


def _find_example_paths(data_dir: Path | None = None) -> list[Path]:
    root = data_dir or _DEFAULT_EXAMPLE_DIR
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.suffix.lower() == ".csv" and p.stem.startswith("Results_")]


def _match_example_path(name: str, data_dir: Path | None = None) -> Path:
    targets = _find_example_paths(data_dir)
    if not targets:
        raise FileNotFoundError("No example cough files found")

    name_norm = name.lower().replace(".csv", "").replace(".npz", "")
    for path in targets:
        stem_norm = path.stem.lower()
        stripped = stem_norm.removeprefix("results_")
        if name_norm in {stem_norm, stripped}:
            return path
    raise FileNotFoundError(
        f"Example '{name}' not found in {data_dir or _DEFAULT_EXAMPLE_DIR}")


def load_example(name: str, *, data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Load an example cough by name.

    Args:
        name: File stem or friendly name (case-insensitive). "Results_" prefix
            can be omitted.
        data_dir: Optional override for the example directory.

    Returns:
        time_s (np.ndarray), flow_rate_Lps (np.ndarray), resolved_name (str).
    """
    path = _match_example_path(name, data_dir=data_dir)

    # Load preprocessed CSV with time_s and flow_rate_lps columns
    data = np.genfromtxt(path, delimiter=",", names=True)
    time = data["time_s"].astype(float)
    flow = data["flow_rate_lps"].astype(float)

    return time, flow, path.stem
