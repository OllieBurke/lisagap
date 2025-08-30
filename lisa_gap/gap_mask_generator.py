import h5py
import json

import numpy as np
from numpy.typing import NDArray
from scipy.stats import expon
from scipy.signal.windows import tukey

from typing import Union, Any, Optional
from collections.abc import Mapping

from pathlib import Path

from lisaconstants import TROPICALYEAR_J2000DAY

# Try to import CuPy for GPU support
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


SECONDS_PER_YEAR = TROPICALYEAR_J2000DAY * 86400
AN_HOUR = 60 * 60


def _get_array_module(use_gpu: bool = False):
    """Get the appropriate array module (numpy or cupy) based on GPU availability and preference."""
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


def _to_numpy(array, xp):
    """Convert array to numpy array if it's a cupy array."""
    if xp is cp and hasattr(array, "get"):
        return array.get()
    return array


def _to_device(array, xp):
    """Convert array to the appropriate device (GPU or CPU)."""
    if xp is cp and not hasattr(array, "get"):
        return xp.asarray(array)
    return array


class GapMaskGenerator:
    """
    A class to generate and manage gap masks for time series data. Original code developed by
    Eleonora Castelli (NASA Goddard) and adapted, packaged and GPU accelerated
    by Ollie Burke (Glasgow).

    Supports both CPU (NumPy) and GPU (CuPy) computation.
    """

    def __init__(
        self,
        sim_t: NDArray[np.float64],
        dt: float,
        gap_definitions: dict[str, dict[str, dict[str, Any]]],
        treat_as_nan: bool = True,
        planseed: int = 11071993,
        unplanseed: int = 16121997,
        use_gpu: bool = False,
    ):
        """
        Parameters
        ----------
        sim_t : np.ndarray
            Array of simulation time values.
        dt : float
            Time step of the simulation.
        gap_definitions : dict
            Dictionary defining planned and unplanned gaps. Each gap entry must include:
            - 'rate_per_year' (float)
            - 'duration_hr' (float)
        treat_as_nan : bool, optional
            If True, gaps are inserted as NaNs. If False, they are inserted as zeros.
        planseed : int
            Seed for planned gap randomization.
        unplanseed : int
            Seed for unplanned gap randomization.
        use_gpu : bool, optional
            If True, use GPU acceleration with CuPy (requires CuPy to be installed).
            If False or CuPy is not available, use CPU with NumPy.
        """

        # Initialize GPU/CPU support
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        if use_gpu and not CUPY_AVAILABLE:
            print("Warning: CuPy not available. Falling back to CPU computation.")

        self.xp = _get_array_module(self.use_gpu)

        # Initialise sampling properties
        self.sim_t = _to_device(sim_t, self.xp)
        self.dt = dt
        self.n_data = len(sim_t)

        # Determine whether or not mask uses nans or zeros
        self.treat_as_nan = treat_as_nan

        # Set seeds
        self.planseed = planseed
        self.unplanseed = unplanseed

        # Read in gap definitions, rates, durations, types.
        self.gap_definitions = {}

        # Quick and simple error checking
        for kind in ("planned", "unplanned"):
            if kind not in gap_definitions:
                raise ValueError(f"Missing '{kind}' section in gap_definitions.")

            self.gap_definitions[kind] = {}

            for name, entry in gap_definitions[kind].items():
                if "rate_per_year" not in entry or "duration_hr" not in entry:
                    raise ValueError(
                        f"Gap '{name}' in '{kind}' must have 'rate_per_year' and 'duration_hr'."
                    )
                if entry["rate_per_year"] < 0 or entry["duration_hr"] < 0:
                    raise ValueError(
                        f"Gap '{name}' in '{kind}' has negative rate or duration."
                    )

                self.gap_definitions[kind][name] = {
                    "rate_per_year": entry["rate_per_year"],
                    "duration_hr": entry["duration_hr"],
                }
        # If the gap definitions pass (not empty, negative etc.)
        # then we can set the gap definitions
        self._update_gap_arrays()

    def _update_gap_arrays(self) -> None:
        """
        Helper function to update the gap arrays based on the gap definitions.
        This function is called during initialization and when gap definitions are updated.
        """

        # Extract gap types for planned/unplanned gaps
        self.planned_labels = list(self.gap_definitions["planned"].keys())
        self.unplanned_labels = list(self.gap_definitions["unplanned"].keys())

        self.planned_rates = (
            self.xp.array(
                [
                    self.gap_definitions["planned"][k]["rate_per_year"]
                    / SECONDS_PER_YEAR
                    for k in self.planned_labels
                ]
            )
            * self.dt
        )
        self.planned_durations = (
            self.xp.array(
                [
                    self.gap_definitions["planned"][k]["duration_hr"] * 3600
                    for k in self.planned_labels
                ]
            )
            / self.dt
        )

        self.unplanned_rates = (
            self.xp.array(
                [
                    self.gap_definitions["unplanned"][k]["rate_per_year"]
                    / SECONDS_PER_YEAR
                    for k in self.unplanned_labels
                ]
            )
            * self.dt
        )
        self.unplanned_durations = (
            self.xp.array(
                [
                    self.gap_definitions["unplanned"][k]["duration_hr"] * 3600
                    for k in self.unplanned_labels
                ]
            )
            / self.dt
        )

    def construct_planned_gap_mask(
        self,
        rate: float,
        gap_length: float,
        seed: Union[int, None] = None,
    ) -> NDArray[np.float64]:
        """
        Construct a planned gap mask with regular spacing and jitter.

        Parameters
        ----------
        rate : float
            Gap rate (in events/s).
        gap_length : float
            Gap length (in samples).
        seed : int or None
            Random seed.

        Returns
        -------
        np.ndarray
            Array with gaps represented as NaNs.
        """
        # Set specific seed for reproducible results
        # Note: CuPy random state is handled separately from NumPy
        if self.use_gpu:
            if seed is not None:
                self.xp.random.seed(seed)
        else:
            np.random.seed(seed)

        mask = self.xp.ones(self.n_data)
        est_num_gaps = int(self.n_data * rate)

        jitter = 0.2 * gap_length * (self.xp.random.rand(est_num_gaps) - 0.5)
        gap_starts = (jitter + (1 / rate) * self.xp.arange(1, est_num_gaps + 1)).astype(
            int
        )
        gap_ends = (gap_starts + gap_length).astype(int)

        # Decide whether or not to treat gaps as NaNs or 0s.
        gap_val = self._gap_value()

        for start, end in zip(gap_starts, gap_ends):
            if end < self.n_data:
                mask[start:end] = gap_val

        return mask

    def construct_unplanned_gap_mask(
        self,
        rate: float,
        gap_length: float,
        seed: Union[int, None] = None,
    ) -> NDArray[np.float64]:
        """
        Construct an unplanned gap mask using an exponential distribution.

        Parameters
        ----------
        rate : float
            Gap rate (in events/s).
        gap_length : float
            Gap length (in samples).
        seed : int or None
            Random seed.

        Returns
        -------
        np.ndarray
            Array with gaps represented as NaNs.
        """
        # Set specific seed for reproducible results
        if self.use_gpu:
            if seed is not None:
                self.xp.random.seed(seed)
        else:
            np.random.seed(seed)

        mask = self.xp.ones(self.n_data)

        est_num_gaps = int(self.n_data * rate)

        # Handle small probability gaps - need to use numpy for scipy compatibility
        if est_num_gaps == 0:
            # Use numpy for random number generation with scipy
            np.random.seed(seed)
            if np.random.rand() < rate * self.n_data:
                est_num_gaps = 1

        if est_num_gaps > 0:
            # Use scipy for exponential distribution (requires numpy)
            np.random.seed(seed)
            start_offsets = expon.rvs(scale=1 / rate, size=est_num_gaps).astype(int)
            gap_starts_cpu = np.cumsum(start_offsets)
            gap_starts_cpu = gap_starts_cpu[gap_starts_cpu + gap_length < self.n_data]
            gap_ends_cpu = (gap_starts_cpu + gap_length).astype(int)

            # Convert to appropriate device
            gap_starts = _to_device(gap_starts_cpu, self.xp)
            gap_ends = _to_device(gap_ends_cpu, self.xp)
        else:
            gap_starts = self.xp.array([], dtype=int)
            gap_ends = self.xp.array([], dtype=int)

        # Decide whether or not to treat gaps as NaNs or 0s.
        gap_val = self._gap_value()

        for start, end in zip(
            _to_numpy(gap_starts, self.xp), _to_numpy(gap_ends, self.xp)
        ):
            mask[start:end] = gap_val

        return mask

    def _gap_value(self) -> float:
        """
        Helper function to determine the value used for gaps in the mask.
        Returns NaN if treat_as_nan is True, otherwise returns 0.0.
        """
        return self.xp.nan if self.treat_as_nan else 0.0

    def generate_mask(
        self,
        include_planned: bool = True,
        include_unplanned: bool = True,
    ) -> NDArray[np.float64]:
        """
        Combine planned and unplanned masks into a final mask.

        Parameters
        ----------
        include_planned : bool
            Include planned gaps.
        include_unplanned : bool
            Include unplanned gaps.

        Returns
        -------
        np.ndarray
            Final gap mask.
        """
        mask = self.xp.ones(self.n_data)

        # Construct planned gap mask
        if include_planned:
            for rate, duration in zip(self.planned_rates, self.planned_durations):
                mask *= self.construct_planned_gap_mask(
                    rate, duration, seed=self.planseed
                )

        # Construct unplanned gap mask
        if include_unplanned:
            for rate, duration in zip(self.unplanned_rates, self.unplanned_durations):
                mask *= self.construct_unplanned_gap_mask(
                    rate, duration, seed=self.unplanseed
                )

        return mask

    def save_to_hdf5(
        self, mask: np.ndarray, filename: str = "gap_mask_data.h5"
    ) -> None:
        """
        Save the gap mask and associated simulation metadata to an HDF5 file.

        Parameters
        ----------
        mask : np.ndarray
            The gap mask array, typically generated using `generate_mask()`.
            Should be of the same length as `sim_t`, and contain either 1s and 0s,
            or 1s and NaNs depending on the `treat_as_nan` setting.

        filename : str, optional
            Path to the HDF5 file to create. Defaults to "gap_mask_data.h5".

        Notes
        -----
        This function stores:
        - The binary or NaN-valued mask under `"gap_mask"`
        - The simulation time array under `"time_array"`
        - Metadata attributes:
            - `"dt"` (time step)
            - `"treat_as_nan"` (boolean mask type flag)
        - Gap configuration details in two groups:
            - `"planned_gaps"`: each with `rate_events_per_year` and `duration_hours`
            - `"unplanned_gaps"`: same structure as planned gaps

        The resulting file can be reloaded using the `from_hdf5()` class method given below.
        """
        # Convert to numpy for HDF5 compatibility
        mask_cpu = _to_numpy(mask, self.xp)
        sim_t_cpu = _to_numpy(self.sim_t, self.xp)

        with h5py.File(filename, "w") as f:
            f.create_dataset("gap_mask", data=mask_cpu, compression="gzip")
            f.create_dataset("time_array", data=sim_t_cpu, compression="gzip")
            f.attrs["dt"] = self.dt
            f.attrs["treat_as_nan"] = self.treat_as_nan
            f.attrs["use_gpu"] = self.use_gpu

            # Save planned gap info
            planned_grp = f.create_group("planned_gaps")
            for label in self.planned_labels:
                grp = planned_grp.create_group(label)
                grp.attrs["rate_events_per_year"] = self.gap_definitions["planned"][
                    label
                ]["rate_per_year"]
                grp.attrs["duration_hours"] = self.gap_definitions["planned"][label][
                    "duration_hr"
                ]

            # Save unplanned gap info
            unplanned_grp = f.create_group("unplanned_gaps")
            for label in self.unplanned_labels:
                grp = unplanned_grp.create_group(label)
                grp.attrs["rate_events_per_year"] = self.gap_definitions["unplanned"][
                    label
                ]["rate_per_year"]
                grp.attrs["duration_hours"] = self.gap_definitions["unplanned"][label][
                    "duration_hr"
                ]

    @classmethod
    def from_hdf5(cls, filename: str) -> "GapMaskGenerator":
        """
        Reconstruct a GapMaskGenerator object from an HDF5 file.
        classmethod, so no need to instantiate the class first.
        This method reads the gap mask, simulation time, and metadata from the file,
        and returns a new instance of GapMaskGenerator.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file.

        Returns
        -------
        GapMaskGenerator
            A new instance reconstructed from the file.
        """

        with h5py.File(filename, "r") as f:
            # Store simulation time, sampling interval, and mask type.
            sim_t = f["time_array"][:]
            dt = float(f.attrs["dt"])
            treat_as_nan = bool(
                f.attrs.get("treat_as_nan", True)
            )  # fallback True if not saved

            # Store information about the planned gaps
            planned_gaps = {}
            for key in f["planned_gaps"]:
                grp = f["planned_gaps"][key]
                planned_gaps[key] = {
                    "rate_per_year": float(grp.attrs["rate_events_per_year"]),
                    "duration_hr": float(grp.attrs["duration_hours"]),
                }

            # Store information about the unplanned gaps
            unplanned_gaps = {}
            for key in f["unplanned_gaps"]:
                grp = f["unplanned_gaps"][key]
                unplanned_gaps[key] = {
                    "rate_per_year": float(grp.attrs["rate_events_per_year"]),
                    "duration_hr": float(grp.attrs["duration_hours"]),
                }

            # Check if GPU support was used in saved file
            use_gpu = f.attrs.get("use_gpu", False)

            gap_definitions = {
                "planned": planned_gaps,
                "unplanned": unplanned_gaps,
            }
            # Return as class object
        return cls(
            sim_t=sim_t,
            dt=dt,
            gap_definitions=gap_definitions,
            treat_as_nan=treat_as_nan,
            use_gpu=use_gpu,
        )

    def summary(
        self,
        mask: NDArray[np.float64] = None,
        export_json_path: Union[str, Path, None] = None,
    ) -> dict[str, Any]:
        """
        Return a structured dictionary summarising the gap configuration
        and optionally the content of a specific mask.

        Parameters
        ----------
        mask : np.ndarray, optional
            If provided, calculates duty cycle and number of gaps based on this mask.

        export_json_path : str or Path, optional
            If provided, writes the summary dictionary to a JSON file at the given path.

        Returns
        -------
        dict
            Summary of configuration and optionally mask content.
        """
        # Extract as much information as possible from the gap definitions
        # and the mask if provided.
        summary_dict = {
            "simulation": {
                "dt": self.dt,
                "n_data": self.n_data,
                "duration_sec": self.n_data * self.dt,
                "duration_days": self.n_data * self.dt / 86400,
            },
            "seeds": {
                "planned": self.planseed,
                "unplanned": self.unplanseed,
            },
            "planned_gaps": {},
            "unplanned_gaps": {},
        }

        for kind in ("planned", "unplanned"):
            for name, info in self.gap_definitions[kind].items():
                duration_sec = info["duration_hr"] * 3600
                duration_samples = int(duration_sec / self.dt)
                rate_per_sec = info["rate_per_year"] / SECONDS_PER_YEAR

                summary_dict[f"{kind}_gaps"][name] = {
                    "rate_events_per_year": info["rate_per_year"],
                    "duration_hr": info["duration_hr"],
                    "rate_events_per_sec": rate_per_sec,
                    "duration_sec": duration_sec,
                    "duration_samples": duration_samples,
                }

        # Add dynamic information from mask if provided
        if mask is not None:
            # Convert to numpy for analysis
            mask_cpu = _to_numpy(mask, self.xp)

            is_gap = np.isnan(mask_cpu) if self.treat_as_nan else (mask_cpu == 0)
            duty_cycle = 1.0 - np.sum(is_gap) / len(mask_cpu)

            summary_dict["mask_analysis"] = {
                "duty_cycle_percent": round(100 * duty_cycle, 4),
                "total_gap_samples": int(np.sum(is_gap)),
                "total_gap_hours": round(np.sum(is_gap) * self.dt / 3600, 2),
            }

            # Optional: Count number of contiguous gaps
            gap_count = 0
            in_gap = False
            for val in is_gap:
                if val and not in_gap:
                    gap_count += 1
                    in_gap = True
                elif not val:
                    in_gap = False
            summary_dict["mask_analysis"]["number_of_gap_segments"] = gap_count

        summary_dict["computation"] = {
            "use_gpu": self.use_gpu,
            "gpu_available": CUPY_AVAILABLE,
        }

        if export_json_path is not None:
            with open(export_json_path, "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, indent=4)
        return summary_dict

    def apply_smooth_taper_to_mask(
        self,
        mask: NDArray[np.float64],
        taper_gap_definitions: Union[None, dict[str, dict[str, dict[str, Any]]]] = None,
    ) -> NDArray[np.float64]:
        """
        Apply Tukey taper smoothing to an existing gap mask, using optional taper parameters.
        This is a function that would probably be used in L2A and L2D, once processed via L01.

        This function takes as input a mask and applies smooth tapers to the end of the gaps

        Parameters
        ----------
        mask : np.ndarray
            Original binary or NaN mask (1 = good, 0/NaN = gap).
        taper_gap_definitions : dict, optional
            An override dictionary containing lobe_lengths_hr per gap type.
            If None, no tapering is applied.

        Returns
        -------
        np.ndarray
            A smoothed mask with tapering applied around each gap.
        """

        # Make sure that the tapering function is consistent with the gap definitions
        self._validate_taper_definitions(taper_gap_definitions)

        if taper_gap_definitions is None:
            return mask  # No tapering applied

        # Convert to CPU for processing since this involves complex indexing
        mask_cpu = _to_numpy(mask, self.xp)
        smoothed_mask = np.copy(mask_cpu)

        # If using nans, force to zero.
        gap_indicator = np.isnan(mask_cpu) if self.treat_as_nan else (mask_cpu == 0)

        in_gap = False
        start = None

        for i in range(self.n_data):
            if gap_indicator[i] and not in_gap:
                in_gap = True
                start = i
            elif not gap_indicator[i] and in_gap:
                in_gap = False
                end = i
                gap_length = end - start

                match = self._match_gap_length_to_label(gap_length)
                if match is None:
                    continue

                kind, label = match
                taper_info = taper_gap_definitions.get(kind, {}).get(label, {})
                lobe_hr = taper_info.get("lobe_lengths_hr", 0.0)
                lobe_samples = int((lobe_hr * 3600) / self.dt)

                if lobe_samples == 0:
                    continue

                win_start = max(0, start - lobe_samples)
                win_end = min(self.n_data, end + lobe_samples)
                win_len = win_end - win_start

                alpha = 2 * lobe_samples / win_len
                taper = 1 - tukey(win_len, alpha)

                smoothed_mask[win_start:win_end] = np.minimum(
                    smoothed_mask[win_start:win_end], taper
                )

        # Convert back to appropriate device
        return _to_device(smoothed_mask, self.xp)

    def build_quality_flags(
        self, data_array: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Build a masking function based on the gap definitions and the provided data array.

        Parameters
        ----------
        data_array : np.ndarray
            The data array to be masked.

        Returns
        -------
        np.ndarray
            A masking function that can be applied to the data array.
        """
        # Convert to CPU for this simple operation
        data_cpu = _to_numpy(data_array, self.xp)

        # Wherever nans appear, replace this with 1.0.
        # Else 0.0 for valid data product.
        mask_cpu = np.where(np.isnan(data_cpu), 1.0, 0.0)

        # Convert back to appropriate device
        return _to_device(mask_cpu, self.xp)

    def _match_gap_length_to_label(
        self, gap_length_samples: int
    ) -> Union[tuple[str, str], None]:
        """
        Match a gap length to one of the known planned/unplanned definitions.

        Parameters
        ----------
        gap_length_samples : int
            Length of the gap in samples.

        Returns
        -------
        tuple of (kind, label) or None
        """
        for kind in ("planned", "unplanned"):
            for label in self.gap_definitions[kind]:
                expected = int(
                    self.gap_definitions[kind][label]["duration_hr"] * 3600 / self.dt
                )
                if (
                    abs(gap_length_samples - expected) < 0.1 * expected
                ):  # allow 10% fuzz
                    return (kind, label)
        return None

    def _validate_taper_definitions(
        self, taper_gap_definitions: dict[str, dict[str, dict[str, Any]]]
    ) -> None:
        """
        Validate the structure and contents of taper_gap_definitions.

        Raises
        ------
        TypeError, ValueError
            If any structural or semantic errors are detected.
        """
        if not isinstance(taper_gap_definitions, Mapping):
            raise TypeError("taper_gap_definitions must be a dictionary.")

        for kind in taper_gap_definitions:
            if kind not in ("planned", "unplanned"):
                raise ValueError(
                    f"Invalid gap kind '{kind}'. Must be 'planned' or 'unplanned'."
                )

            if kind not in self.gap_definitions:
                raise ValueError(f"'{kind}' gap kind not found in gap_definitions.")

            for label, entry in taper_gap_definitions[kind].items():
                if label not in self.gap_definitions[kind]:
                    raise ValueError(
                        f"Gap label '{label}' not found in gap_definitions[{kind}]."
                    )

                if not isinstance(entry, Mapping) or "lobe_lengths_hr" not in entry:
                    raise ValueError(
                        f"Missing or invalid 'lobe_lengths_hr' for '{label}' in '{kind}'."
                    )

                if (
                    not isinstance(entry["lobe_lengths_hr"], (int, float))
                    or entry["lobe_lengths_hr"] < 0
                ):
                    raise ValueError(
                        f"'lobe_lengths_hr' must be a non-negative number for '{label}'."
                    )
