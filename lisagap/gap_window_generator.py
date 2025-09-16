"""
GapWindowGenerator - Advanced gap mask processing with tapering capabilities.

This module provides enhanced gap mask functionality built on top of lisaglitch.GapMaskGenerator,
focusing specifically on smooth tapering and window transitions.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import tukey
from typing import Any, Optional, Dict
from collections.abc import Mapping

from lisaglitch import GapMaskGenerator

class GapWindowGenerator:
    """
    Advanced gap mask generator with smooth tapering capabilities.
    
    This class wraps an existing lisaglitch.GapMaskGenerator instance and adds 
    sophisticated tapering and windowing functionality for smooth transitions 
    around gap edges.
    
    Parameters
    ----------
    gap_mask_generator : GapMaskGenerator
        An instantiated GapMaskGenerator object from lisaglitch.
        
    Examples
    --------
    >>> from lisaglitch import GapMaskGenerator
    >>> from lisagap import GapWindowGenerator
    >>> 
    >>> # Create the core gap generator
    >>> gap_gen = GapMaskGenerator(sim_t=sim_t, gap_definitions=gap_defs)
    >>> 
    >>> # Wrap it with windowing capabilities
    >>> window = GapWindowGenerator(gap_gen)
    >>> 
    >>> # Generate masks with optional tapering
    >>> mask = window.generate_mask(apply_tapering=True, taper_definitions=taper_defs)
    """
    
    def __init__(self, gap_mask_generator: GapMaskGenerator):
        """
        Initialize a GapWindowGenerator from an existing GapMaskGenerator.
        
        Parameters
        ----------
        gap_mask_generator : GapMaskGenerator
            An instantiated GapMaskGenerator object from lisaglitch.
        """
        # Store the core generator
        self.gap_mask_generator = gap_mask_generator
        
        # Inherit key attributes from the core generator
        self.sim_t = gap_mask_generator.sim_t
        self.dt = gap_mask_generator.dt
        self.gap_definitions = gap_mask_generator.gap_definitions
        self.treat_as_nan = gap_mask_generator.treat_as_nan
        self.n_data = gap_mask_generator.n_data
        
        # Extract gap labels for tapering validation
        self.planned_labels = list(self.gap_definitions.get("planned", {}).keys())
        self.unplanned_labels = list(self.gap_definitions.get("unplanned", {}).keys())
    
    def generate_window(
        self,
        include_planned: bool = True,
        include_unplanned: bool = True,
        apply_tapering: bool = False,
        taper_definitions: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    ) -> NDArray:
        """
        Generate gap mask with optional tapering.
        
        This method combines gap generation from the underlying GapMaskGenerator
        with optional smooth tapering to create production-ready masks.
        
        Parameters
        ----------
        include_planned : bool, optional
            Include planned gaps in the mask. Default is True.
        include_unplanned : bool, optional
            Include unplanned gaps in the mask. Default is True.
        apply_tapering : bool, optional
            Whether to apply smooth tapering around gaps. Default is False.
        taper_definitions : dict, optional
            Tapering parameters for each gap type. Required if apply_tapering=True.
            Expected structure:
            {
                "planned": {
                    "gap_name": {"lobe_lengths_hr": float}
                },
                "unplanned": {
                    "gap_name": {"lobe_lengths_hr": float}
                }
            }
            
        Returns
        -------
        np.ndarray
            Gap mask with 1.0 for good data and 0.0/NaN for gaps.
            If tapering is applied, values between 0 and 1 indicate
            the tapering transition regions.
            
        Examples
        --------
        >>> # Basic mask generation
        >>> mask = window.generate_mask()
        >>> 
        >>> # Only planned gaps
        >>> planned_mask = window.generate_mask(include_unplanned=False)
        >>> 
        >>> # With tapering
        >>> taper_defs = {
        ...     "planned": {"maintenance": {"lobe_lengths_hr": 2.0}}
        ... }
        >>> tapered_mask = window.generate_mask(
        ...     apply_tapering=True, 
        ...     taper_definitions=taper_defs
        ... )
        """
        # Generate the base mask using the underlying GapMaskGenerator
        mask = self.gap_mask_generator.generate_mask(
            include_planned=include_planned,
            include_unplanned=include_unplanned
        )
        
        # Apply tapering if requested
        if apply_tapering:
            if taper_definitions is None:
                raise ValueError(
                    "taper_definitions must be provided when apply_tapering=True"
                )
            mask = self.apply_smooth_taper_to_mask(mask, taper_definitions)
        
        return mask
    
    def apply_smooth_taper_to_mask(
        self,
        mask: NDArray,
        taper_gap_definitions: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    ):
        """
        Apply Tukey taper smoothing to an existing gap mask.
        
        This function takes as input a mask and applies smooth tapers to the end of the gaps
        using Tukey windows, which helps reduce spectral artifacts in frequency domain analysis.
        
        Parameters
        ----------
        mask : np.ndarray
            Original binary or NaN mask (1 = good, 0/NaN = gap).
        taper_gap_definitions : dict
            Dictionary containing taper parameters per gap type.
            Expected structure:
            {
                "planned": {
                    "gap_name": {"lobe_lengths_hr": float}
                },
                "unplanned": {
                    "gap_name": {"lobe_lengths_hr": float}
                }
            }
            
        Returns
        -------
        np.ndarray
            A smoothed mask with tapering applied around each gap.
        """
        # Validate taper definitions
        self._validate_taper_definitions(taper_gap_definitions)
        
        if taper_gap_definitions is None:
            return mask  # No tapering applied
        
        # Work with a copy to avoid modifying the original
        smoothed_mask = mask.copy().astype(float)
        
        # Find gap regions in the mask
        gap_value = 0.0 if not self.treat_as_nan else np.nan
        
        # Identify gap regions
        if self.treat_as_nan:
            gap_mask = np.isnan(mask)
        else:
            gap_mask = (mask == gap_value)
        
        # Find start and end of gap segments
        gap_diff = np.diff(np.concatenate(([False], gap_mask, [False])).astype(int))
        gap_starts = np.where(gap_diff == 1)[0]
        gap_ends = np.where(gap_diff == -1)[0]
        
        # Apply tapering to each gap
        for start, end in zip(gap_starts, gap_ends):
            # Determine which gap type this corresponds to (simplified approach)
            # In practice, you might want more sophisticated gap type identification
            for kind in ["planned", "unplanned"]:
                labels = self.planned_labels if kind == "planned" else self.unplanned_labels
                
                for label in labels:
                    taper_info = taper_gap_definitions.get(kind, {}).get(label, {})
                    lobe_hr = taper_info.get("lobe_lengths_hr", 0.0)
                    
                    if lobe_hr > 0:
                        # Convert hours to samples
                        lobe_samples = int(lobe_hr * 3600 / self.dt)
                        
                        # Create tapering window
                        win_start = max(0, start - lobe_samples)
                        win_end = min(len(mask), end + lobe_samples)
                        actual_win_len = win_end - win_start
                        
                        if actual_win_len > 2:  # Need at least 3 points for tapering
                            alpha = 2 * lobe_samples / actual_win_len
                            alpha = min(alpha, 1.0)  # Ensure alpha <= 1
                            
                            taper = 1 - tukey(actual_win_len, alpha)
                            smoothed_mask[win_start:win_end] = np.minimum(
                                smoothed_mask[win_start:win_end], taper
                            )
        
        return smoothed_mask
    
    def _validate_taper_definitions(
        self, taper_gap_definitions: Optional[Dict[str, Dict[str, Dict[str, Any]]]]
    ) -> None:
        """
        Validate the structure and contents of taper_gap_definitions.
        
        Parameters
        ----------
        taper_gap_definitions : dict or None
            The taper definitions to validate.
            
        Raises
        ------
        TypeError, ValueError
            If any structural or semantic errors are detected.
        """
        if taper_gap_definitions is None:
            return  # None is valid
            
        if not isinstance(taper_gap_definitions, Mapping):
            raise TypeError("taper_gap_definitions must be a dictionary.")
        
        for kind in taper_gap_definitions:
            if kind not in ("planned", "unplanned"):
                raise ValueError(
                    f"Invalid gap kind '{kind}'. Must be 'planned' or 'unplanned'."
                )
            
            if not isinstance(taper_gap_definitions[kind], Mapping):
                raise TypeError(f"taper_gap_definitions['{kind}'] must be a dictionary.")
            
            # Check that all gap names in taper definitions exist in gap definitions
            available_labels = (
                self.planned_labels if kind == "planned" else self.unplanned_labels
            )
            
            for gap_name in taper_gap_definitions[kind]:
                if gap_name not in available_labels:
                    raise ValueError(
                        f"Gap '{gap_name}' in taper_gap_definitions['{kind}'] "
                        f"not found in gap_definitions. Available: {available_labels}"
                    )
                
                taper_params = taper_gap_definitions[kind][gap_name]
                if not isinstance(taper_params, Mapping):
                    raise TypeError(
                        f"Taper parameters for '{gap_name}' must be a dictionary."
                    )
                
                # Validate required parameters
                if "lobe_lengths_hr" not in taper_params:
                    raise ValueError(
                        f"Missing 'lobe_lengths_hr' in taper parameters for '{gap_name}'."
                    )
                
                lobe_hr = taper_params["lobe_lengths_hr"]
                if not isinstance(lobe_hr, (int, float)) or lobe_hr < 0:
                    raise ValueError(
                        f"'lobe_lengths_hr' for '{gap_name}' must be a non-negative number."
                    )
    
    def summary(self, mask: Optional[NDArray[np.float64]] = None) -> Dict[str, Any]:
        """
        Return a summary of the gap configuration and mask statistics.
        
        Parameters
        ----------
        mask : np.ndarray, optional
            If provided, includes statistics about this specific mask.
            
        Returns
        -------
        dict
            Summary dictionary with configuration and statistics.
        """
        return self.gap_mask_generator.summary(mask)
    
    def save_to_hdf5(self, mask: NDArray, filename: str = 'gap_mask_data.h5', **kwargs):
        """Save a gap mask to an HDF5 file."""
        return self.gap_mask_generator.save_to_hdf5(mask, filename, **kwargs)
    
    def construct_planned_gap_mask(self, **kwargs):
        """Generate only planned gaps mask."""
        return self.gap_mask_generator.construct_planned_gap_mask(**kwargs)
    
    def construct_unplanned_gap_mask(self, **kwargs):
        """Generate only unplanned gaps mask."""
        return self.gap_mask_generator.construct_unplanned_gap_mask(**kwargs)
    
    @property
    def planned_rates(self):
        """Get planned gap rates."""
        return self.gap_mask_generator.planned_rates
    
    @property
    def unplanned_rates(self):
        """Get unplanned gap rates."""
        return self.gap_mask_generator.unplanned_rates
    
    @property
    def planned_durations(self):
        """Get planned gap durations."""
        return self.gap_mask_generator.planned_durations
    
    @property
    def unplanned_durations(self):
        """Get unplanned gap durations.""" 
        return self.gap_mask_generator.unplanned_durations
    
    @staticmethod
    def apply_proportional_tapering(
        mask_data: NDArray,
        dt: float = 1.0,
        short_taper_fraction: float = 0.25,
        medium_taper_fraction: float = 0.05,
        long_taper_fraction: float = 0.05,
        min_gap_points: int = 5,
        short_gap_threshold_minutes: float = 10.0,
        long_gap_threshold_hours: float = 10.0
    ) -> NDArray:
        """
        Apply proportional tapering to gaps in a mask loaded from .npy array.
        
        This method automatically detects gaps in the input mask and applies
        Tukey window tapering proportional to the gap duration. Different
        taper fractions are applied based on gap length categories.
        
        Parameters
        ----------
        mask_data : np.ndarray
            Input mask data from .npy file. Can contain NaN or 0 for gaps.
        dt : float
            Time step in seconds between samples.
        short_taper_fraction : float, optional
            Fraction of gap duration to taper on each side for short gaps.
            Default is 0.25 (25% each side = 50% total taper).
        medium_taper_fraction : float, optional
            Fraction of gap duration to taper on each side for medium gaps.
            Default is 0.05 (5% each side = 10% total taper).
        long_taper_fraction : float, optional
            Fraction of gap duration to taper on each side for long gaps.
            Default is 0.05 (5% each side = 10% total taper).
        min_gap_points : int, optional
            Minimum number of consecutive gap points to apply tapering.
            Gaps shorter than this are left unchanged. Default is 5.
        short_gap_threshold_minutes : float, optional
            Threshold in minutes to distinguish short from medium gaps.
            Default is 10.0 minutes.
        long_gap_threshold_hours : float, optional
            Threshold in hours to distinguish medium from long gaps.
            Default is 10.0 hours.
            
        Returns
        -------
        np.ndarray
            Tapered mask with smooth transitions around gap edges.
            
        Examples
        --------
        >>> # Load mask from .npy file
        >>> mask = np.load('gap_mask.npy')
        >>> 
        >>> # Apply proportional tapering
        >>> tapered_mask = GapWindowGenerator.apply_proportional_tapering(
        ...     mask, dt=1.0
        ... )
        >>> 
        >>> # Custom tapering for different gap categories
        >>> tapered_mask = GapWindowGenerator.apply_proportional_tapering(
        ...     mask, 
        ...     dt=0.25,
        ...     short_taper_fraction=0.3,   # 30% each side for short gaps
        ...     medium_taper_fraction=0.1,  # 10% each side for medium gaps
        ...     long_taper_fraction=0.02    # 2% each side for long gaps
        ... )
        """
        # Convert thresholds to samples
        short_threshold_samples = int(short_gap_threshold_minutes * 60 / dt)
        long_threshold_samples = int(long_gap_threshold_hours * 3600 / dt)
        
        # Work with a copy to avoid modifying the original
        tapered_mask = mask_data.copy().astype(float)
        
        # Detect gaps (both NaN and zero values)
        is_nan_gap = np.isnan(mask_data)
        # Replace any NaN values with zeros
        tapered_mask[is_nan_gap] = 0.0
        
        is_zero_gap = (mask_data == 0.0)
        gap_mask = is_zero_gap | is_nan_gap  # Include BOTH zero and NaN gaps
        
        
        # Find gap segments
        gap_diff = np.diff(np.concatenate(([False], gap_mask, [False])).astype(int))
        gap_starts = np.where(gap_diff == 1)[0]
        gap_ends = np.where(gap_diff == -1)[0]
        
        
        for i, (start, end) in enumerate(zip(gap_starts, gap_ends)):
            gap_length = end - start
            gap_duration_minutes = gap_length * dt / 60
            
            # Skip very short gaps
            if gap_length < min_gap_points:
                continue
            
            # Categorize gap and select taper fraction
            if gap_length < short_threshold_samples:
                # Short gap
                taper_fraction = short_taper_fraction
                category = "short"
            elif gap_length < long_threshold_samples:
                # Medium gap
                taper_fraction = medium_taper_fraction  
                category = "medium"
            else:
                # Long gap
                taper_fraction = long_taper_fraction
                category = "long"
            
            # Calculate taper window parameters
            taper_samples = int(taper_fraction * gap_length)
            
            # Ensure we have enough samples for tapering
            if taper_samples < 1:
                taper_samples = 1
            elif 2 * taper_samples >= gap_length:
                # If taper would overlap, use maximum possible
                taper_samples = gap_length // 2
            
            # Create extended window including taper regions
            win_start = max(0, start - taper_samples)
            win_end = min(len(mask_data), end + taper_samples)
            window_length = win_end - win_start
            
            if window_length < 3:
                continue
            
            # Calculate alpha for Tukey window
            # Alpha = fraction of window that is tapered
            alpha = 2 * taper_samples / window_length
            alpha = min(alpha, 1.0)  # Ensure alpha <= 1
            
            # Generate Tukey window (1 - tukey gives us the gap shape)
            taper_window = 1 - tukey(window_length, alpha)
            
            # Apply tapering
            tapered_mask[win_start:win_end] = np.minimum(
                tapered_mask[win_start:win_end], 
                taper_window
            )
     
        return tapered_mask
    
    # Delegate other methods to the underlying gap_mask_generator
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying GapMaskGenerator."""
        return getattr(self.gap_mask_generator, name)
