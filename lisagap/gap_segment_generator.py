"""
DataSegmentGenerator - Segment data into continuous chunks for separate analysis.

This module provides functionality to segment time series data based on gap masks,
enabling separate analysis of continuous data segments rather than using tapering/windowing approaches.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Tuple, Optional
from .gap_window_generator import GapWindowGenerator


class DataSegmentGenerator:
    """
    Generator for segmenting time series data into continuous chunks based on gap masks.
    
    This class takes a binary mask (1s for valid data, NaN/0s for gaps) and segments
    the corresponding data into continuous chunks. Each segment contains the data,
    time stamps, mask information, and indices for separate analysis.
    
    Parameters
    ----------
    mask : NDArray
        Binary mask where 1 indicates valid data and NaN/0 indicates gaps.
    data : NDArray
        Time series data corresponding to the mask.
    dt : float
        Sampling interval (time step between samples).
    t0 : float, optional
        Start time for the time series. Default is 0.0.
        
    Examples
    --------
    >>> import numpy as np
    >>> from lisagap import DataSegmentGenerator
    >>> 
    >>> # Create sample data with gaps
    >>> data = np.random.randn(1000)
    >>> mask = np.ones_like(data)
    >>> mask[200:300] = np.nan  # Create a gap
    >>> mask[500:520] = np.nan  # Another gap
    >>> 
    >>> # Create segmenter
    >>> segmenter = DataSegmentGenerator(mask=mask, data=data, dt=0.1, t0=0.0)
    >>> 
    >>> # Get time domain segments
    >>> segments = segmenter.get_time_segments()
    >>> 
    >>> # Get frequency domain information
    >>> freq_info = segmenter.get_freq_info_from_segments()
    """
    
    def __init__(
        self,
        mask: NDArray,
        data: NDArray,
        dt: float,
        t0: float = 0.0
    ):
        """
        Initialize the DataSegmentGenerator.
        
        Parameters
        ----------
        mask : NDArray
            Binary mask where 1 indicates valid data and NaN/0 indicates gaps.
        data : NDArray
            Time series data corresponding to the mask.
        dt : float
            Sampling interval (time step between samples).
        t0 : float, optional
            Start time for the time series. Default is 0.0.
        """
        self.mask = np.array(mask)
        self.data = np.array(data)
        self.dt = dt
        self.t0 = t0
        
        # Validate inputs
        if len(self.mask) != len(self.data):
            raise ValueError("Mask and data must have the same length")
        
        if self.dt <= 0:
            raise ValueError("Sampling interval dt must be positive")
            
        # Convert mask to binary (handle NaN values)
        self.binary_mask = np.where(np.isnan(self.mask) | (self.mask == 0), 0, 1)
        
        # Find continuous segments
        self._find_segments()
    
    def _find_segments(self) -> None:
        """
        Find continuous segments of valid data in the mask.
        
        This method identifies the start and end indices of continuous
        segments where the mask indicates valid data (value = 1).
        """
        # Find transitions in the binary mask
        diff = np.diff(np.concatenate(([0], self.binary_mask, [0])))
        
        # Find start and end indices of segments
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Store segment information
        self.segment_indices = []
        for start, end in zip(starts, ends):
            if end > start:  # Only keep non-empty segments
                self.segment_indices.append((start, end))
    
    def get_time_segments(
        self, 
        apply_window: bool = False,
        left_edge_taper: Optional[int] = None,
        right_edge_taper: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get time domain segments of the data.
        
        Parameters
        ----------
        apply_window : bool, optional
            If True, apply the windowing/tapering to the segmented data.
            Default is False.
        left_edge_taper : int, optional
            Number of samples to taper on the left edge of the first segment.
            Only applied when apply_window=True. Default is None (no edge tapering).
        right_edge_taper : int, optional
            Number of samples to taper on the right edge of the last segment.
            Only applied when apply_window=True. Default is None (no edge tapering).
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary containing segmented data with keys:
            - 'data': Data array for the segment (windowed if apply_window=True)
            - 'time': Time array for the segment  
            - 'mask': Mask array for the segment (showing any tapering applied)
            - 'start_idx': Start index in original array
            - 'end_idx': End index in original array
        """
        segments = {}
        
        for i, (start, end) in enumerate(self.segment_indices):
            segment_key = f"segment_{i+1}"
            
            # Extract segment data and mask
            segment_data = self.data[start:end].copy()
            segment_mask = self.mask[start:end].copy()
            
            # Apply windowing if requested
            if apply_window:
                # Apply the existing mask windowing
                segment_data = segment_data * segment_mask
                
                # Apply edge tapering for first and last segments
                segment_length = end - start
                
                # Left edge tapering for first segment
                if i == 0 and left_edge_taper is not None:
                    if left_edge_taper > 0 and left_edge_taper < segment_length:
                        # Create one-sided Tukey window (ramp up from 0 to 1)
                        left_taper = np.ones(segment_length)
                        for j in range(min(left_edge_taper, segment_length)):
                            # Cosine ramp from 0 to 1
                            left_taper[j] = 0.5 * (1 - np.cos(np.pi * j / left_edge_taper))
                        
                        # Apply left edge taper to data and update mask
                        segment_data = segment_data * left_taper
                        segment_mask = segment_mask * left_taper
                
                # Right edge tapering for last segment
                if i == len(self.segment_indices) - 1 and right_edge_taper is not None:
                    if right_edge_taper > 0 and right_edge_taper < segment_length:
                        # Create one-sided Tukey window (ramp down from 1 to 0)
                        right_taper = np.ones(segment_length)
                        for j in range(min(right_edge_taper, segment_length)):
                            # Cosine ramp from 1 to 0
                            idx = segment_length - 1 - j
                            right_taper[idx] = 0.5 * (1 - np.cos(np.pi * j / right_edge_taper))
                        
                        # Apply right edge taper to data and update mask
                        segment_data = segment_data * right_taper
                        segment_mask = segment_mask * right_taper
            
            # Create time array for this segment
            segment_length = end - start
            segment_time = self.t0 + (start + np.arange(segment_length)) * self.dt
            
            segments[segment_key] = {
                'data': segment_data,
                'time': segment_time,
                'mask': segment_mask,
                'start_idx': start,
                'end_idx': end
            }
        
        return segments
    
    def get_freq_info_from_segments(self) -> Dict[str, Dict[str, Any]]:
        """
        Get frequency domain information for each segment.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary containing frequency information with keys:
            - 'frequencies': Frequency bins for the segment
            - 'fft': FFT of the segment data
            - 'start_idx': Start index in original array
            - 'end_idx': End index in original array
        """
        freq_info = {}
        
        for i, (start, end) in enumerate(self.segment_indices):
            segment_key = f"segment_{i+1}"
            
            # Extract segment data
            segment_data = self.data[start:end]
            segment_length = end - start
            
            # Compute FFT
            fft_data = np.fft.rfft(segment_data)
            
            # Create frequency array
            frequencies = np.fft.rfftfreq(segment_length, d=self.dt)
            
            freq_info[segment_key] = {
                'frequencies': frequencies,
                'fft': fft_data,
                'start_idx': start,
                'end_idx': end
            }
        
        return freq_info
    
    @classmethod
    def from_gap_generator(
        cls,
        gap_window_generator: GapWindowGenerator,
        data: NDArray,
        dt: float,
        t0: float = 0.0,
        apply_tapering: bool = False,
        taper_definitions: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        **kwargs
    ) -> Tuple['DataSegmentGenerator', NDArray]:
        """
        Create DataSegmentGenerator from a GapWindowGenerator.
        
        This class method generates a mask using the provided GapWindowGenerator
        and returns both the DataSegmentGenerator instance and the mask for
        downstream reuse.
        
        Parameters
        ----------
        gap_window_generator : GapWindowGenerator
            Configured GapWindowGenerator instance.
        data : NDArray
            Time series data to segment.
        dt : float
            Sampling interval.
        t0 : float, optional
            Start time. Default is 0.0.
        apply_tapering : bool, optional
            Whether to apply tapering to the mask. Default is False.
        taper_definitions : dict, optional
            Tapering definitions for the mask.
        **kwargs
            Additional arguments passed to generate_window().
            
        Returns
        -------
        Tuple[DataSegmentGenerator, NDArray]
            Tuple containing:
            - DataSegmentGenerator instance
            - The generated mask (for downstream reuse)
        """
        # Generate mask using the GapWindowGenerator
        mask = gap_window_generator.generate_window(
            apply_tapering=apply_tapering,
            taper_definitions=taper_definitions,
            **kwargs
        )
        
        # Create DataSegmentGenerator instance
        segmenter = cls(mask=mask, data=data, dt=dt, t0=t0)
        
        return segmenter, mask
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary information about the segmentation.
        
        Returns
        -------
        Dict[str, Any]
            Summary containing:
            - Number of segments
            - Total data length
            - Total valid data length
            - Segment lengths
            - Gap information
        """
        total_length = len(self.data)
        valid_length = np.sum(self.binary_mask)
        gap_length = total_length - valid_length
        
        segment_lengths = [end - start for start, end in self.segment_indices]
        
        return {
            'total_segments': len(self.segment_indices),
            'total_data_length': total_length,
            'valid_data_length': valid_length,
            'gap_data_length': gap_length,
            'data_fraction_valid': valid_length / total_length if total_length > 0 else 0,
            'segment_lengths': segment_lengths,
            'min_segment_length': min(segment_lengths) if segment_lengths else 0,
            'max_segment_length': max(segment_lengths) if segment_lengths else 0,
            'mean_segment_length': np.mean(segment_lengths) if segment_lengths else 0,
            'sampling_interval': self.dt,
            'start_time': self.t0
        }