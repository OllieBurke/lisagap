"""
Unit tests for DataSegmentGenerator class.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from lisagap.gap_segment_generator import DataSegmentGenerator


class TestDataSegmentGenerator:
    """Test suite for DataSegmentGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test data
        self.data_length = 100
        self.dt = 0.1
        self.t0 = 0.0
        
        # Simple test data
        self.data = np.sin(2 * np.pi * 0.1 * np.arange(self.data_length))
        
        # Mask with two gaps: [20:30] and [70:80]
        self.mask = np.ones(self.data_length)
        self.mask[20:30] = np.nan  # First gap
        self.mask[70:80] = np.nan  # Second gap
        
    def test_initialization_basic(self):
        """Test basic initialization."""
        segmenter = DataSegmentGenerator(
            mask=self.mask, 
            data=self.data, 
            dt=self.dt, 
            t0=self.t0
        )
        
        assert len(segmenter.mask) == self.data_length
        assert len(segmenter.data) == self.data_length
        assert segmenter.dt == self.dt
        assert segmenter.t0 == self.t0
        
    def test_initialization_validation(self):
        """Test input validation."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="Mask and data must have the same length"):
            DataSegmentGenerator(
                mask=self.mask[:-10], 
                data=self.data, 
                dt=self.dt
            )
        
        # Invalid dt
        with pytest.raises(ValueError, match="Sampling interval dt must be positive"):
            DataSegmentGenerator(
                mask=self.mask, 
                data=self.data, 
                dt=-0.1
            )
    
    def test_segment_finding(self):
        """Test that segments are correctly identified."""
        segmenter = DataSegmentGenerator(
            mask=self.mask, 
            data=self.data, 
            dt=self.dt
        )
        
        # Should find 3 segments: [0:20], [30:70], [80:100]
        expected_segments = [(0, 20), (30, 70), (80, 100)]
        assert segmenter.segment_indices == expected_segments
    
    def test_get_time_segments(self):
        """Test time domain segmentation."""
        segmenter = DataSegmentGenerator(
            mask=self.mask, 
            data=self.data, 
            dt=self.dt, 
            t0=self.t0
        )
        
        segments = segmenter.get_time_segments()
        
        # Should have 3 segments
        assert len(segments) == 3
        assert 'segment_1' in segments
        assert 'segment_2' in segments
        assert 'segment_3' in segments
        
        # Check first segment
        seg1 = segments['segment_1']
        assert seg1['start_idx'] == 0
        assert seg1['end_idx'] == 20
        assert len(seg1['data']) == 20
        assert len(seg1['time']) == 20
        assert len(seg1['mask']) == 20
        
        # Check time array construction
        expected_time = self.t0 + np.arange(20) * self.dt
        np.testing.assert_array_almost_equal(seg1['time'], expected_time)
        
        # Check data integrity
        np.testing.assert_array_equal(seg1['data'], self.data[0:20])
        np.testing.assert_array_equal(seg1['mask'], self.mask[0:20])
    
    def test_get_freq_info_from_segments(self):
        """Test frequency domain information."""
        segmenter = DataSegmentGenerator(
            mask=self.mask, 
            data=self.data, 
            dt=self.dt
        )
        
        freq_info = segmenter.get_freq_info_from_segments()
        
        # Should have 3 segments
        assert len(freq_info) == 3
        
        # Check first segment
        seg1 = freq_info['segment_1']
        assert 'frequencies' in seg1
        assert 'fft' in seg1
        assert seg1['start_idx'] == 0
        assert seg1['end_idx'] == 20
        
        # Check frequency array
        expected_freqs = np.fft.rfftfreq(20, d=self.dt)
        np.testing.assert_array_almost_equal(seg1['frequencies'], expected_freqs)
        
        # Check FFT data
        expected_fft = np.fft.rfft(self.data[0:20])
        np.testing.assert_array_almost_equal(seg1['fft'], expected_fft)
    
    def test_empty_segments(self):
        """Test handling of data with no valid segments."""
        # All gaps
        all_gap_mask = np.full(self.data_length, np.nan)
        segmenter = DataSegmentGenerator(
            mask=all_gap_mask, 
            data=self.data, 
            dt=self.dt
        )
        
        segments = segmenter.get_time_segments()
        freq_info = segmenter.get_freq_info_from_segments()
        
        assert len(segments) == 0
        assert len(freq_info) == 0
    
    def test_single_segment(self):
        """Test handling of data with no gaps."""
        # No gaps
        no_gap_mask = np.ones(self.data_length)
        segmenter = DataSegmentGenerator(
            mask=no_gap_mask, 
            data=self.data, 
            dt=self.dt
        )
        
        segments = segmenter.get_time_segments()
        
        assert len(segments) == 1
        seg1 = segments['segment_1']
        assert seg1['start_idx'] == 0
        assert seg1['end_idx'] == self.data_length
        assert len(seg1['data']) == self.data_length
    
    def test_mask_with_zeros(self):
        """Test handling of mask with zeros instead of NaN."""
        # Use zeros for gaps instead of NaN
        mask_with_zeros = np.ones(self.data_length)
        mask_with_zeros[20:30] = 0
        mask_with_zeros[70:80] = 0
        
        segmenter = DataSegmentGenerator(
            mask=mask_with_zeros, 
            data=self.data, 
            dt=self.dt
        )
        
        # Should find same segments as with NaN
        expected_segments = [(0, 20), (30, 70), (80, 100)]
        assert segmenter.segment_indices == expected_segments
    
    def test_tapered_mask(self):
        """Test handling of mask with tapering (values between 0 and 1)."""
        # Create mask with tapering
        tapered_mask = np.ones(self.data_length)
        tapered_mask[20:30] = np.nan  # Gap
        tapered_mask[18:20] = [0.5, 0.8]  # Taper before gap
        tapered_mask[30:32] = [0.8, 0.5]  # Taper after gap
        
        segmenter = DataSegmentGenerator(
            mask=tapered_mask, 
            data=self.data, 
            dt=self.dt
        )
        
        segments = segmenter.get_time_segments()
        
        # Should still identify segments correctly
        # The binary mask should treat tapered values as valid (>0)
        assert len(segments) >= 2  # At least before and after the gap
        
        # Check that tapered values are preserved in the mask
        for _, seg_info in segments.items():
            if seg_info['start_idx'] <= 18 < seg_info['end_idx']:
                # This segment should contain the taper
                local_idx = 18 - seg_info['start_idx']
                assert seg_info['mask'][local_idx] == 0.5
    
    def test_summary(self):
        """Test the summary method."""
        segmenter = DataSegmentGenerator(
            mask=self.mask, 
            data=self.data, 
            dt=self.dt, 
            t0=self.t0
        )
        
        summary = segmenter.summary()
        
        assert summary['total_segments'] == 3
        assert summary['total_data_length'] == self.data_length
        assert summary['valid_data_length'] == 80  # 100 - 20 (gaps)
        assert summary['gap_data_length'] == 20
        assert summary['data_fraction_valid'] == 0.8
        assert summary['segment_lengths'] == [20, 40, 20]
        assert summary['min_segment_length'] == 20
        assert summary['max_segment_length'] == 40
        assert summary['mean_segment_length'] == 80/3  # (20+40+20)/3
        assert summary['sampling_interval'] == self.dt
        assert summary['start_time'] == self.t0
    
    def test_different_t0(self):
        """Test with non-zero start time."""
        t0 = 5.0
        segmenter = DataSegmentGenerator(
            mask=self.mask, 
            data=self.data, 
            dt=self.dt, 
            t0=t0
        )
        
        segments = segmenter.get_time_segments()
        seg1 = segments['segment_1']
        
        # Time should start from t0
        expected_time = t0 + np.arange(20) * self.dt
        np.testing.assert_array_almost_equal(seg1['time'], expected_time)
    
    @patch('lisagap.gap_segment_generator.GapWindowGenerator')
    def test_from_gap_generator(self, _mock_gap_window_generator):
        """Test the from_gap_generator class method."""
        # Mock the generate_window method
        mock_instance = Mock()
        mock_instance.generate_window.return_value = self.mask
        
        # Call the class method
        segmenter, returned_mask = DataSegmentGenerator.from_gap_generator(
            gap_window_generator=mock_instance,
            data=self.data,
            dt=self.dt,
            t0=self.t0,
            apply_tapering=True
        )
        
        # Check that generate_window was called with correct parameters
        mock_instance.generate_window.assert_called_once_with(
            apply_tapering=True,
            taper_definitions=None
        )
        
        # Check return values
        assert isinstance(segmenter, DataSegmentGenerator)
        np.testing.assert_array_equal(returned_mask, self.mask)
        
        # Check that segmenter was initialized correctly
        assert segmenter.dt == self.dt
        assert segmenter.t0 == self.t0
        np.testing.assert_array_equal(segmenter.data, self.data)
    
    def test_edge_case_single_sample_segment(self):
        """Test handling of single-sample segments."""
        # Create mask with single-sample segments
        mask = np.full(10, np.nan)
        mask[3] = 1  # Single valid sample
        mask[7] = 1  # Another single valid sample
        data = np.arange(10)
        
        segmenter = DataSegmentGenerator(mask=mask, data=data, dt=0.1)
        segments = segmenter.get_time_segments()
        
        assert len(segments) == 2
        
        # Check first single-sample segment
        seg1 = segments['segment_1']
        assert seg1['start_idx'] == 3
        assert seg1['end_idx'] == 4
        assert len(seg1['data']) == 1
        assert seg1['data'][0] == data[3]
    
    def test_edge_case_gaps_at_boundaries(self):
        """Test handling of gaps at the start and end."""
        # Gap at start and end
        mask = np.ones(20)
        mask[0:5] = np.nan  # Gap at start
        mask[15:20] = np.nan  # Gap at end
        data = np.arange(20)
        
        segmenter = DataSegmentGenerator(mask=mask, data=data, dt=0.1)
        segments = segmenter.get_time_segments()
        
        # Should have one segment from 5 to 15
        assert len(segments) == 1
        seg1 = segments['segment_1']
        assert seg1['start_idx'] == 5
        assert seg1['end_idx'] == 15
    
    def test_apply_window_functionality(self):
        """Test the apply_window parameter."""
        # Create mask with tapering values
        mask = np.ones(20)
        mask[5:8] = [0.2, 0.5, 0.8]  # Tapered region
        mask[10:15] = np.nan  # Gap
        data = np.ones(20) * 10  # Constant data for easy testing
        
        segmenter = DataSegmentGenerator(mask=mask, data=data, dt=0.1)
        
        # Test without apply_window
        segments_raw = segmenter.get_time_segments(apply_window=False)
        seg1 = segments_raw['segment_1']
        # Data should be unchanged
        np.testing.assert_array_equal(seg1['data'], np.ones(10) * 10)
        
        # Test with apply_window
        segments_windowed = segmenter.get_time_segments(apply_window=True)
        seg1_windowed = segments_windowed['segment_1']
        # Data should be multiplied by mask
        expected_data = np.array([10, 10, 10, 10, 10, 2, 5, 8, 10, 10])
        np.testing.assert_array_almost_equal(seg1_windowed['data'], expected_data)
        
        # Mask should be preserved in both cases
        np.testing.assert_array_equal(seg1['mask'], seg1_windowed['mask'])
    
    def test_edge_tapering_left(self):
        """Test left edge tapering on first segment."""
        # Simple case with no gaps
        mask = np.ones(100)
        data = np.ones(100) * 10
        
        segmenter = DataSegmentGenerator(mask=mask, data=data, dt=0.1)
        
        # Apply left edge tapering
        segments = segmenter.get_time_segments(
            apply_window=True, 
            left_edge_taper=10
        )
        
        seg1 = segments['segment_1']
        
        # Check that left edge is tapered
        assert seg1['data'][0] == 0.0  # Should start at 0
        assert seg1['data'][5] < seg1['data'][15]  # Should ramp up
        assert seg1['data'][15] == 10.0  # Should reach full value
        
        # Check that mask reflects the tapering
        assert seg1['mask'][0] == 0.0
        assert seg1['mask'][15] == 1.0
    
    def test_edge_tapering_right(self):
        """Test right edge tapering on last segment."""
        # Single segment case
        mask = np.ones(100)
        data = np.ones(100) * 10
        
        segmenter = DataSegmentGenerator(mask=mask, data=data, dt=0.1)
        
        # Apply right edge tapering
        segments = segmenter.get_time_segments(
            apply_window=True, 
            right_edge_taper=10
        )
        
        seg1 = segments['segment_1']
        
        # Check that right edge is tapered
        assert seg1['data'][-1] == 0.0  # Should end at 0
        assert seg1['data'][-5] < seg1['data'][-15]  # Should ramp down
        assert seg1['data'][50] == 10.0  # Should be full value in middle
        
        # Check that mask reflects the tapering
        assert seg1['mask'][-1] == 0.0
        assert seg1['mask'][50] == 1.0
    
    def test_edge_tapering_both_edges(self):
        """Test both left and right edge tapering."""
        # Single segment case
        mask = np.ones(100)
        data = np.ones(100) * 10
        
        segmenter = DataSegmentGenerator(mask=mask, data=data, dt=0.1)
        
        # Apply both edge taperings
        segments = segmenter.get_time_segments(
            apply_window=True, 
            left_edge_taper=15,
            right_edge_taper=20
        )
        
        seg1 = segments['segment_1']
        
        # Check left edge
        assert seg1['data'][0] == 0.0
        assert seg1['data'][20] == 10.0  # Past left taper
        
        # Check right edge
        assert seg1['data'][-1] == 0.0
        assert seg1['data'][-25] == 10.0  # Past right taper
        
        # Check middle is untouched
        assert seg1['data'][50] == 10.0
    
    def test_edge_tapering_multiple_segments(self):
        """Test that edge tapering only affects first and last segments."""
        # Create mask with multiple segments
        mask = np.ones(200)
        mask[50:60] = np.nan  # Gap creating two segments
        mask[120:130] = np.nan  # Another gap creating three segments
        data = np.ones(200) * 10
        
        segmenter = DataSegmentGenerator(mask=mask, data=data, dt=0.1)
        
        segments = segmenter.get_time_segments(
            apply_window=True,
            left_edge_taper=10,
            right_edge_taper=15
        )
        
        # Should have 3 segments
        assert len(segments) == 3
        
        # First segment should have left taper
        seg1 = segments['segment_1']
        assert seg1['data'][0] == 0.0  # Left taper applied
        assert seg1['data'][-1] == 10.0  # No right taper
        
        # Middle segment should have no edge tapering
        seg2 = segments['segment_2']
        assert seg2['data'][0] == 10.0  # No left taper
        assert seg2['data'][-1] == 10.0  # No right taper
        
        # Last segment should have right taper
        seg3 = segments['segment_3']
        assert seg3['data'][0] == 10.0  # No left taper
        assert seg3['data'][-1] == 0.0  # Right taper applied
    
    def test_edge_tapering_no_window(self):
        """Test that edge tapering is ignored when apply_window=False."""
        mask = np.ones(50)
        data = np.ones(50) * 10
        
        segmenter = DataSegmentGenerator(mask=mask, data=data, dt=0.1)
        
        # Edge taper parameters should be ignored
        segments = segmenter.get_time_segments(
            apply_window=False,  # Key: no windowing
            left_edge_taper=10,
            right_edge_taper=10
        )
        
        seg1 = segments['segment_1']
        
        # Data should be unchanged (no tapering applied)
        np.testing.assert_array_equal(seg1['data'], np.ones(50) * 10)
        np.testing.assert_array_equal(seg1['mask'], np.ones(50))


if __name__ == '__main__':
    pytest.main([__file__])