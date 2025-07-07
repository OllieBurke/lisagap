"""Tests for lisa_gap package."""

import numpy as np
import pytest
from lisa_gap import GapMaskGenerator


class TestGapMaskGenerator:
    """Test the GapMaskGenerator class."""

    def test_import(self):
        """Test that GapMaskGenerator can be imported."""
        assert GapMaskGenerator is not None

    def test_basic_initialization(self):
        """Test basic initialization of GapMaskGenerator."""
        # Create simple test data
        dt = 1.0
        sim_t = np.arange(0, 100, dt)
        
        gap_definitions = {
            "planned": {
                "test_gap": {
                    "rate_per_year": 1,
                    "duration_hr": 1.0
                }
            },
            "unplanned": {}
        }
        
        # Test CPU initialization
        gap_gen = GapMaskGenerator(
            sim_t=sim_t,
            dt=dt,
            gap_definitions=gap_definitions,
            use_gpu=False
        )
        
        assert gap_gen is not None
        assert gap_gen.sim_t.shape == sim_t.shape
        assert gap_gen.dt == dt
        assert gap_gen.use_gpu == False
        
        # Test GPU initialization (should fall back to CPU if CuPy not available)
        gap_gen_gpu = GapMaskGenerator(
            sim_t=sim_t,
            dt=dt,
            gap_definitions=gap_definitions,
            use_gpu=True
        )
        
        assert gap_gen_gpu is not None

    def test_gap_mask_generation(self):
        """Test that gap mask can be generated."""
        dt = 1.0
        sim_t = np.arange(0, 1000, dt)
        
        gap_definitions = {
            "planned": {
                "test_gap": {
                    "rate_per_year": 365,  # High rate for testing
                    "duration_hr": 0.1     # Short duration
                }
            },
            "unplanned": {}
        }
        
        # Test CPU version
        gap_gen = GapMaskGenerator(
            sim_t=sim_t,
            dt=dt,
            gap_definitions=gap_definitions,
            use_gpu=False
        )
        
        gap_mask = gap_gen.generate_mask()
        
        # Check that gap mask has correct shape
        assert gap_mask.shape == sim_t.shape
        
        # Check that gap mask is numeric (floats with 1s and 0s)
        assert np.issubdtype(gap_mask.dtype, np.floating)
        
        # Check that values are either 0 or 1
        unique_values = np.unique(gap_mask)
        assert len(unique_values) <= 2  # Should have at most 2 unique values
        assert all(val in [0.0, 1.0] for val in unique_values)  # Should be 0s and 1s
        
        # Test GPU version (should work even if CuPy not available)
        gap_gen_gpu = GapMaskGenerator(
            sim_t=sim_t,
            dt=dt,
            gap_definitions=gap_definitions,
            use_gpu=True
        )
        
        gap_mask_gpu = gap_gen_gpu.generate_mask()
        assert gap_mask_gpu.shape == sim_t.shape

    def test_empty_gap_definitions(self):
        """Test with empty gap definitions."""
        dt = 1.0
        sim_t = np.arange(0, 100, dt)
        
        gap_definitions = {
            "planned": {},
            "unplanned": {}
        }
        
        gap_gen = GapMaskGenerator(
            sim_t=sim_t,
            dt=dt,
            gap_definitions=gap_definitions,
            use_gpu=False
        )
        
        gap_mask = gap_gen.generate_mask()
        
        # With no gaps defined, mask should be all 1s (no gaps to mask)
        assert np.all(gap_mask == 1.0)

    def test_gpu_cpu_consistency(self):
        """Test that GPU and CPU versions produce equivalent results."""
        dt = 1.0
        sim_t = np.arange(0, 100, dt)
        
        gap_definitions = {
            "planned": {},
            "unplanned": {}
        }
        
        # CPU version
        gap_gen_cpu = GapMaskGenerator(
            sim_t=sim_t,
            dt=dt,
            gap_definitions=gap_definitions,
            use_gpu=False,
            planseed=42,
            unplanseed=42
        )
        
        # GPU version (will fall back to CPU if CuPy not available)
        gap_gen_gpu = GapMaskGenerator(
            sim_t=sim_t,
            dt=dt,
            gap_definitions=gap_definitions,
            use_gpu=True,
            planseed=42,
            unplanseed=42
        )
        
        mask_cpu = gap_gen_cpu.generate_mask()
        mask_gpu = gap_gen_gpu.generate_mask()
        
        # Convert GPU result to CPU for comparison
        if hasattr(mask_gpu, 'get'):
            mask_gpu = mask_gpu.get()
        
        # Results should be identical for empty gap definitions
        assert np.allclose(mask_cpu, mask_gpu, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__])
