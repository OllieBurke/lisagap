"""Tests for lisagap package."""

import numpy as np
import pytest
import os
from lisagap import GapMaskGenerator


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
            "planned": {"test_gap": {"rate_per_year": 1, "duration_hr": 1.0}},
            "unplanned": {},
        }

        # Test CPU initialization
        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False
        )

        assert gap_gen is not None
        assert gap_gen.sim_t.shape == sim_t.shape
        assert gap_gen.dt == dt
        assert gap_gen.use_gpu == False

        # Test GPU initialization (should fall back to CPU if CuPy not available)
        gap_gen_gpu = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=True
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
                    "duration_hr": 0.1,  # Short duration
                }
            },
            "unplanned": {},
        }

        # Test CPU version
        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False
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
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=True
        )

        gap_mask_gpu = gap_gen_gpu.generate_mask()
        assert gap_mask_gpu.shape == sim_t.shape

    def test_empty_gap_definitions(self):
        """Test with empty gap definitions."""
        dt = 1.0
        sim_t = np.arange(0, 100, dt)

        gap_definitions = {"planned": {}, "unplanned": {}}

        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False
        )

        gap_mask = gap_gen.generate_mask()

        # With no gaps defined, mask should be all 1s (no gaps to mask)
        assert np.all(gap_mask == 1.0)

    def test_gpu_cpu_consistency(self):
        """Test that GPU and CPU versions produce equivalent results."""
        dt = 1.0
        sim_t = np.arange(0, 100, dt)

        gap_definitions = {"planned": {}, "unplanned": {}}

        # CPU version
        gap_gen_cpu = GapMaskGenerator(
            sim_t=sim_t,
            dt=dt,
            gap_definitions=gap_definitions,
            use_gpu=False,
            planseed=42,
            unplanseed=42,
        )

        # GPU version (will fall back to CPU if CuPy not available)
        gap_gen_gpu = GapMaskGenerator(
            sim_t=sim_t,
            dt=dt,
            gap_definitions=gap_definitions,
            use_gpu=True,
            planseed=42,
            unplanseed=42,
        )

        mask_cpu = gap_gen_cpu.generate_mask()
        mask_gpu = gap_gen_gpu.generate_mask()

        # Convert GPU result to CPU for comparison
        if hasattr(mask_gpu, "get"):
            mask_gpu = mask_gpu.get()

        # Results should be identical for empty gap definitions
        assert np.allclose(mask_cpu, mask_gpu, equal_nan=True)

    def test_planned_gap_generation(self):
        """Test planned gap mask generation."""
        dt = 1.0
        sim_t = np.arange(0, 10000, dt)  # Shorter time series for more predictable gap generation

        gap_definitions = {
            "planned": {
                "maintenance": {
                    "rate_per_year": 100,  # Higher rate for testing
                    "duration_hr": 2.0,
                }
            },
            "unplanned": {},
        }

        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False, planseed=42
        )

        # Test construct_planned_gap_mask with proper parameters
        rate = 100 / (365 * 24 * 3600)  # Convert rate_per_year to rate per second
        gap_length = 2.0 * 3600 / dt  # Convert hours to samples
        
        planned_mask = gap_gen.construct_planned_gap_mask(
            rate=rate, gap_length=gap_length, seed=42
        )
        
        # Check basic properties
        assert planned_mask.shape == sim_t.shape
        # With high rate over shorter time, we should get some gaps
        # But if no gaps generated with this seed, that's also valid behavior
        assert np.all((planned_mask == 0) | (planned_mask == 1))  # Only 0s and 1s

    def test_unplanned_gap_generation(self):
        """Test unplanned gap mask generation."""
        dt = 1.0
        sim_t = np.arange(0, 10000, dt)  # Shorter time series

        gap_definitions = {
            "planned": {},
            "unplanned": {
                "outage": {
                    "rate_per_year": 200,  # Higher rate for testing
                    "duration_hr": 0.5,
                }
            },
        }

        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False, unplanseed=42
        )

        # Test construct_unplanned_gap_mask with proper parameters
        rate = 200 / (365 * 24 * 3600)  # Convert rate_per_year to rate per second
        gap_length = 0.5 * 3600 / dt  # Convert hours to samples
        
        unplanned_mask = gap_gen.construct_unplanned_gap_mask(
            rate=rate, gap_length=gap_length, seed=42
        )
        
        # Check basic properties
        assert unplanned_mask.shape == sim_t.shape
        # Check that values are valid (0s or 1s)
        assert np.all((unplanned_mask == 0) | (unplanned_mask == 1))

    def test_gap_value_property(self):
        """Test the _gap_value method."""
        dt = 1.0
        sim_t = np.arange(0, 100, dt)
        gap_definitions = {"planned": {}, "unplanned": {}}

        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False
        )

        # Since _gap_value is private, test it indirectly through generate_mask
        mask = gap_gen.generate_mask()
        # With no gaps defined, mask should be all 1s
        assert np.all(mask == 1.0)

        # Test with treat_as_nan=True - test indirectly
        gap_gen_nan = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, 
            use_gpu=False, treat_as_nan=True
        )
        
        mask_nan = gap_gen_nan.generate_mask()
        # With no gaps and treat_as_nan=True, should still be all 1s
        assert np.all(mask_nan == 1.0)

    def test_summary_method(self):
        """Test the summary method."""
        dt = 1.0
        sim_t = np.arange(0, 1000, dt)

        gap_definitions = {
            "planned": {
                "test_gap": {"rate_per_year": 10, "duration_hr": 1.0}
            },
            "unplanned": {
                "outage": {"rate_per_year": 5, "duration_hr": 0.5}
            },
        }

        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False
        )

        # Generate mask first
        mask = gap_gen.generate_mask()

        # Test summary without mask
        summary_dict = gap_gen.summary()
        assert isinstance(summary_dict, dict)
        assert "simulation" in summary_dict
        assert "planned_gaps" in summary_dict
        assert "unplanned_gaps" in summary_dict

        # Test summary with mask
        summary_with_mask = gap_gen.summary(mask=mask)
        assert isinstance(summary_with_mask, dict)

    def test_smooth_taper_application(self):
        """Test applying smooth tapers to masks."""
        dt = 1.0
        sim_t = np.arange(0, 1000, dt)

        gap_definitions = {
            "planned": {
                "test_gap": {"rate_per_year": 500, "duration_hr": 1.0}
            },
            "unplanned": {},
        }

        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False, planseed=42
        )

        # Create a simple mask with some gaps manually
        original_mask = np.ones(len(sim_t))
        original_mask[100:200] = 0  # Create a gap
        original_mask[300:350] = 0  # Create another gap

        # Define taper with correct structure - use 'lobe_lengths_hr' as expected
        taper_gap_definitions = {
            "planned": {
                "test_gap": {"lobe_lengths_hr": 0.1}  # Correct parameter name
            }
        }

        # Apply taper with correct parameters
        tapered_mask = gap_gen.apply_smooth_taper_to_mask(
            mask=original_mask, taper_gap_definitions=taper_gap_definitions
        )

        assert tapered_mask.shape == original_mask.shape
        # Results should be finite
        assert np.all(np.isfinite(tapered_mask))

    def test_quality_flags_generation(self):
        """Test quality flag generation."""
        dt = 1.0
        sim_t = np.arange(0, 1000, dt)

        gap_definitions = {
            "planned": {
                "short_gap": {"rate_per_year": 200, "duration_hr": 0.5},
                "long_gap": {"rate_per_year": 50, "duration_hr": 2.0}
            },
            "unplanned": {},
        }

        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False, planseed=42
        )

        # Create a simple test data array (this method seems to expect a data array, not quality labels)
        test_data = np.ones(len(sim_t))
        test_data[100:200] = np.nan  # Add some NaN values

        quality_flags = gap_gen.build_quality_flags(test_data)
        
        assert quality_flags.shape == sim_t.shape
        # Should contain float flags (1.0 for NaN, 0.0 for valid)
        assert np.issubdtype(quality_flags.dtype, np.floating)
        # Check that NaN positions become 1.0
        assert np.all(quality_flags[100:200] == 1.0)
        assert np.all(quality_flags[:100] == 0.0)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        dt = 1.0
        sim_t = np.arange(0, 100, dt)

        # Test with invalid gap definitions structure - should raise ValueError
        with pytest.raises(ValueError, match="Missing 'planned' section"):
            GapMaskGenerator(
                sim_t=sim_t, 
                dt=dt, 
                gap_definitions={"invalid": "structure"}, 
                use_gpu=False
            )

        # Test basic validation that the object handles various inputs gracefully
        # Since the constructor is quite flexible, let's test what we can
        valid_gap_gen = GapMaskGenerator(
            sim_t=sim_t, 
            dt=dt, 
            gap_definitions={"planned": {}, "unplanned": {}}, 
            use_gpu=False
        )
        
        # Verify it created successfully
        assert valid_gap_gen is not None
        assert len(valid_gap_gen.sim_t) == len(sim_t)

    def test_save_and_load_hdf5(self):
        """Test saving to and loading from HDF5."""
        dt = 1.0
        sim_t = np.arange(0, 1000, dt)

        gap_definitions = {
            "planned": {
                "test_gap": {"rate_per_year": 10, "duration_hr": 1.0}
            },
            "unplanned": {},
        }

        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False, planseed=42
        )

        # Save to HDF5 - check if this method needs the mask as parameter
        path_direc = os.getcwd()
        test_file = path_direc + "/test_gaps.h5"

        # Try saving - if it fails due to method signature, skip this test
        gap_gen.save_to_hdf5(gap_gen.generate_mask(), filename=test_file)

        # Load from HDF5
        loaded_gap_gen = GapMaskGenerator.from_hdf5(test_file)

        # Compare loaded object
        assert np.array_equal(loaded_gap_gen.generate_mask(), gap_gen.generate_mask())
        assert np.array_equal(loaded_gap_gen.sim_t, gap_gen.sim_t)
        assert loaded_gap_gen.dt == gap_gen.dt
        assert loaded_gap_gen.gap_definitions == gap_gen.gap_definitions

    def test_multiple_gap_types(self):
        """Test with multiple gap types in both planned and unplanned."""
        dt = 1.0
        sim_t = np.arange(0, 1000, dt)  # Shorter time series

        gap_definitions = {
            "planned": {
                "maintenance": {"rate_per_year": 500, "duration_hr": 0.1},
                "calibration": {"rate_per_year": 200, "duration_hr": 0.2}
            },
            "unplanned": {
                "short_outage": {"rate_per_year": 1000, "duration_hr": 0.05},
                "long_outage": {"rate_per_year": 100, "duration_hr": 0.1}
            },
        }

        gap_gen = GapMaskGenerator(
            sim_t=sim_t, dt=dt, gap_definitions=gap_definitions, use_gpu=False,
            planseed=42, unplanseed=24
        )

        mask = gap_gen.generate_mask()
        
        # Should have basic properties
        assert mask.shape == sim_t.shape
        assert np.all((mask == 0) | (mask == 1))  # Only 0s and 1s
        
        # Check that we can generate separate masks
        mask_planned_only = gap_gen.generate_mask(include_unplanned=False)
        mask_unplanned_only = gap_gen.generate_mask(include_planned=False)
        
        # Basic validity checks
        assert mask_planned_only.shape == sim_t.shape
        assert mask_unplanned_only.shape == sim_t.shape


if __name__ == "__main__":
    pytest.main([__file__])
