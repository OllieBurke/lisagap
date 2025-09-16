"""Tests for the GapWindowGenerator class."""

import numpy as np
import pytest

try:
    from lisagap import GapWindowGenerator
    from lisaglitch import GapMaskGenerator
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Required dependencies not available")
class TestGapWindowGenerator:
    """Test the GapWindowGenerator class."""

    @pytest.fixture
    def gap_mask_generator(self):
        """Create a basic GapMaskGenerator for testing."""
        dt = 1.0
        sim_t = np.arange(0, 100, dt)
        gap_definitions = {
            "planned": {
                "test_gap": {"rate_per_year": 1, "duration_hr": 2}
            },
            "unplanned": {
                "test_gap": {"rate_per_year": 0.5, "duration_hr": 1}
            }
        }
        return GapMaskGenerator(
            sim_t=sim_t,
            gap_definitions=gap_definitions
        )

    def test_import(self):
        """Test that GapWindowGenerator can be imported."""
        assert GapWindowGenerator is not None

    def test_basic_initialization(self, gap_mask_generator):
        """Test basic initialization of GapWindowGenerator."""
        # Test initialization with GapMaskGenerator instance
        gap_gen = GapWindowGenerator(gap_mask_generator)
        
        # Check attributes are inherited correctly
        assert np.array_equal(gap_gen.sim_t, gap_mask_generator.sim_t)
        assert gap_gen.dt == gap_mask_generator.dt
        assert gap_gen.gap_definitions == gap_mask_generator.gap_definitions
        assert gap_gen.treat_as_nan == gap_mask_generator.treat_as_nan
        assert hasattr(gap_gen, 'gap_mask_generator')

    def test_gap_labels_extraction(self):
        """Test that gap labels are correctly extracted."""
        dt = 1.0
        sim_t = np.arange(0, 100, dt)
        
        gap_definitions = {
            "planned": {
                "maintenance": {"rate_per_year": 2, "duration_hr": 4},
                "calibration": {"rate_per_year": 1, "duration_hr": 1}
            },
            "unplanned": {
                "hardware_failure": {"rate_per_year": 0.1, "duration_hr": 8},
                "communication_loss": {"rate_per_year": 0.5, "duration_hr": 2}
            }
        }
        
        gap_mask_gen = GapMaskGenerator(
            sim_t=sim_t,
            gap_definitions=gap_definitions
        )
        gap_gen = GapWindowGenerator(gap_mask_gen)
        
        assert set(gap_gen.planned_labels) == {"maintenance", "calibration"}
        assert set(gap_gen.unplanned_labels) == {"hardware_failure", "communication_loss"}

    def test_generate_mask_basic(self, gap_mask_generator):
        """Test basic mask generation functionality."""
        gap_gen = GapWindowGenerator(gap_mask_generator)
        
        # Test basic mask generation
        mask = gap_gen.generate_mask()
        assert mask is not None
        assert len(mask) == len(gap_mask_generator.sim_t)
        assert isinstance(mask, np.ndarray)
        
        # Test selective generation
        planned_mask = gap_gen.generate_mask(include_unplanned=False)
        unplanned_mask = gap_gen.generate_mask(include_planned=False)
        
        assert len(planned_mask) == len(mask)
        assert len(unplanned_mask) == len(mask)

    def test_apply_smooth_taper_basic(self, gap_mask_generator):
        """Test basic tapering functionality."""
        gap_gen = GapWindowGenerator(gap_mask_generator)
        
        # Create a simple test mask with a gap
        mask = np.ones(len(gap_mask_generator.sim_t))
        mask[40:60] = 0  # Create a gap
        
        # Simple taper definitions
        taper_defs = {
            "planned": {
                "test_gap": {"lobe_lengths_hr": 0.5}
            }
        }
        
        # Apply tapering
        tapered_mask = gap_gen.apply_smooth_taper_to_mask(mask, taper_defs)
        
        # Check that we get a result
        assert tapered_mask is not None
        assert len(tapered_mask) == len(mask)
        assert isinstance(tapered_mask, np.ndarray)

    def test_generate_mask_with_tapering(self, gap_mask_generator):
        """Test mask generation with tapering."""
        gap_gen = GapWindowGenerator(gap_mask_generator)
        
        taper_defs = {
            "planned": {
                "test_gap": {"lobe_lengths_hr": 0.5}
            }
        }
        
        # Generate mask with tapering
        mask = gap_gen.generate_window(
            apply_tapering=True,
            taper_definitions=taper_defs
        )
        
        assert mask is not None
        assert len(mask) == len(gap_mask_generator.sim_t)

    def test_generate_mask_tapering_error(self, gap_mask_generator):
        """Test that proper error is raised when tapering requested without definitions."""
        gap_gen = GapWindowGenerator(gap_mask_generator)
        
        with pytest.raises(ValueError, match="taper_definitions must be provided"):
            gap_gen.generate_window(apply_tapering=True)

    def test_validate_taper_definitions(self, gap_mask_generator):
        """Test taper definition validation."""
        gap_gen = GapWindowGenerator(gap_mask_generator)
        
        # Test with None (should not raise error)
        gap_gen._validate_taper_definitions(None)  # Should not raise error
        
        # Test with valid structure
        valid_taper_defs = {
            "planned": {
                "test_gap": {"lobe_lengths_hr": 0.5}
            }
        }
        gap_gen._validate_taper_definitions(valid_taper_defs)  # Should not raise error

    def test_attribute_inheritance(self, gap_mask_generator):
        """Test that GapWindowGenerator properly inherits attributes."""
        gap_gen = GapWindowGenerator(gap_mask_generator)
        
        # Test direct attribute access
        assert gap_gen.planned_rates is not None
        assert gap_gen.unplanned_rates is not None
        assert gap_gen.planned_durations is not None 
        assert gap_gen.unplanned_durations is not None
        
        # Test that attributes match the original
        assert np.array_equal(gap_gen.planned_rates, gap_mask_generator.planned_rates)
        assert np.array_equal(gap_gen.unplanned_rates, gap_mask_generator.unplanned_rates)

    def test_method_delegation(self, gap_mask_generator):
        """Test that methods are properly delegated."""
        gap_gen = GapWindowGenerator(gap_mask_generator)
        
        # Test summary method
        summary = gap_gen.summary()
        assert isinstance(summary, dict)
        
        # Test that summary matches original
        original_summary = gap_mask_generator.summary()
        assert summary == original_summary

    def test_proportional_tapering_static_method(self):
        """Test the static proportional tapering method."""
        # Create test mask with different gap types
        mask = np.ones(1000)
        mask[100:150] = 0        # Short gap: 50 samples
        mask[300:600] = np.nan   # Medium gap: 300 samples  
        mask[800:950] = 0        # Medium gap: 150 samples
        
        # Apply proportional tapering
        tapered = GapWindowGenerator.apply_proportional_tapering(
            mask, 
            dt=1.0,
            short_taper_fraction=0.25,
            medium_taper_fraction=0.05
        )
        
        # Basic checks
        assert tapered is not None
        assert len(tapered) == len(mask)
        assert isinstance(tapered, np.ndarray)
        
        # Check that tapering was applied (should have values between 0 and 1)
        unique_values = np.unique(tapered[~np.isnan(tapered)])
        has_intermediate_values = np.any((unique_values > 0) & (unique_values < 1))
        assert has_intermediate_values, "Tapering should create intermediate values"
        
        # Check range
        assert np.nanmin(tapered) >= 0.0
        assert np.nanmax(tapered) <= 1.0

    def test_proportional_tapering_edge_cases(self):
        """Test edge cases for proportional tapering."""
        # Test with very short gaps (should be skipped)
        mask = np.ones(100)
        mask[10:12] = 0  # 2-point gap (less than min_gap_points=5)
        
        tapered = GapWindowGenerator.apply_proportional_tapering(
            mask, dt=1.0, min_gap_points=5
        )
        
        # Should be unchanged since gap is too short
        np.testing.assert_array_equal(tapered, mask)
        
        # Test with empty mask (no gaps)
        mask_no_gaps = np.ones(100)
        tapered_no_gaps = GapWindowGenerator.apply_proportional_tapering(
            mask_no_gaps, dt=1.0
        )
        
        # Should be unchanged
        np.testing.assert_array_equal(tapered_no_gaps, mask_no_gaps)

    def test_proportional_tapering_parameters(self):
        """Test proportional tapering with different parameters."""
        mask = np.ones(500)
        mask[100:200] = 0  # 100-sample gap
        
        # Test with different taper fractions
        tapered1 = GapWindowGenerator.apply_proportional_tapering(
            mask, dt=1.0, short_taper_fraction=0.1
        )
        
        tapered2 = GapWindowGenerator.apply_proportional_tapering(
            mask, dt=1.0, short_taper_fraction=0.3
        )
        
        # More aggressive tapering should create more intermediate values
        n_intermediate1 = np.sum((tapered1 > 0) & (tapered1 < 1))
        n_intermediate2 = np.sum((tapered2 > 0) & (tapered2 < 1))
        
        assert n_intermediate2 > n_intermediate1, "Higher taper fraction should create more intermediate values"


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Required dependencies not available")
class TestGapMaskGeneratorImport:
    """Test that GapMaskGenerator from lisaglitch can be imported."""

    def test_gap_mask_generator_import(self):
        """Test that GapMaskGenerator can be imported from lisagap."""
        assert GapMaskGenerator is not None

    def test_gap_mask_generator_is_from_lisaglitch(self):
        """Test that GapMaskGenerator comes from lisaglitch."""
        # This test verifies that we're using the external dependency
        assert hasattr(GapMaskGenerator, '__module__')
        assert 'lisaglitch' in GapMaskGenerator.__module__
