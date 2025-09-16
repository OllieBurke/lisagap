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
