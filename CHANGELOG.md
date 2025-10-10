# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Gap merging functionality** in `GapWindowGenerator.generate_window()` to handle closely-spaced gaps
  - New `merge_close_gaps` parameter (default: `False`)
  - New `min_freq_resolution_hz` parameter (default: `1/(3*3600)` Hz ≈ 0.0926 mHz for 3-hour minimum segments)
  - Automatically merges short data segments that would cause spectral leakage issues
  - Returns merge statistics: segments merged, duty cycle changes, additional data lost
- **DataSegmentGenerator class** for splitting data into continuous segments
- **Edge tapering functionality** with configurable left and right edge tapering
- **Advanced windowing support** in `get_time_segments()` method with `apply_window` parameter
- **One-sided Tukey window tapering** for smooth boundary transitions
- **Frequency domain analysis support** with built-in FFT capabilities
- **Comprehensive test suite** for all new segmentation and edge tapering features
- **Enhanced documentation** covering all new features and workflows

### Enhanced
- **get_time_segments() method** now supports:
  - `apply_window` (bool): Apply windowing to data segments
  - `left_edge_taper` (int): Taper samples on left edge of first segment
  - `right_edge_taper` (int): Taper samples on right edge of last segment
- **Tutorial notebook** with detailed examples of edge tapering functionality
- **API documentation** with comprehensive parameter descriptions
- **README** with advanced usage examples

### Technical Details
- **Gap merging** operates by identifying continuous data segments shorter than `1/min_freq_resolution_hz` and converting them to gaps
- Short segments between closely-spaced gaps are eliminated to prevent spectral leakage in Whittle likelihood estimation
- Edge tapering uses one-sided Tukey windows for optimal spectral properties
- Only first and last segments are affected by edge tapering
- Middle segments remain unaffected to preserve data integrity
- **BREAKING CHANGE**: `generate_window()` now returns `(mask, stats)` tuple instead of just `mask`. Set `merge_close_gaps=False` to get `stats=None` for backward compatibility.
- Full test coverage for new features (20/20 tests passing)

### Benefits
- **Gap merging prevents spectral leakage** from short segments in colored noise parameter estimation
- **Improves Whittle likelihood accuracy** by ensuring adequate frequency resolution
- **Configurable frequency thresholds** for different analysis requirements (e.g., 0.1 mHz - 1 Hz band)
- **Detailed merge statistics** for understanding data loss and duty cycle impact
- **Prevents spectral leakage** in frequency domain analysis
- **Smooth boundary transitions** reduce artifacts
- **Configurable taper lengths** for fine control
- **Preserves data integrity** while enabling clean frequency analysis
- **Independent segment analysis** for large datasets

## [0.4.0] - Previous Release

### Added
- Proportional tapering with automatic gap categorization
- Extended lobe tapering functionality
- Enhanced GapWindowGenerator capabilities
- Comprehensive tutorial notebook

### Fixed
- Various bug fixes and performance improvements

## [0.3.0] - Previous Release

### Added
- Initial release with basic gap generation
- Integration with lisaglitch package
- HDF5 save/load functionality