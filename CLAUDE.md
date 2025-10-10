# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Recommended: Install with uv (fast Python package manager)
uv sync --extra dev --extra docs

# Alternative: Traditional method
python -m venv gap_env
source gap_env/bin/activate  # On Windows: gap_env\Scripts\activate
pip install -e ".[dev,docs]"
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=lisagap

# Run specific test file
uv run python -m pytest tests/test_gap_window_generator.py

# Run specific test
uv run python -m pytest tests/test_gap_window_generator.py::TestGapWindowGenerator::test_basic_functionality
```

### Code Quality
```bash
# Format code
uv run black lisagap/ tests/

# Type checking
uv run mypy lisagap/

# Linting
uv run pylint lisagap/
uv run flake8 lisagap/

# All quality checks in sequence
uv run black lisagap/ tests/ && uv run mypy lisagap/ && uv run pylint lisagap/ && uv run flake8 lisagap/
```

### Documentation
```bash
# Build documentation locally
cd docs/
uv run python -m sphinx build source build
# Or: make html

# View documentation
open build/index.html  # macOS
# Or navigate to docs/build/index.html in browser
```

### Package Building
```bash
# Build wheel and source distribution
uv build

# Build and upload to test PyPI
uv build && uv run twine upload --repository testpypi dist/*

# Build and upload to PyPI
uv build && uv run twine upload dist/*
```

## Architecture Overview

### Core Design Pattern
This package uses a **wrapper/decorator pattern** around the `lisaglitch.GapMaskGenerator` to add advanced tapering functionality:

- **Base Layer**: `lisaglitch.GapMaskGenerator` handles gap generation and basic masking
- **Enhancement Layer**: `lisagap.GapWindowGenerator` adds sophisticated Tukey window tapering
- **Analysis Layer**: `lisagap.DataSegmentGenerator` segments data for continuous analysis

### Main Components

#### GapWindowGenerator (`lisagap/gap_window_generator.py`)
- **Purpose**: Enhanced gap mask processing with customizable smooth tapering
- **Key Methods**:
  - `generate_window()` - Main interface for gap mask generation with optional tapering
  - `apply_smooth_taper_to_mask()` - Apply Tukey window tapering to existing masks
  - `apply_proportional_tapering()` - Static method for proportional tapering based on gap duration
- **Design**: Wraps `GapMaskGenerator` and delegates core functionality while adding tapering features

#### DataSegmentGenerator (`lisagap/gap_segment_generator.py`)
- **Purpose**: Segment time series data into continuous chunks for separate analysis
- **Key Methods**:
  - `get_time_segments()` - Extract continuous data segments
  - `get_freq_info_from_segments()` - Compute FFT for each segment
  - `from_gap_generator()` - Class method to create from `GapWindowGenerator`
- **Use Case**: Alternative to tapering when discrete analysis of continuous segments is preferred

### Dependencies and Integration
- **Core Dependency**: `lisaglitch` (provides `GapMaskGenerator`)
- **External Dependencies**: `lisaconstants` (LISA mission parameters)
- **Scientific Stack**: NumPy, SciPy (especially `scipy.signal.windows.tukey`)
- **Data I/O**: h5py for HDF5 file operations

### Configuration Structure
Gap definitions follow a hierarchical structure:
```python
gap_definitions = {
    "planned": {
        "maintenance": {"rate_per_year": 12, "duration_hr": 2.0},
        "antenna_repointing": {"rate_per_year": 26, "duration_hr": 3.3}
    },
    "unplanned": {
        "hardware_failure": {"rate_per_year": 4, "duration_hr": 0.5},
        "safe_mode": {"rate_per_year": 3, "duration_hr": 60}
    }
}
```

Taper definitions mirror this structure:
```python
taper_definitions = {
    "planned": {
        "maintenance": {"lobe_lengths_hr": 1.0}
    },
    "unplanned": {
        "hardware_failure": {"lobe_lengths_hr": 0.5}
    }
}
```

## Project Structure

```
lisagap/
├── __init__.py                 # Package initialization, imports main classes
├── gap_window_generator.py     # Main tapering functionality
├── gap_segment_generator.py    # Data segmentation for continuous analysis
└── py.typed                    # Type hints marker file

tests/
├── test_gap_window_generator.py     # Tests for tapering functionality
└── test_data_segment_generator.py   # Tests for segmentation functionality

docs/
├── source/                     # Sphinx documentation source
├── build/                      # Built documentation (generated)
└── requirements.txt            # Documentation dependencies
```

## Development Conventions

### Code Style
- **Formatting**: Black (line length 88)
- **Type Hints**: Full typing throughout, validated with mypy
- **Docstrings**: NumPy-style docstrings with examples
- **Imports**: Standard scientific Python conventions

### Testing Strategy
- **Framework**: pytest with coverage reporting
- **Structure**: Mirror package structure in tests/
- **Coverage**: Aim for high coverage, especially for public APIs
- **Test Data**: Use synthetic data generation for reproducible tests

### Type Safety
- Package includes `py.typed` marker for type checkers
- All public APIs have complete type annotations
- mypy configuration in `pyproject.toml` enforces strict typing

### Error Handling
- Validate inputs early with clear error messages
- Use appropriate exception types (ValueError, TypeError)
- Provide helpful guidance in error messages for common mistakes

## LISA Domain Knowledge

### Scientific Context
- **LISA**: Laser Interferometer Space Antenna - future space-based gravitational wave detector
- **Data Gaps**: Planned (maintenance, pointing) and unplanned (hardware failures) interruptions
- **Tapering**: Critical for frequency domain analysis to reduce spectral artifacts from sharp discontinuities

### Typical Use Cases
1. **Simulation**: Generate realistic gap patterns for LISA data processing pipeline testing
2. **Analysis**: Apply smooth tapering for frequency domain studies
3. **Segmentation**: Analyze continuous data chunks separately when tapering isn't suitable

### Gap Categories
- **Short gaps** (< 10 minutes): Aggressive tapering (25% each side)
- **Medium gaps** (10 minutes - 10 hours): Moderate tapering (5% each side)
- **Long gaps** (> 10 hours): Conservative tapering (5% each side)