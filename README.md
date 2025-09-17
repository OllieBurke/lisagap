# lisa-gap

A Python package for simulating planned and unplanned data gaps in LISA time series data. Our package is currently available on test.pypi [here](https://test.pypi.org/project/lisa-gap/).

The work here builds off the work in the `lisaglitch` package, which provides core functionality for generating gap masks. `lisa-gap` extends this functionality with advanced features such as customizable smooth tapering around gap edges using Tukey windows, data segmentation with edge tapering, and proportional tapering for frequency domain analysis.

## Description

`lisa-gap` provides tools for generating realistic gap masks that can be applied to LISA time series data. The package supports both planned gaps (e.g., scheduled maintenance) and unplanned gaps (e.g., hardware failures) with configurable statistical distributions.

The package includes advanced features for:
- **Smooth tapering** around gap edges using customizable Tukey windows
- **Data segmentation** with edge tapering for spectral analysis
- **Proportional tapering** with automatic gap categorization
- **Edge boundary management** to prevent spectral artifacts

These features are particularly useful for frequency domain analysis where sharp discontinuities can introduce spectral artifacts.

## Installation

### Recommended: Using uv (fastest and most reliable)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that simplifies dependency management:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install the package
git clone https://github.com/ollieburke/lisa-gap.git
cd lisa-gap
uv sync
```

To use the package:
```bash
uv run python your_script.py
```

### Alternative: Traditional virtual environment

If you prefer using the standard Python virtual environment:

```bash
python -m venv gap_env
source gap_env/bin/activate  # On Windows use `gap_env\Scripts\activate`
```

Then install the package:
```bash
pip install lisa-gap
```

Available on [Test PyPI](https://test.pypi.org/project/lisa-gap/):
```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple lisa-gap==0.4.0
```

### From source (traditional method)

```bash
git clone https://github.com/ollieburke/lisa-gap.git
cd lisa-gap
pip install .
```

### Development installation

**Recommended with uv:**
```bash
git clone https://github.com/ollieburke/lisa-gap.git
cd lisa-gap
uv sync --extra dev --extra docs
```

**Traditional method:**
```bash
git clone https://github.com/ollieburke/lisa-gap.git
cd lisa-gap
pip install -e ".[dev, docs]"
```

### Verify installation with tests

**With uv:**
```bash
uv run pytest
```

**Traditional:**
```bash
pytest 

## Quick Start

```python
from lisaglitch import GapMaskGenerator
from lisagap import GapWindowGenerator, DataSegmentGenerator
import numpy as np

# Create time array
dt = 5.0  # seconds
duration = 86400  # 1 day in seconds
sim_t = np.arange(0, duration, dt)

# Define gap configuration
gap_definitions = {
    "planned": {
        "maintenance": {
            "rate_per_year": 12,  # 12 times per year
            "duration_hr": 2.0    # 2 hours each
        }
    },
    "unplanned": {
        "hardware_failure": {
            "rate_per_year": 4,   # 4 times per year
            "duration_hr": 0.5    # 30 minutes each
        }
    }
}

# Create gap mask generator
gap_gen = GapMaskGenerator(
    sim_t=sim_t,
    gap_definitions=gap_definitions,
    treat_as_nan=False
)

# Create window generator for advanced features
window = GapWindowGenerator(gap_gen)

# Generate gap mask with proportional tapering
gap_mask = GapWindowGenerator.apply_proportional_tapering(
    window.generate_mask(),
    dt=dt,
    short_taper_fraction=0.25,   # 25% each side for short gaps
    medium_taper_fraction=0.05,  # 5% each side for medium gaps
    long_taper_fraction=0.02     # 2% each side for long gaps
)

# Generate data stream
data = np.sin(2*np.pi * 0.01 * sim_t) + 0.1 * np.random.randn(len(sim_t))

# Option 1: Apply gaps directly
data_w_gaps = data * gap_mask

# Option 2: Use segmentation for independent analysis
segmenter = DataSegmentGenerator(
    mask=gap_mask,
    data=data,
    dt=dt
)

# Get segments with edge tapering for frequency analysis
segments = segmenter.get_time_segments(
    apply_window=True,
    left_edge_taper=1000,   # Taper first segment left edge
    right_edge_taper=1000   # Taper last segment right edge
)

# Analyze frequency content of each segment
freq_info = segmenter.get_freq_info_from_segments()
```
gap_definitions = {
    "planned": {
        "maintenance": {
            "rate_per_year": 12,  # 12 times per year
            "duration_hr": 2.0    # 2 hours each
        }
    },
    "unplanned": {
        "hardware_failure": {
            "rate_per_year": 4,   # 4 times per year
            "duration_hr": 0.5    # 30 minutes each
        }
    }
}

# Create gap mask generator
gap_gen = GapMaskGenerator(
    sim_t=sim_t,
    gap_definitions=gap_definitions,
    treat_as_nan=True
)

# Create window generator for advanced features
window = GapWindowGenerator(gap_gen)

# Generate gap mask
gap_mask = window.generate_mask()

# Generate data stream
data = np.sin(2*np.pi * sim_t) # Generate fake data
# Apply gaps to your data
data_w_gaps = data * gap_mask
```

## Documentation

Full documentation is available at [Read the Docs](https://lisagap.readthedocs.io/).

### Tutorial Notebook

See the included **`docs/source/gap_notebook.ipynb`** for a comprehensive tutorial that covers:

- Setting up realistic gap configurations for LISA
- Generating gap masks with planned and unplanned gaps  
- Saving and loading gap configurations to/from HDF5 files. Gap metadata can be saved to .json files
- **Customizable smooth tapering** for frequency domain analysis
- **Proportional tapering** with automatic gap categorization
- **Data segmentation** with edge tapering for spectral analysis
- **Advanced windowing** techniques for boundary artifact prevention
- Quality flag generation and gap analysis

The tutorial notebook can be viewed within the documentation.

## Advanced Features

### Data Segmentation with Edge Tapering

Split data into continuous segments and apply edge tapering to prevent spectral artifacts:

```python
from lisagap import DataSegmentGenerator

# Create segmenter
segmenter = DataSegmentGenerator(mask=gap_mask, data=your_data, dt=dt)

# Get segments with edge tapering for frequency analysis
segments = segmenter.get_time_segments(
    apply_window=True,
    left_edge_taper=1000,   # Smooth left edge of first segment
    right_edge_taper=1500   # Smooth right edge of last segment  
)

# Analyze frequency content
freq_info = segmenter.get_freq_info_from_segments()
```

### Proportional Tapering

Automatically categorize gaps and apply proportional tapering:

```python
# Apply smart tapering based on gap duration
tapered_mask = GapWindowGenerator.apply_proportional_tapering(
    gap_mask,
    dt=dt,
    short_taper_fraction=0.25,   # 25% of gap length for short gaps
    medium_taper_fraction=0.05,  # 5% of gap length for medium gaps
    long_taper_fraction=0.02,    # 2% of gap length for long gaps
    short_gap_threshold_minutes=30,   # <30 min = short gap
    long_gap_threshold_hours=2        # >2 hours = long gap
)
```
**Flexible Taper Control**: Users have complete freedom to choose their own tapering strategy for each gap type:

```python
from lisaglitch import GapMaskGenerator
from lisagap import GapWindowGenerator
import numpy as np

# Set up gap configuration
gap_definitions = {
    "planned": {
        "antenna repointing": {"rate_per_year": 26, "duration_hr": 3.3},
        "TM stray potential": {"rate_per_year": 2, "duration_hr": 24}
    },
    "unplanned": {
        "platform safe mode": {"rate_per_year": 3, "duration_hr": 60},
        "QPD loss micrometeoroid": {"rate_per_year": 5, "duration_hr": 24}
    }
}

# Generate gap mask
gap_gen = GapMaskGenerator(sim_t, gap_definitions)
window = GapWindowGenerator(gap_gen)
gap_mask = window.generate_mask()

# Define custom tapering per gap type (lobe lengths in hours)
taper_definitions = {
    "planned": {
        "antenna repointing": {"lobe_lengths_hr": 5.0},   # Long taper for repointing
        "TM stray potential": {"lobe_lengths_hr": 0.5}    # Short taper for TM events
    },
    "unplanned": {
        "platform safe mode": {"lobe_lengths_hr": 1.0},  # Medium taper for safe mode
        "QPD loss micrometeoroid": {"lobe_lengths_hr": 2.0}  # Custom taper for QPD loss
    }
}

# Apply smooth Tukey window tapering
smoothed_mask = window.apply_smooth_taper_to_mask(
    gap_mask, 
    taper_definitions
)
```

This flexibility allows users to optimize tapering strategies for different gap types based on their specific analysis requirements, whether working in time or frequency domains.

## Features

- Generate realistic gap patterns for LISA time series
- Support for both planned and unplanned gaps
- Configurable gap rates and durations
- **Flexible smooth tapering with user-defined Tukey windows per gap type**
- **Proportional tapering with automatic gap categorization**
- **Data segmentation with edge tapering for spectral analysis**
- **Advanced boundary management to prevent frequency artifacts**
- **Complete freedom to customize taper lengths for different gap categories**
- Built on top of the robust `lisaglitch` package for core gap generation
- Save/load gap configurations to/from HDF5 files
- Quality flag generation and gap analysis tools
- Comprehensive documentation and tutorial notebooks

## Requirements

- Python ≥ 3.10
- numpy
- scipy
- h5py
- lisaglitch
- lisaconstants 

## License

MIT License

## Citation

If you use this package in your research, please cite:

```bibtex
@software{lisagap,
  author = {Burke, Ollie and Castelli, Eleonora},
  title = {lisa-gap: A tool for simulating data gaps in LISA time series},
  url = {https://github.com/ollieburke/lisa-gap},
  version = {0.1.0},
  year = {2025}
}
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](https://lisagap.readthedocs.io/en/latest/contributing.html) for detailed information on how to contribute to the project.

## Support

This gap generation tool is suitable for LISA data processing pipelines including L01, SIM, L2D, and L2A data products. 
