# lisa-gap

A Python package for simulating planned and unplanned data gaps in LISA time series data. Our package is currently available on test.pypi [here](https://test.pypi.org/project/lisa-gap/).

The work here builds off the work in the `lisaglitch` package, which provides core functionality for generating gap masks. `lisa-gap` extends this functionality with advanced features such as customizable smooth tapering around gap edges using Tukey windows, making it particularly suitable for frequency domain analysis.

## Description

`lisa-gap` provides tools for generating realistic gap masks that can be applied to LISA time series data. The package supports both planned gaps (e.g., scheduled maintenance) and unplanned gaps (e.g., hardware failures) with configurable statistical distributions.

The package includes advanced features for smooth tapering around gap edges using customizable Tukey windows, which is particularly useful for frequency domain analysis where sharp discontinuities can introduce spectral artifacts.

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
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple lisa-gap==0.3.4
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
uv sync --dev
```

**Traditional method:**
```bash
git clone https://github.com/ollieburke/lisa-gap.git
cd lisa-gap
pip install -e ".[dev]"
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
from lisagap import GapWindowGenerator
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
- Quality flag generation and gap analysis

The tutorial notebook can be viewed within the documentation [here]
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
- **Complete freedom to customize taper lengths for different gap categories**
- Built on top of the robust `lisaglitch` package for core gap generation
- Save/load gap configurations to/from HDF5 files
- Quality flag generation and gap analysis tools
- Comprehensive documentation

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
