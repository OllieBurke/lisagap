# lisa-gap

A Python package for simulating planned and unplanned data gaps in LISA time series data.

## Description

`lisa-gap` provides tools for generating realistic gap masks that can be applied to LISA time series data. The package supports both planned gaps (e.g., scheduled maintenance) and unplanned gaps (e.g., hardware failures) with configurable statistical distributions.

The package includes advanced features for smooth tapering around gap edges using customizable Tukey windows, which is particularly useful for frequency domain analysis where sharp discontinuities can introduce spectral artifacts.

Original code developed by Eleonora Castelli (NASA Goddard) and adapted by Ollie Burke (University of Glasgow).

## Installation

### From PyPI (recommended)

```bash
pip install lisa-gap
```

### With GPU support (optional)

**Using pip:**
```bash
# For CUDA 11.x (Linux x86_64 only)
pip install lisa-gap[cuda11x]

# For CUDA 12.x (Linux x86_64 only)  
pip install lisa-gap[cuda12x]

# Auto-detect CUDA version (may require compilation)
pip install lisa-gap[gpu]
```

### From source

```bash
git clone https://github.com/ollieburke/lisa-gap.git
cd lisa-gap
pip install .
```

### Development installation

```bash
git clone https://github.com/ollieburke/lisa-gap.git
cd lisa-gap
pip install -e .[dev]
```

## Quick Start

```python
from lisa_gap import GapMaskGenerator
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

# Create gap mask generator (CPU)
gap_gen = GapMaskGenerator(
    sim_t=sim_t,
    dt=dt,
    gap_definitions=gap_definitions,
    treat_as_nans=True,
    use_gpu=False  # Set to True for GPU acceleration
)

# Generate gap mask
gap_mask = gap_gen.generate_mask()

# Generate data stream
data = np.sin(2*np.pi * sim_t) # Generate fake data
# Apply gaps to your data
data_w_gaps = data * gap_mask
```

### GPU Acceleration

For large datasets, you can enable GPU acceleration (requires CUDA installation on Linux x86_64):

```python
# Enable GPU acceleration (requires CUDA 11.x or 12.x on Linux x86_64)
gap_gen = GapMaskGenerator(
    sim_t=sim_t,
    dt=dt,
    gap_definitions=gap_definitions,
    use_gpu=True  # Automatically falls back to CPU if CuPy unavailable
)
```

## Documentation

Full documentation is available at [Read the Docs](https://lisa-gap.readthedocs.io/).

### Tutorial Notebook

See the included **`gap_notebook.ipynb`** for a comprehensive tutorial that covers:

- Setting up realistic gap configurations for LISA
- Generating gap masks with planned and unplanned gaps
- Saving and loading gap configurations to/from HDF5 files
- **Customizable smooth tapering** for frequency domain analysis
- Quality flag generation and gap analysis

The tutorial demonstrates real-world usage patterns and provides examples for different LISA data processing levels (L01, L2A, L2D).

### Key Features Highlighted

**Flexible Taper Control**: Users have complete freedom to choose their own tapering strategy for each gap type:

```python
from lisa_gap import GapMaskGenerator
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
gap_gen = GapMaskGenerator(sim_t, dt, gap_definitions)
gap_mask = gap_gen.generate_mask()

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
smoothed_mask = gap_gen.apply_smooth_taper_to_mask(
    gap_mask, 
    taper_gap_definitions=taper_definitions
)
```

This flexibility allows users to optimize tapering strategies for different gap types based on their specific analysis requirements, whether working in time or frequency domains.

## Features

- Generate realistic gap patterns for LISA time series
- Support for both planned and unplanned gaps
- Configurable gap rates and durations
- Statistical distributions for gap timing
- **Flexible smooth tapering with user-defined Tukey windows per gap type**
- **Complete freedom to customize taper lengths for different gap categories**
- **GPU acceleration support with CuPy for large datasets**
- **CPU/GPU agnostic operation with automatic fallback**
- Save/load gap configurations to/from HDF5 files
- Quality flag generation and gap analysis tools
- Easy integration with existing LISA data analysis pipelines

## Requirements

- Python ≥ 3.8
- numpy
- scipy
- h5py
- lisaconstants

### Optional dependencies

**For GPU acceleration (Linux x86_64 only), choose one of:**
- **CUDA 12.x**: `pip install lisa-gap[cuda12x]` (includes cupy-cuda12x and fastemriwaveforms-cuda12x)
- **CUDA 11.x**: `pip install lisa-gap[cuda11x]` (includes cupy-cuda11x and fastemriwaveforms-cuda11x)
- **Auto-detect**: `pip install lisa-gap[gpu]` (generic cupy, may require compilation)

## License

MIT License

## Citation

If you use this package in your research, please cite:

```bibtex
@software{lisa_gap,
  author = {Burke, Ollie and Castelli, Eleonora},
  title = {lisa-gap: A tool for simulating data gaps in LISA time series},
  url = {https://github.com/ollieburke/lisa-gap},
  version = {0.1.0},
  year = {2025}
}
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](https://lisa-gap.readthedocs.io/en/latest/contributing.html) for detailed information on how to contribute to the project.

## Support

This gap generation tool is suitable for LISA data processing pipelines including L01, SIM, L2D, and L2A data products. 