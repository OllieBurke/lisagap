# lisa-gap

A Python package for simulating planned and unplanned data gaps in LISA time series data.

## Description

`lisa-gap` provides tools for generating realistic gap masks that can be applied to LISA time series data. The package supports both planned gaps (e.g., scheduled maintenance) and unplanned gaps (e.g., hardware failures) with configurable statistical distributions.

Original code developed by Eleonora Castelli (NASA Goddard) and adapted by Ollie Burke (University of Glasgow).

## Installation

### From PyPI (recommended)

```bash
pip install lisa-gap
```

### With GPU support (optional)

For GPU acceleration using CuPy:

```bash
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
dt = 1.0  # seconds
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
    use_gpu=False  # Set to True for GPU acceleration
)

# Generate gap mask
gap_mask = gap_gen.generate_mask()

# Apply gaps to your data
data_with_gaps = your_data.copy()
data_with_gaps[gap_mask] = np.nan
```

### GPU Acceleration

For large datasets, you can enable GPU acceleration:

```python
# Enable GPU acceleration (requires CuPy)
gap_gen = GapMaskGenerator(
    sim_t=sim_t,
    dt=dt,
    gap_definitions=gap_definitions,
    use_gpu=True  # Automatically falls back to CPU if CuPy unavailable
)
```

## Documentation

See the included `gap_notebook.ipynb` for detailed examples and usage patterns.

## Features

- Generate realistic gap patterns for LISA time series
- Support for both planned and unplanned gaps
- Configurable gap rates and durations
- Statistical distributions for gap timing
- **GPU acceleration support with CuPy for large datasets**
- **CPU/GPU agnostic operation with automatic fallback**
- Easy integration with existing LISA data analysis pipelines

## Requirements

- Python ≥ 3.8
- numpy
- scipy
- h5py
- lisaconstants

### Optional dependencies

- **cupy**: For GPU acceleration (install with `pip install lisa-gap[gpu]`)

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

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

This gap generation tool is suitable for LISA data processing pipelines including L01, SIM, L2D, and L2A data products. 