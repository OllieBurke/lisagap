Tutorial
========

This tutorial demonstrates the key functionality of the lisa-gap package through a comprehensive example.
The tutorial covers setting up realistic gap configurations for LISA, generating gap masks, and applying
smooth tapering for frequency domain analysis.

Complete Tutorial Notebook
---------------------------

The following notebook shows a complete workflow using lisa-gap:

.. toctree::
   :maxdepth: 1

   gap_notebook.ipynb

Quick Start Example
-------------------

Here's a minimal example to get you started:

.. code-block:: python

   import numpy as np
   from lisa_gap import GapMaskGenerator
   from lisaconstants import TROPICALYEAR_J2000DAY

   # Set up simulation properties
   A_YEAR = TROPICALYEAR_J2000DAY * 86400  # seconds in a year
   dt = 0.25  # seconds
   t_obs = 0.5 * A_YEAR  # 6 months
   sim_t = np.arange(0, t_obs, dt)

   # Define realistic gap configuration
   gap_definitions = {
       "planned": {
           "antenna repointing": {"rate_per_year": 26, "duration_hr": 3.3},
           "TM stray potential": {"rate_per_year": 2, "duration_hr": 24},
           "TTL calibration": {"rate_per_year": 4, "duration_hr": 48},
       },
       "unplanned": {
           "platform safe mode": {"rate_per_year": 3, "duration_hr": 60},
           "payload safe mode": {"rate_per_year": 4, "duration_hr": 66},
           "QPD loss micrometeoroid": {"rate_per_year": 5, "duration_hr": 24},
           "HR GRS loss micrometeoroid": {"rate_per_year": 19, "duration_hr": 24},
           "WR GRS loss micrometeoroid": {"rate_per_year": 6, "duration_hr": 24},
       }
   }

   # Create gap mask generator
   gap_gen = GapMaskGenerator(
       sim_t, 
       dt, 
       gap_definitions, 
       treat_as_nan=False, # Treat as gating function (zeros for gap segment)
       use_gpu=False  # Set to True for GPU acceleration
   )

   # Generate gap mask
   gap_mask = gap_gen.generate_mask(include_unplanned=True, include_planned=True)

   # Calculate duty cycle
   duty_cycle = 100 * (1 - np.sum(gap_mask == 0) / len(gap_mask))
   print(f"Duty cycle: {duty_cycle:.2f}%")

Advanced Features
-----------------

Smooth Tapering
~~~~~~~~~~~~~~~

Apply smooth Tukey window tapering around gaps to reduce spectral artifacts:

.. code-block:: python

   # Define custom tapering per gap type
   taper_definitions = {
       "planned": {
           "antenna repointing": {"lobe_lengths_hr": 5.0},
           "TM stray potential": {"lobe_lengths_hr": 0.5},
           "TTL calibration": {"lobe_lengths_hr": 2.0}
       },
       "unplanned": {
           "platform safe mode": {"lobe_lengths_hr": 1.0},
           "QPD loss micrometeoroid": {"lobe_lengths_hr": 1.0},
           "HR GRS loss micrometeoroid": {"lobe_lengths_hr": 7.0},
           "WR GRS loss micrometeoroid": {"lobe_lengths_hr": 10.0}
       }
   }

   # Apply smooth tapering
   smoothed_mask = gap_gen.apply_smooth_taper_to_mask(
       gap_mask, 
       taper_gap_definitions=taper_definitions
   )

Save and Load Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save gap configurations and masks for later use:

.. code-block:: python

   # Save to HDF5 file
   gap_gen.save_to_hdf5(gap_mask, filename="gap_mask_data.h5")

   # Load from HDF5 file
   loaded_gap_gen = GapMaskGenerator.from_hdf5(filename="gap_mask_data.h5")
   regenerated_mask = loaded_gap_gen.generate_mask()

Gap Analysis
~~~~~~~~~~~~

Generate summary statistics and quality flags:

.. code-block:: python

   # Generate summary
   summary = gap_gen.summary(mask=gap_mask, export_json_path="gap_summary.json")
   print(summary)

   # Build quality flags for data analysis
   quality_flags = gap_gen.build_quality_flags(gap_mask)

GPU Acceleration
~~~~~~~~~~~~~~~~

For large datasets, enable GPU acceleration:

.. code-block:: python

   # Enable GPU acceleration (requires CuPy)
   gap_gen_gpu = GapMaskGenerator(
       sim_t, 
       dt, 
       gap_definitions,
       use_gpu=True  # Automatically falls back to CPU if CuPy unavailable
   )

   gap_mask_gpu = gap_gen_gpu.generate_mask()

The tutorial notebook provides detailed examples of all these features with realistic LISA operational scenarios.
