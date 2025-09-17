Tutorial
========

This tutorial demonstrates the key functionality of lisa-gap, focusing on the new proportional tapering 
capabilities built on top of lisaglitch.

Quick Start Example
-------------------

Basic gap mask generation:

.. code-block:: python

   import numpy as np
   from lisaglitch import GapMaskGenerator
   from lisagap import GapWindowGenerator
   from lisaconstants import TROPICALYEAR_J2000DAY

   # Set up simulation properties
   A_YEAR = TROPICALYEAR_J2000DAY * 86400
   dt = 0.25  # seconds
   t_obs = 0.5 * A_YEAR  # 6 months
   sim_t = np.arange(0, t_obs, dt)

   # Define realistic gap configuration
   gap_definitions = {
       "planned": {
           "antenna repointing": {"rate_per_year": 26, "duration_hr": 3.3},
           "TM stray potential": {"rate_per_year": 2, "duration_hr": 24},
       },
       "unplanned": {
           "platform safe mode": {"rate_per_year": 3, "duration_hr": 60},
           "HR GRS loss micrometeoroid": {"rate_per_year": 19, "duration_hr": 24},
       }
   }

   # Create gap mask generator
   gap_gen = GapMaskGenerator(sim_t, gap_definitions)
   
   # Generate binary mask
   mask = gap_gen.generate_mask(include_unplanned=True, include_planned=True)

Proportional Tapering
---------------------

Apply smart tapering based on gap duration:

.. code-block:: python

   # Convert to quality flags and create mask
   quality_flags = gap_gen.build_quality_flags(mask)
   gap_mask = GapMaskGenerator.quality_flags_to_mask(quality_flags)
   
   # Apply proportional tapering
   tapered_mask = GapWindowGenerator.apply_proportional_tapering(
       gap_mask,
       dt=dt,
       short_taper_fraction=0.25,   # 25% each side for short gaps
       medium_taper_fraction=0.05,  # 5% each side for medium gaps  
       long_taper_fraction=0.02,    # 2% each side for long gaps
       short_gap_threshold_minutes=30,  # <30 min = short
       long_gap_threshold_hours=2       # >2 hours = long
   )

Extended Lobe Tapering
---------------------

Create ultra-smooth transitions with extended lobes:

.. code-block:: python

   # Extended tapering with 2.5x gap width
   extended_tapered = GapWindowGenerator.apply_proportional_tapering(
       gap_mask,
       dt=dt,
       medium_taper_fraction=0.1,
       lobe_extension_factor=2.5    # Taper window 2.5x gap width
   )

This creates much longer, more gradual transitions extending beyond gap boundaries.

Data Segmentation
-----------------

Split data into continuous segments for independent analysis:

.. code-block:: python

   from lisagap import DataSegmentGenerator
   
   # Create sample data with gaps
   sample_data = np.sin(2 * np.pi * 0.01 * sim_t) + 0.1 * np.random.randn(len(sim_t))
   
   # Create segmenter using tapered mask
   segmenter = DataSegmentGenerator(
       mask=tapered_mask,
       data=sample_data, 
       dt=dt,
       t0=0.0
   )
   
   # Get basic segmentation info
   summary = segmenter.summary()
   print(f"Found {summary['total_segments']} continuous segments")
   print(f"Data fraction valid: {summary['data_fraction_valid']:.2%}")

Advanced Segmentation with Edge Tapering
----------------------------------------

Apply edge tapering to prevent spectral artifacts at segment boundaries:

.. code-block:: python

   # Basic segmentation (preserves original data)
   segments = segmenter.get_time_segments(apply_window=False)
   
   # Apply windowing with existing mask tapering
   windowed_segments = segmenter.get_time_segments(apply_window=True)
   
   # Add edge tapering for frequency domain analysis
   edge_tapered_segments = segmenter.get_time_segments(
       apply_window=True,
       left_edge_taper=1000,   # Taper first 1000 samples of first segment
       right_edge_taper=1500   # Taper last 1500 samples of last segment
   )

Edge tapering uses one-sided Tukey windows to create smooth ramps:

* **Left edge**: Ramps from 0 to 1 over specified samples on first segment only
* **Right edge**: Ramps from 1 to 0 over specified samples on last segment only  
* **Middle segments**: Unaffected by edge tapering
* **Prevents spectral leakage**: Essential for clean frequency domain analysis

Frequency Domain Analysis
-------------------------

Analyze segments in frequency domain with proper windowing:

.. code-block:: python

   # Get frequency information for all segments
   freq_info = segmenter.get_freq_info_from_segments()
   
   # Plot power spectra
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(12, 6))
   for seg_name, seg_freq in freq_info.items():
       psd = np.abs(seg_freq['fft'])**2
       plt.loglog(seg_freq['frequencies'][1:], psd[1:], 
                  label=seg_name, alpha=0.8)
   
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('Power Spectral Density')
   plt.legend()
   plt.show()

Integration Workflows
--------------------

Create segmenters directly from gap generators:

.. code-block:: python

   # Integrated workflow
   segmenter, reusable_mask = DataSegmentGenerator.from_gap_generator(
       gap_window_generator=window_gen,
       data=sample_data,
       dt=dt,
       apply_tapering=True  # Apply windowing during mask generation
   )

This creates much longer, more gradual transitions extending beyond gap boundaries.

   # Create gap mask generator
   gap_gen = GapMaskGenerator(
       sim_t, 
       dt, 
       gap_definitions, 
       treat_as_nan=False, # Treat as gating function (zeros for gap segment)
       use_gpu=False  # Set to True for GPU acceleration
   )

Traditional Windowing
--------------------

For specific gap types, use traditional lobe-length tapering:

.. code-block:: python

   # Wrap with windowing capabilities
   window_gen = GapWindowGenerator(gap_gen)
   
   # Define custom tapering per gap type
   taper_definitions = {
       "planned": {
           "antenna repointing": {"lobe_lengths_hr": 5.0},
           "TM stray potential": {"lobe_lengths_hr": 0.5},
       },
       "unplanned": {
           "platform safe mode": {"lobe_lengths_hr": 1.0},
           "HR GRS loss micrometeoroid": {"lobe_lengths_hr": 7.0},
       }
   }

   # Apply smooth tapering
   smoothed_mask = window_gen.generate_window(
       include_planned=True,
       include_unplanned=True, 
       apply_tapering=True,
       taper_definitions=taper_definitions
   )

Save and Load
-------------

Save configurations for reproducible results:

.. code-block:: python

   # Save to HDF5
   gap_gen.save_to_hdf5(mask, filename="gap_mask_data.h5")

   # Load from HDF5
   loaded_gen = GapMaskGenerator.from_hdf5("gap_mask_data.h5")
   mask_copy = loaded_gen.generate_mask()

Complete Tutorial Notebook
---------------------------

See the complete notebook for detailed examples:

.. toctree::
   :maxdepth: 1

   gap_notebook.ipynb

Key Advantages
--------------

**Proportional Tapering Benefits:**

* **Automatic categorization** - Short, medium, long gaps handled differently
* **Proportional to gap duration** - Larger gaps get more tapering  
* **Configurable thresholds** - Customize gap categories for your use case
* **Extended lobes** - Ultra-smooth transitions for frequency analysis

**Data Segmentation Benefits:**

* **Independent analysis** - Analyze each continuous segment separately
* **Preserves data integrity** - No modification of original data
* **Edge tapering** - Prevents spectral artifacts at boundaries
* **Frequency domain ready** - Built-in FFT support with proper windowing

**Edge Tapering Benefits:**

* **Spectral cleanliness** - Prevents leakage in frequency analysis
* **Boundary-specific** - Only affects first and last segments
* **Tukey windowing** - Optimal spectral properties with smooth transitions
* **Configurable length** - User-defined taper samples for fine control

**Compared to traditional windowing:**

* **Smarter** - Adapts to gap characteristics automatically
* **More flexible** - Works with external .npy gap masks  
* **Better for FFTs** - Longer, smoother transitions reduce artifacts
* **Segmentation support** - Clean separation for independent analysis

The notebook provides detailed examples showing these features in action.
