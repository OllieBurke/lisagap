API Reference
=============

This page contains the API reference for all public classes and functions in the lisa-gap package.

Core Classes
------------

GapMaskGenerator
~~~~~~~~~~~~~~~~

.. autoclass:: lisagap.GapMaskGenerator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

GapWindowGenerator
~~~~~~~~~~~~~~~~~~

.. autoclass:: lisagap.GapWindowGenerator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

DataSegmentGenerator
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lisagap.DataSegmentGenerator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Key Methods and Features
------------------------

Edge Tapering
~~~~~~~~~~~~~

The ``DataSegmentGenerator.get_time_segments()`` method now supports advanced edge tapering for spectral analysis:

* **apply_window** (bool): Apply windowing to data segments
* **left_edge_taper** (int): Number of samples for left edge tapering on first segment
* **right_edge_taper** (int): Number of samples for right edge tapering on last segment

Edge tapering uses one-sided Tukey windows to smoothly ramp data from 0 to full amplitude (left edge) or from full amplitude to 0 (right edge), preventing spectral leakage in frequency domain analysis.

Proportional Tapering
~~~~~~~~~~~~~~~~~~~~~

The ``GapWindowGenerator.apply_proportional_tapering()`` static method provides intelligent gap tapering:

* Automatically detects gaps in mask data
* Applies different taper fractions based on gap duration (short/medium/long)
* Uses optimal Tukey window parameters for smooth transitions
* Preserves data integrity while reducing spectral artifacts

Utility Functions
-----------------

The following utility functions are available for internal use but may be useful for advanced users:

.. automodule:: lisagap.gap_mask_generator
   :members: _get_array_module, _to_numpy, _to_device
   :undoc-members:
