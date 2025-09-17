lisa-gap Documentation
======================

**lisa-gap** is a Python package for simulating planned and unplanned data gaps in LISA time series data.

The package provides tools for generating realistic gap masks that can be applied to LISA time series data. 
It supports both planned gaps (e.g., scheduled maintenance) and unplanned gaps (e.g., hardware failures) 
with configurable statistical distributions.

Built on top of `lisaglitch <https://github.com/LISA-Consortium/lisaglitch>`_, lisa-gap adds advanced 
windowing, proportional tapering, and data segmentation capabilities for frequency domain analysis.

.. note::
   Original code developed by Eleonora Castelli (NASA Goddard) and adapted, packaged and enhanced 
   by Ollie Burke (University of Glasgow).

Features
--------

* Generate realistic gap patterns for LISA time series
* Support for both planned and unplanned gaps  
* Configurable gap rates and durations
* **Proportional tapering** with automatic gap categorization
* **Extended lobe tapering** for ultra-smooth transitions
* **Data segmentation** with edge tapering for spectral analysis
* **Advanced windowing** with boundary artifact prevention
* Built on proven `lisaglitch` foundation
* Save/load gap configurations to/from HDF5 files

Core Classes
------------

* **GapMaskGenerator** - Generate gap masks from statistical definitions
* **GapWindowGenerator** - Apply proportional and traditional tapering
* **DataSegmentGenerator** - Segment data and apply edge tapering for frequency analysis

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial
   api
   examples
   contributing
   citation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
