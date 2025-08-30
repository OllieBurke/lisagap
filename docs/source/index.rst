lisa-gap Documentation
======================

**lisa-gap** is a Python package for simulating planned and unplanned data gaps in LISA time series data.

The package provides tools for generating realistic gap masks that can be applied to LISA time series data. 
It supports both planned gaps (e.g., scheduled maintenance) and unplanned gaps (e.g., hardware failures) 
with configurable statistical distributions.

The package includes advanced features for smooth tapering around gap edges using customizable Tukey windows, 
which is particularly useful for frequency domain analysis where sharp discontinuities can introduce spectral artifacts.

.. note::
   Original code developed by Eleonora Castelli (NASA Goddard) and adapted, packaged and GPU accelerated 
   by Ollie Burke (University of Glasgow).

Features
--------

* Generate realistic gap patterns for LISA time series
* Support for both planned and unplanned gaps  
* Configurable gap rates and durations
* Flexible smooth tapering with freedom to choose lobe lengths in units of time. 
* CPU/GPU agnostic operation with automatic fallback
* Save/load gap configurations to/from HDF5 files
* Easy integration with existing LISA data analysis pipelines

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
