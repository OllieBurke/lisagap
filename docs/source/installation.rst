Installation
============

Requirements
------------

* Python >= 3.10
* numpy
* scipy  
* h5py
* lisaglitch
* lisaconstants

From PyPI (Recommended)
-----------------------

.. code-block:: bash

   pip install lisa-gap

Using uv (Faster)
-----------------

.. code-block:: bash

   uv add lisa-gap

From Source
-----------

.. code-block:: bash

   git clone https://github.com/ollieburke/lisa-gap.git
   cd lisa-gap
   pip install .

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/ollieburke/lisa-gap.git
   cd lisa-gap
   pip install -e .[dev]

This installs the package in development mode with additional tools for testing and development.

Verification
------------

Test your installation:

.. code-block:: python

   import lisagap
   from lisaglitch import GapMaskGenerator
   from lisagap import GapWindowGenerator
   
   print("lisa-gap successfully installed!")
   print(f"Version: {lisagap.__version__}")
