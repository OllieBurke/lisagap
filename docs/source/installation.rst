Installation
============

Requirements
------------

* Python >= 3.8
* numpy
* scipy
* h5py

Available on [Test PyPI](https://test.pypi.org/project/lisa-gap/)

.. code-block:: bash
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple lisa-gap==0.3.4b0

From PyPI (Recommended)
-----------------------

.. code-block:: bash

   pip install lisa-gap

GPU Support (Optional)
----------------------

For GPU acceleration using CuPy, choose the appropriate CUDA version for your system:

**Using pip:**

.. code-block:: bash

   # For CUDA 11.x (Linux x86_64 only)
   pip install lisa-gap[cuda11x]

   # For CUDA 12.x (Linux x86_64 only)  
   pip install lisa-gap[cuda12x]

   # Auto-detect CUDA version (may require compilation)
   pip install lisa-gap[gpu]

The CUDA-specific installations include both cupy and fastemriwaveforms with matching CUDA versions, 
but are only available on Linux x86_64 systems.

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

To verify your installation, run:

.. code-block:: python

   import lisagap
   print(lisagap.__version__)

To test GPU functionality (if installed):

.. code-block:: python

   from lisagap import GapMaskGenerator
   import numpy as np
   
   # Quick test
   sim_t = np.arange(0, 3600, 1.0)  # 1 hour, 1-second sampling
   gap_definitions = {
       "planned": {"maintenance": {"rate_per_year": 1, "duration_hr": 1}},
       "unplanned": {"failure": {"rate_per_year": 1, "duration_hr": 0.5}}
   }
   
   # Test CPU
   gap_gen_cpu = GapMaskGenerator(sim_t, 1.0, gap_definitions, use_gpu=False)
   mask_cpu = gap_gen_cpu.generate_mask()
   print(f"CPU test: Generated mask with {len(mask_cpu)} points")
   
   # Test GPU (if available)
   try:
       gap_gen_gpu = GapMaskGenerator(sim_t, 1.0, gap_definitions, use_gpu=True)
       mask_gpu = gap_gen_gpu.generate_mask()
       print(f"GPU test: Generated mask with {len(mask_gpu)} points")
       print("GPU acceleration is working!")
   except Exception as e:
       print(f"GPU not available: {e}")

Optional Dependencies
---------------------

**For GPU acceleration (Linux x86_64 only), choose one of:**

* **CUDA 12.x**: ``pip install lisa-gap[cuda12x]`` (includes cupy-cuda12x and fastemriwaveforms-cuda12x)
* **CUDA 11.x**: ``pip install lisa-gap[cuda11x]`` (includes cupy-cuda11x and fastemriwaveforms-cuda11x)
* **Auto-detect**: ``pip install lisa-gap[gpu]`` (generic cupy, may require compilation)
