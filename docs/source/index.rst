BIPP Documentation
===================
Image synthesis in radio astronomy is done with interferometry, a powerful technique allowing observation of the sky with antenna arrays with otherwise inaccessible angular resolutions and sensitivities. The Bluebild algorithm offers a novel approach to image synthesis, leveraging fPCA to decompose the sky image into distinct energy eigenimages. Bluebild Imaging++ is an HPC implementation of Bluebild.


Requirements
============
Bipp requires the following:

- C++17 compliant compiler
- CMake 3.11 and later
- BLAS and LAPACK library like OpenBLAS or Intel MKL
- `neonufft <https://github.com/epfl-radio-astro/neonufft>`_

Bipp can be configured with additional features (check the CMake options below). The optional requirements are:

- Python header files and `pybind11 <https://github.com/pybind/pybind11>`_ for building the Python interface
- CUDA 9.0 and later for Nvidia GPU hardware
- ROCm 5.0 and later for AMD GPU hardware
- `Umpire <https://github.com/LLNL/Umpire>`_ for advanced memory management

The Python module has the following dependencies:

- numpy
- astropy
- matplotlib
- tqdm
- pyproj
- scipy
- pandas
- healpy
- casacore


Installation
============
Bipp uses CMake to configure the build.

CMake options
-------------
Bipp can be configured with the following options:

.. list-table::
   :widths: 25 20 10 50
   :header-rows: 1

   * - Option
     - Values
     - Default
     - Description
   * - BIPP_PYTHON
     - ON, OFF
     - ON
     - Build Python interface
   * - BIPP_VC
     - ON, OFF
     - OFF
     - Use the VC library for vectorization
   * - BIPP_GPU
     - OFF, CUDA, ROCM
     - OFF
     - Select GPU backend
   * - BIPP_MAGMA
     - ON, OFF
     - OFF
     - Use MAGMA as eigensolver on GPU
   * - BIPP_BUILD_TESTS
     - ON, OFF
     - OFF
     - Build test executables
   * - BIPP_INSTALL
     - LIB, PYTHON, OFF
     - LIB
     - Set installation target
   * - BIPP_UMPIRE
     - ON, OFF
     - OFF
     - Use the UMPIRE library for memory allocations
   * - BIPP_BUNDLED_LIBS
     - ON, OFF
     - ON
     - Download and build spdlog, pybind11, googletest and json library


Some useful general CMake options are:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Option
     - Description
   * - CMAKE_PREFIX_PATH
     - Semicolon separated list of search paths for external libraries
   * - CMAKE_INSTALL_PREFIX
     - Path to installation target directory
   * - BUILD_SHARED_LIBS
     - Build shared libraries when enabled (ON). Static libraries otherwise (OFF)
   * - CMAKE_CUDA_ARCHITECTURES
     - Semicolon separated list of CUDA architectures to compile for
   * - CMAKE_HIP_ARCHITECTURES
     - Semicolon separated list of HIP architectures to compile for

Manual Build
------------
The build process follows the standard CMake workflow.

To install a minimal build of the library without Python support:

.. code-block:: bash

   mkdir build
   cd build
   cmake .. -DBIPP_PYTHON=OFF -DCMAKE_INSTALL_PREFIX=/usr/local -DBIPP_INSTALL=LIB
   make -j8 install


To build bipp with Python support and install the python module to custom directory:

.. code-block:: bash

   mkdir build
   cd build
   cmake .. -DBIPP_PYTHON=ON -DBIPP_INSTALL=PYTHON -DCMAKE_INSTALL_PREFIX=${path_to_install_to} -DBIPP_PYBIND11_DOWNLOAD=ON
   make -j8 install
   export PYTHONPATH=${path_to_install_to}:$PYTHONPATH

Python - Pip
------------
Bipp uses skbuild to build the Python module with CMake and Pip. The CMake options can be set through environment variables. Example:

.. code-block:: bash

   BIPP_GPU=CUDA CMAKE_PREFIX_PATH="${path_to_neonufft};${path_to_cuneonufft};${CMAKE_PREFIX_PATH}" python3 -m pip install .


.. toctree::
   :maxdepth: 2
   :caption: Python API REFERENCE:
   :hidden:

   python

.. toctree::
   :maxdepth: 2
   :caption: C++ API REFERENCE:
   :hidden:

   context
   exceptions
   nufft_synthesis
   standard_synthesis


.. toctree::
   :maxdepth: 2
   :caption: C API REFERENCE:
   :hidden:

   bipp_c


.. Indices and tables
.. ==================

.. * :ref:`genindex`


