[![Documentation](https://readthedocs.org/projects/bipp/badge/?version=latest)](https://bipp.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![CI](https://github.com/epfl-radio-astro/bipp/actions/workflows/ci.yml/badge.svg)](https://github.com/epfl-radio-astro/bipp/actions/workflows/ci.yml)


# BIPP - Bluebild Imaging++
Image synthesis in radio astronomy is done with interferometry, a powerful technique allowing observation of the sky with antenna arrays with otherwise inaccessible angular resolutions and sensitivities. The Bluebild algorithm offers a novel approach to image synthesis, leveraging fPCA to decompose the sky image into distinct energy eigenimages. Bluebild Imaging++ is an HPC implementation of Bluebild.


## Requirements
Bipp requires the following:
- C++17 compliant compiler
- CMake 3.11 and later
- BLAS and LAPACK library like OpenBLAS or Intel MKL
- [fiNUFFT](https://github.com/flatironinstitute/finufft)

Bipp can be configured with additional features (check the CMake options below). The optional requirements are:
- Python header files and [pybind11](https://github.com/pybind/pybind11) for building the Python interface
- CUDA 9.0 and later for Nvidia GPU hardware
- ROCm 5.0 and later for AMD GPU hardware
- cuFINUUFT with type 3 transform support with CUDA or ROCm enabled. Currently, only available in a fork (t3_d3 branch): https://github.com/AdhocMan/cufinufft/tree/t3_d3
- [Umpire](https://github.com/LLNL/Umpire) for advanced memory management
- [MAGMA](https://icl.utk.edu/magma/) for an alternative eigensolver implementation on GPU. Currently mandatory with ROCm
- [VC](https://github.com/VcDevel/Vc) for improved CPU performance through customized vectorization

The Python module has the following dependencies:
- numpy
- astropy
- matplotlib
- tqdm
- pyproj
- scipy
- pandas
- healpy


## Installation
Bipp uses CMake to configure the build.

### CMake options
Bipp can be configured with the following options:
| Option                   |  Values                   | Default | Description                                                             |
|--------------------------|---------------------------|---------|-------------------------------------------------------------------------|
| `BIPP_PYTHON`            |  `ON`, `OFF`              | `ON`    | Build Python interface                                                  |
| `BIPP_OMP`               |  `ON`, `OFF`              | `ON`    | Enable multi-threading with OpenMP                                      |
| `BIPP_VC`                |  `ON`, `OFF`              | `OFF`   | Use the VC library for vectorization                                    |
| `BIPP_GPU`               |  `OFF`, `CUDA`, `ROCM`    | `OFF`   | Select GPU backend                                                      |
| `BIPP_MAGMA`             |  `ON`, `OFF`              | `OFF`   | Use MAGMA as eigensolver on GPU                                         |
| `BIPP_BUILD_TESTS`       |  `ON`, `OFF`              | `OFF`   | Build test executables                                                  |
| `BIPP_INSTALL`           |  `LIB`, `PYTHON`, `OFF`   | `LIB`   | Set installation target                                                 |
| `BIPP_UMPIRE`            |  `ON`, `OFF`              | `OFF`   | Use the UMPIRE library for memory allocations                           |
| `BIPP_BUNDLED_LIBS`      |  `ON`, `OFF`              | `ON`    | Download and build spdlog, pybind11, googletest and json library.       |


Some useful general CMake options are:
| Option                     |  Description                                                                    |
|----------------------------|---------------------------------------------------------------------------------|
| `CMAKE_PREFIX_PATH`        |  Semicolon separated list of search paths for external libraries                |
| `CMAKE_INSTALL_PREFIX`     |  Path to installation target directory                                          |
| `BUILD_SHARED_LIBS`        |  Build shared libraries when enabled (`ON`). Static libraries otherwise (`OFF`) |
| `CMAKE_CUDA_ARCHITECTURES` |  Semicolon separated list of CUDA architectures to compile for                  |
| `CMAKE_HIP_ARCHITECTURES`  |  Semicolon separated list of HIP architectures to compile for                   |

### Manual Build
The build process follows the standard CMake workflow.

To install a minimal build of the library without Python support:
```console
mkdir build
cd build
cmake .. -DBIPP_PYTHON=OFF -DCMAKE_INSTALL_PREFIX=/usr/local -DBIPP_INSTALL=LIB
make -j8 install
```


To build bipp with Python support and install the python module to custom directory:
```console
mkdir build
cd build
cmake .. -DBIPP_PYTHON=ON -DBIPP_INSTALL=PYTHON -DCMAKE_INSTALL_PREFIX=${path_to_install_to} -DBIPP_PYBIND11_DOWNLOAD=ON
make -j8 install
export PYTHONPATH=${path_to_install_to}:$PYTHONPATH
```

### Python - Pip
Bipp uses skbuild to build the Python module with CMake and Pip. The CMake options can be set through environment variables. Example:

```console
BIPP_GPU=CUDA CMAKE_PREFIX_PATH="${path_to_finufft};${path_to_cufinufft};${CMAKE_PREFIX_PATH}" python3 -m pip install .
```

