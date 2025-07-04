[![API Documentation](https://readthedocs.org/projects/bipp/badge/?version=latest)](https://bipp.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![CI](https://github.com/epfl-radio-astro/bipp/actions/workflows/ci.yml/badge.svg)](https://github.com/epfl-radio-astro/bipp/actions/workflows/ci.yml)


# BIPP - Bluebild Imaging++
Image synthesis in radio astronomy is done with interferometry, a powerful technique allowing observation of the sky with antenna arrays with otherwise inaccessible angular resolutions and sensitivities. The Bluebild algorithm offers a novel approach to image synthesis, leveraging fPCA to decompose the sky image into distinct energy eigenimages. Bluebild Imaging++ is an HPC implementation of Bluebild.


## Requirements
Bipp requires the following:
- C++17 compliant compiler
- CMake 3.11 and later
- BLAS and LAPACK library like OpenBLAS or Intel MKL
- [Neonufft](https://github.com/epfl-radio-astro/neonufft) for NUFFT computation

Bipp can be configured with additional features (check the CMake options below). The optional requirements are:
- Python header files and [pybind11](https://github.com/pybind/pybind11) for building the Python interface
- CUDA 11.0 and later for Nvidia GPU hardware
- ROCm 6.0 and later for AMD GPU hardware
- [Umpire](https://github.com/LLNL/Umpire) for advanced memory management

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

## Installation

### Pip packages
For x86 systems, binaray pip packages are available for CPU and CUDA configurations.

To install a CPU only version:
```bash
python -m pip install bipp
```

To install the CUDA 12 version:
```bash
python -m pip install bipp-cuda12x
```
The package does not come with bundled CUDA libraries. It therefore requires the cuda libraries to be visible to the runtime linker. 
On some systems, this can be done by setting `LD_LIBRARY_PATH`. For example, if the cuda libraries are located at `/usr/local/cuda/lib64`, then setting `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` might be required.


### Building From Source
Bipp uses CMake to configure the build and has the following options:

| Option                      |  Values                   | Default     | Description                                                                                               |
|-----------------------------|---------------------------|-------------|-----------------------------------------------------------------------------------------------------------|
| `BIPP_PYTHON`               |  `ON`, `OFF`              | `ON`        | Build Python interface                                                                                    |
| `BIPP_GPU`                  |  `OFF`, `CUDA`, `ROCM`    | `OFF`       | Select GPU backend                                                                                        |
| `BIPP_BUILD_TESTS`          |  `ON`, `OFF`              | `OFF`       | Build test executables                                                                                    |
| `BIPP_INSTALL`              |  `LIB`, `PYTHON`, `OFF`   | `LIB`       | Set installation target                                                                                   |
| `BIPP_UMPIRE`               |  `ON`, `OFF`              | `OFF`       | Use the UMPIRE library for memory allocations                                                             |
| `BIPP_BUNDLED_LIBS`         |  `ON`, `OFF`              | `ON`        | Download and build spdlog, pybind11, googletest and json library.                                         |
| `BIPP_INSTALL_LIB`          |  `ON`, `OFF`              | `ON`        | Add library to list of install targets.                                                                   |
| `BIPP_INSTALL_PYTHON`       |  `ON`, `OFF`              | `ON`        | Add python module to list of install targets.                                                             |
| `BIPP_INSTALL_LIB_SUFFIX`   |  string                   | lib or lib64| Installation path suffix appended to `CMAKE_INSTALL_PREFIX` for library target                            |

Some useful general CMake options are:
| Option                     |  Description                                                                    |
|----------------------------|---------------------------------------------------------------------------------|
| `CMAKE_PREFIX_PATH`        |  Semicolon separated list of search paths for external libraries                |
| `CMAKE_INSTALL_PREFIX`     |  Path to installation target directory                                          |
| `BUILD_SHARED_LIBS`        |  Build shared libraries when enabled (`ON`). Static libraries otherwise (`OFF`) |
| `CMAKE_CUDA_ARCHITECTURES` |  Semicolon separated list of CUDA architectures to compile for                  |
| `CMAKE_HIP_ARCHITECTURES`  |  Semicolon separated list of HIP architectures to compile for                   |


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

To build using pip from source, the following can be used:
```console
BIPP_GPU=CUDA CMAKE_PREFIX_PATH="${path_to_neonufft};${CMAKE_PREFIX_PATH}" python3 -m pip install .
```

## Command Line Interface
BIPP can be used through the Python / C++ API, or through a command line interface.
This section describes the five steps required for imaging a measurement set through the Python command line interface.

### Dataset conversion
Compute and store the eigenvectors and eigvenvalues of the visibilities.
```console
python3 -m bipp dataset -t SKAlow -ms EOS_21cm-gf_202MHz_4h1d_200.MS -a 512 -o skalow.h5
```

### Selecting Eigenvalues
Select eigenvalues and filters.
A selection is descripted through a 6-value tuple consisting of filter, number of levels, sigma value [0, 1.0], cluster function, minimum and maximum.
The sigma value is used to exclude large values from the cluster computation.

This example shows creating a selection with 5 levels for eigenvalues in [0,inf) using the 95% smallest eigenvalues with log function for clustering, and one level containing all negative eigenvalues:
```console
python3 -m bipp selection -d skalow.h5 -s lsq,5,0.95,log,0,inf -s lsq,1,1.0,none,-inf,0 -o selection.json
```

### Creating Image Properties
Create image properties with fov and image size values.
```console
python3 -m bipp image_prop -f 10.2 -w 1024 -d skalow.h5 -o image_prop.h5
```

### Image Synthesis
Compute the image synthesis.
```console
python3 -m bipp synthesis -d skalow.h5 -s selection.json -i image_prop.h5  -o images.h5
```

### Plotting
Export the images as PNG files.
```console
python3 -m bipp plot -i images.h5
```
