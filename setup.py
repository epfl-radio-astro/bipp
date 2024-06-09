from skbuild import setup
import os
import pathlib
from setuptools import find_packages
import shlex


current_dir = pathlib.Path(__file__).parent.resolve()
with open(str(current_dir) + "/VERSION") as f:
    version = f.readline().strip()

bipp_gpu = str(os.getenv("BIPP_GPU", "OFF"))
bipp_umpire = str(os.getenv("BIPP_UMPIRE", "OFF"))
bipp_omp = str(os.getenv("BIPP_OMP", "ON"))
bipp_vc = str(os.getenv("BIPP_VC", "OFF"))
bipp_mpi = str(os.getenv("BIPP_MPI", "OFF"))
bipp_magma = str(os.getenv("BIPP_MAGMA", "OFF"))
bipp_cmake_args = str(os.getenv("BIPP_CMAKE_ARGS", ""))
bipp_cmake_args_list = shlex.split(bipp_cmake_args) if bipp_cmake_args else []

setup(
    name="bipp",
    version=version,
    description="Bluebild imaging algorithm written in C++",
    long_description="Bluebild imaging algorithm written in C++",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    cmake_install_dir="python",  # must match package dir name. Otherwise, installed libraries are seen as independent data
    package_data={"bipp": ["data/instrument/*.csv"], "bipp.imot_tools": ["data/math/special/*.csv", "data/io/colormap/*.csv", "data/io/colormap/imot_tools.mplstyle"]},
    include_package_data=True,
    python_requires=">=3.6",
    license_files = ('LICENSE',),
    license="GPLv3",
    cmake_args=[
        "-DBIPP_GPU=" + bipp_gpu,
        "-DBIPP_UMPIRE=" + bipp_umpire,
        "-DBIPP_OMP=" + bipp_omp,
        "-DBIPP_MPI=" + bipp_mpi,
        "-DBIPP_VC=" + bipp_vc,
        "-DBIPP_MAGMA=" + bipp_magma,
        "-DBIPP_BUNDLED_LIBS=ON",
        "-DBUILD_SHARED_LIBS=ON",
        "-DBIPP_INSTALL_LIB=OFF",
        "-DBIPP_INSTALL_PYTHON=ON",
        "-DBIPP_INSTALL_PYTHON_SUFFIX=",
        #  "-DBIPP_INSTALL_PYTHON_DEPS=ON",
    ]
    + bipp_cmake_args_list,
    install_requires=[
        "numpy",
        "astropy",
        "matplotlib",
        "tqdm",
        "pyproj",
        "scipy",
        "pandas",
        "healpy",
        "scipy",
        "scikit-learn",
        "python-casacore",
    ]
)
