cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
file(READ "VERSION" BIPP_VERSION)
string(STRIP ${BIPP_VERSION} BIPP_VERSION)
project(bipp LANGUAGES CXX VERSION "${BIPP_VERSION}")
set(BIPP_SO_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})

# allow {module}_ROOT variables to be set
if(POLICY CMP0074)
	cmake_policy(SET CMP0074 NEW)
endif()

# use INTERFACE_LINK_LIBRARIES property if available
if(POLICY CMP0022)
	cmake_policy(SET CMP0022 NEW)
endif()

# update time stamps when using FetchContent
if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()

# set default build type to RELEASE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo"
	)
endif()

set(BIPP_BUILD_TYPE "OFF" CACHE STRING "If set, overrides the CMAKE_BUILD_TYPE variable.")
set_property(CACHE BIPP_BUILD_TYPE PROPERTY STRINGS
	"Debug" "Release" "MinSizeRel" "RelWithDebInfo")
if(BIPP_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE ${BIPP_BUILD_TYPE} CACHE STRING "Build type" FORCE)
endif()


# set language and standard
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)

# Get GNU standard install prefixes
include(GNUInstallDirs)

include(FetchContent)
include(CMakeDependentOption)

#add local module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)

# Options
set(BUILD_SHARED_LIBS "ON" CACHE STRING "Build as shared library") # default to shared
option(BIPP_BUILD_TESTS "Build tests" OFF)
option(BIPP_PYTHON "Build python module" ON)
option(BIPP_MAGMA "Use MAGMA library as eigensolver" OFF)
option(BIPP_UMPIRE "Use Umpire memory pool library" OFF)
option(BIPP_VC "Use VC vectorization library" OFF)
option(BIPP_OMP "Use OpenMP for multi-threading" ON)
option(BIPP_MPI "Use MPI for multi-process parallization" OFF)

option(BIPP_INSTALL_LIB "Install library" ON)
cmake_dependent_option(BIPP_INSTALL_PYTHON "Install Python module" ON "BIPP_PYTHON" OFF)
cmake_dependent_option(BIPP_INSTALL_PYTHON_DEPS "Install dependencies bundled with Python module" OFF "BIPP_PYTHON" OFF)
set(BIPP_INSTALL_PYTHON_PREFIX "" CACHE STRING "Python install prefix. If not set, CMAKE_INSTALL_PREFIX will be used.")
set(BIPP_INSTALL_PYTHON_SUFFIX "platlib" CACHE STRING "Python installation suffix. If platlib, python platlib path will be used.")

# Library install location
set(BIPP_INSTALL_LIB_SUFFIX "${CMAKE_INSTALL_LIBDIR}" CACHE STRING "Lib install suffix")

option(BIPP_BUNDLED_LIBS "Use bundled libraries for spdlog, googletest and json" ON)
cmake_dependent_option(BIPP_BUNDLED_SPDLOG "Use bundled spdlog lib" ON "BIPP_BUNDLED_LIBS" OFF)
cmake_dependent_option(BIPP_BUNDLED_PYBIND11 "Use bundled pybind11 lib" ON "BIPP_BUNDLED_LIBS" OFF)
cmake_dependent_option(BIPP_BUNDLED_GOOGLETEST "Use bundled googletest lib" ON "BIPP_BUNDLED_LIBS" OFF)
cmake_dependent_option(BIPP_BUNDLED_JSON "Use bundled json lib" ON "BIPP_BUNDLED_LIBS" OFF)

set(BIPP_GPU "OFF" CACHE STRING "GPU backend")
set_property(CACHE BIPP_GPU PROPERTY STRINGS
	"OFF" "CUDA" "ROCM"
	)
set(CMAKE_HIP_ARCHITECTURES "gfx803;gfx900;gfx906" CACHE STRING "HIP GPU architectures")

set(BIPP_BUILD_TYPE "OFF" CACHE STRING "If set, overrides the CMAKE_BUILD_TYPE variable.")
set_property(CACHE BIPP_BUILD_TYPE PROPERTY STRINGS
	"Debug" "Release" "MinSizeRel" "RelWithDebInfo")
if(BIPP_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE ${BIPP_BUILD_TYPE})
endif()


# Python install location
if(BIPP_PYTHON)
	set(Python_FIND_FRAMEWORK LAST) # Prefer Brew/Conda to Apple framework python

	find_package(Python 3.6 REQUIRED COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.Embed)

	if(NOT BIPP_INSTALL_PYTHON_PREFIX)
	  set(BIPP_INSTALL_PYTHON_PREFIX ${CMAKE_INSTALL_PREFIX})
	endif()

	if(BIPP_INSTALL_PYTHON_SUFFIX STREQUAL "platlib")
		execute_process(
		  COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/python_install_path.py ${BIPP_INSTALL_PYTHON_PREFIX}
			OUTPUT_VARIABLE BIPP_INSTALL_PYTHON_PATH 
			OUTPUT_STRIP_TRAILING_WHITESPACE
			COMMAND_ERROR_IS_FATAL ANY)
	else()
	  set(BIPP_INSTALL_PYTHON_PATH "${BIPP_INSTALL_PYTHON_PREFIX}/${BIPP_INSTALL_PYTHON_SUFFIX}")
	endif()
endif()

# Options combination check
set(BIPP_CUDA OFF)
set(BIPP_ROCM OFF)
if(BIPP_GPU)
	if(BIPP_GPU STREQUAL "CUDA")
		set(BIPP_CUDA ON)
	elseif(BIPP_GPU STREQUAL "ROCM")
		set(BIPP_ROCM ON)
		set(BIPP_MAGMA ON)
	else()
		message(FATAL_ERROR "Invalid GPU backend")
	endif()
endif()

if(BIPP_INSTALL STREQUAL "PYTHON" OR BIPP_INSTALL STREQUAL "PIP")
	set(BIPP_PYTHON ON)
endif()


set(BIPP_FLAGS "")
set(BIPP_EXTERNAL_LIBS "")
set(BIPP_EXTERNAL_LIBS_PUBLIC "")
set(BIPP_INCLUDE_DIRS "")

# CUDA
if(BIPP_CUDA)
	enable_language(CUDA)
	if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0") 
		find_package(CUDAToolkit REQUIRED)
	else()
		find_library(CUDA_CUDART_LIBRARY cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		if(NOT TARGET CUDA::cudart)
			add_library(CUDA::cudart INTERFACE IMPORTED)
		endif()
		set_property(TARGET CUDA::cudart PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUDART_LIBRARY})
		set_property(TARGET CUDA::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

		find_library(CUDA_CUBLAS_LIBRARY cublas PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		if(NOT TARGET CUDA::cublas)
			add_library(CUDA::cublas INTERFACE IMPORTED)
		endif()
		set_property(TARGET CUDA::cublas PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUBLAS_LIBRARY})
		set_property(TARGET CUDA::cublas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

		find_library(CUDA_CUSOLVER_LIBRARY cusolver PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		if(NOT TARGET CUDA::cusolver)
			add_library(CUDA::cusolver INTERFACE IMPORTED)
		endif()
		set_property(TARGET CUDA::cusolver PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUSOLVER_LIBRARY})
		set_property(TARGET CUDA::cusolver PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	endif()

	list(APPEND BIPP_EXTERNAL_LIBS CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()

# ROCm
if(BIPP_ROCM)
	list(APPEND BIPP_FLAGS -fno-gpu-rdc)

	include(CheckCXXCompilerFlag)

	check_cxx_compiler_flag("--offload-arch=gfx906" _IS_OFFLOAD_HIPCC)
	if(_IS_OFFLOAD_HIPCC)
		foreach(arch ${CMAKE_HIP_ARCHITECTURES})
			list(APPEND BIPP_FLAGS --offload-arch=${arch})
		endforeach()
	else()
		check_cxx_compiler_flag("--amdgpu-target=gfx906" _IS_AMDGPU_HIPCC)
		if(NOT _IS_AMDGPU_HIPCC)
			message(FATAL_ERROR "ROCm build requires hipcc to be set as CXX compiler.")
		endif()
		foreach(arch ${CMAKE_HIP_ARCHITECTURES})
			list(APPEND BIPP_FLAGS --amdgpu-target=${arch})
		endforeach()
	endif()

	find_package(hip CONFIG REQUIRED)
	find_package(rocblas CONFIG REQUIRED)
	find_package(hipcub CONFIG REQUIRED)
	list(APPEND BIPP_EXTERNAL_LIBS hip::host roc::rocblas hip::hipcub)
endif()

# MAGMA
if(BIPP_MAGMA)
	find_package(MAGMA MODULE REQUIRED)
	list(APPEND BIPP_EXTERNAL_LIBS MAGMA::MAGMA)
	list(APPEND BIPP_FLAGS -DUSE_HIP) # required to compile magma header
endif()

# Umpire
if(BIPP_UMPIRE)
	find_package(umpire CONFIG REQUIRED)
	list(APPEND BIPP_EXTERNAL_LIBS umpire)
endif()

set(BLA_SIZEOF_INTEGER 4) # 32 bit interface to blas / lapack

# MPI
if(BIPP_MPI)
  find_package(MPI COMPONENTS CXX REQUIRED)
  list(APPEND BIPP_EXTERNAL_LIBS_PUBLIC MPI::MPI_CXX)
endif()

# BLAS
find_package(BLAS REQUIRED)
if(NOT TARGET BLAS::BLAS)
	# target is only available with CMake 3.18.0 and later
	add_library(BLAS::BLAS INTERFACE IMPORTED)
	set_property(TARGET BLAS::BLAS PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif()
list(APPEND BIPP_EXTERNAL_LIBS BLAS::BLAS)

# LAPACK
find_package(LAPACK REQUIRED)
if(NOT TARGET LAPACK::LAPACK)
	# target is only available with CMake 3.18.0 and later
	add_library(LAPACK::LAPACK INTERFACE IMPORTED)
	set_property(TARGET LAPACK::LAPACK PROPERTY INTERFACE_LINK_LIBRARIES ${LAPACK_LINKER_FLAGS} ${LAPACK_LIBRARIES})
endif()
list(APPEND BIPP_EXTERNAL_LIBS LAPACK::LAPACK)

# HDF5
find_package(HDF5 MODULE REQUIRED COMPONENTS C)
list(APPEND BIPP_EXTERNAL_LIBS hdf5::hdf5)

find_package(neonufft CONFIG REQUIRED)
list(APPEND BIPP_EXTERNAL_LIBS neonufft::neonufft)
# finufft has fftw in interface headers
find_package(FFTW REQUIRED)
find_package(FFTWF REQUIRED)
list(APPEND BIPP_EXTERNAL_LIBS FFTW::FFTW FFTWF::FFTWF)

if(BIPP_CUDA OR BIPP_ROCM)
	find_package(CUFINUFFT REQUIRED)
	list(APPEND BIPP_EXTERNAL_LIBS CUFINUFFT::cufinufft)
endif()

# OpenMP
if(BIPP_OMP)
  find_package(OpenMP REQUIRED)
  list(APPEND BIPP_EXTERNAL_LIBS OpenMP::OpenMP_CXX)
endif()

if(BIPP_VC)
  find_package(Vc CONFIG REQUIRED)
  list(APPEND BIPP_EXTERNAL_LIBS Vc::Vc)
endif()


if(BIPP_BUNDLED_SPDLOG)
  FetchContent_Declare(
	spdlog
	URL https://github.com/gabime/spdlog/archive/refs/tags/v1.14.1.tar.gz
	URL_MD5 f2c3f15c20e67b261836ff7bfda302cf
  )
  FetchContent_GetProperties(spdlog)
  if(NOT spdlog_POPULATED)
	FetchContent_Populate(spdlog)
  endif()
  list(APPEND BIPP_INCLUDE_DIRS ${spdlog_SOURCE_DIR}/include)
else()
  find_package(spdlog CONFIG REQUIRED)
  list(APPEND BIPP_EXTERNAL_LIBS spdlog::spdlog)
endif()

# check if C api is available for blas and lapack
# include(CheckCXXSymbolExists)
set(CMAKE_REQUIRED_LIBRARIES ${BIPP_EXTERNAL_LIBS})
include(CheckFunctionExists)

unset(BIPP_BLAS_C CACHE) # Result is cached, so change of library will not lead to a new check automatically
CHECK_FUNCTION_EXISTS(cblas_zgemm BIPP_BLAS_C)

unset(BIPP_LAPACK_C CACHE) # Result is cached, so change of library will not lead to a new check automatically
CHECK_FUNCTION_EXISTS(LAPACKE_chegv BIPP_LAPACK_C)

# generate config.h
configure_file(include/bipp/config.h.in ${PROJECT_BINARY_DIR}/bipp/config.h)

list(APPEND BIPP_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)
list(APPEND BIPP_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
list(APPEND BIPP_INCLUDE_DIRS ${PROJECT_BINARY_DIR})

#############################################################################
# All include dirs and definitions must be set before sub-directory is added!
#############################################################################
add_subdirectory(src)

if(BIPP_PYTHON)
	add_subdirectory(python)
endif()

# add tests for developement
if(BIPP_BUILD_TESTS)
	add_subdirectory(tests)
endif()
