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

# Options
set(BUILD_SHARED_LIBS "ON" CACHE STRING "Build as shared library") # default to shared
option(BIPP_BUILD_TESTS "Build tests" OFF)
option(BIPP_BUILD_APPS "Build apps" ON)
option(BIPP_PYTHON "Build python module" ON)
option(BIPP_UMPIRE "Use Umpire memory pool library" OFF)
option(BIPP_MPI "Use MPI for multi-process parallization" OFF)

option(BIPP_INSTALL_LIB "Install library" ON)
cmake_dependent_option(BIPP_INSTALL_APPS "BIPP_INSTALL_APPS" ON "BIPP_INSTALL_LIB" OFF)

cmake_dependent_option(BIPP_INSTALL_PYTHON "Install Python module" OFF "BIPP_PYTHON" OFF)
cmake_dependent_option(BIPP_INSTALL_PYTHON_DEPS "Install dependencies bundled with Python module" OFF "BIPP_PYTHON" OFF)
set(BIPP_INSTALL_PYTHON_MODE "platlib" CACHE STRING "Python installation mode. Can be \"platlib\" or \"skbuild\".")

# Library install location
set(BIPP_INSTALL_LIB_SUFFIX "${CMAKE_INSTALL_LIBDIR}" CACHE STRING "Lib install suffix")

option(BIPP_BUNDLED_LIBS "Use bundled libraries for spdlog, googletest and json" ON)
cmake_dependent_option(BIPP_BUNDLED_SPDLOG "Use bundled spdlog lib" ON "BIPP_BUNDLED_LIBS" OFF)
cmake_dependent_option(BIPP_BUNDLED_PYBIND11 "Use bundled pybind11 lib" ON "BIPP_BUNDLED_LIBS" OFF)
cmake_dependent_option(BIPP_BUNDLED_GOOGLETEST "Use bundled googletest lib" ON "BIPP_BUNDLED_LIBS" OFF)
cmake_dependent_option(BIPP_BUNDLED_JSON "Use bundled json lib" ON "BIPP_BUNDLED_LIBS" OFF)
cmake_dependent_option(BIPP_BUNDLED_CLI11 "Use bundled CLI11 lib" ON "BIPP_BUNDLED_LIBS" OFF)

set(BIPP_GPU "OFF" CACHE STRING "GPU backend")
set_property(CACHE BIPP_GPU PROPERTY STRINGS
	"OFF" "CUDA" "ROCM"
	)

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

	if(BIPP_INSTALL_PYTHON)
	  if(BIPP_INSTALL_PYTHON_MODE STREQUAL "skbuild")
		if(NOT SKBUILD_PROJECT_NAME)
		  message(FATAL_ERROR "Expected SKBUILD_PROJECT_NAME to be defined")
		endif()
		set(BIPP_INSTALL_PYTHON_PATH ${SKBUILD_PROJECT_NAME})
	  else()
		set(BIPP_INSTALL_PYTHON_PREFIX ${CMAKE_INSTALL_PREFIX})
		execute_process(
		  COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/python_install_path.py ${BIPP_INSTALL_PYTHON_PREFIX}
			OUTPUT_VARIABLE BIPP_INSTALL_PYTHON_PATH 
			OUTPUT_STRIP_TRAILING_WHITESPACE
			COMMAND_ERROR_IS_FATAL ANY)
		set(BIPP_INSTALL_PYTHON_PATH ${BIPP_INSTALL_PYTHON_PATH}/bipp)
	  endif()
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

set(BIPP_TEST_LIBRARIES)
set(BIPP_APP_LIBRARIES)

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

		# find_library(CUDA_CUSOLVER_LIBRARY cusolver PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		# if(NOT TARGET CUDA::cusolver)
		#     add_library(CUDA::cusolver INTERFACE IMPORTED)
		# endif()
		# set_property(TARGET CUDA::cusolver PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUSOLVER_LIBRARY})
		# set_property(TARGET CUDA::cusolver PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	endif()

	list(APPEND BIPP_EXTERNAL_LIBS CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()

# ROCm
if(BIPP_ROCM)
	enable_language(HIP)

	find_package(hip CONFIG REQUIRED)
	find_package(rocblas CONFIG REQUIRED)
	find_package(hipcub CONFIG REQUIRED)
	list(APPEND BIPP_EXTERNAL_LIBS hip::host roc::rocblas hip::hipcub)
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

if(BIPP_CUDA OR BIPP_ROCM)
	if(NOT TARGET neonufft::neonufft_gpu)
	  message(FATAL_ERROR "Neonufft not compiled with GPU support.")
	endif()
	list(APPEND BIPP_EXTERNAL_LIBS neonufft::neonufft_gpu)
endif()

if(BIPP_BUNDLED_SPDLOG)
  set(SPDLOG_INSTALL OFF CACHE BOOL "")
  FetchContent_Declare(
	spdlog
	URL https://github.com/gabime/spdlog/archive/refs/tags/v1.14.1.tar.gz
	URL_MD5 f2c3f15c20e67b261836ff7bfda302cf
  )
  FetchContent_MakeAvailable(spdlog)
  list(APPEND BIPP_EXTERNAL_LIBS spdlog::spdlog_header_only)
else()
  find_package(spdlog CONFIG REQUIRED)
  list(APPEND BIPP_EXTERNAL_LIBS spdlog::spdlog)
endif()

# header-only command line parser
if(BIPP_BUILD_APPS)
  if(BIPP_BUNDLED_CLI11)
	set(CLI11_INSTALL OFF CACHE BOOL "")
	set(CLI11_BUILD_TESTS OFF CACHE BOOL "")
	set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "")
	FetchContent_Declare(
	  cli11
	  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz
	  URL_MD5 f9c0acb8c483a524e9dd275955227254
	)
	FetchContent_MakeAvailable(cli11)
  else()
	find_package(CLI11 CONFIG REQUIRED)
  endif()
  list(APPEND BIPP_APP_LIBRARIES CLI11::CLI11)
endif()

# json parser

if(BIPP_BUILD_TESTS OR BIPP_BUILD_APPS)
  if(BIPP_BUNDLED_JSON)
	set(JSON_Install OFF CACHE BOOL "")
	set(JSON_BuildTests OFF CACHE INTERNAL "")
	  FetchContent_Declare(
	  json
	  URL https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
	  URL_MD5 c2528c3e04faccaaee44f1f8f3d30d99
	)
	FetchContent_MakeAvailable(json)
  else()
	find_package(nlohmann_json CONFIG REQUIRED)
  endif()
  list(APPEND BIPP_APP_LIBRARIES nlohmann_json::nlohmann_json)
  list(APPEND BIPP_TEST_LIBRARIES nlohmann_json::nlohmann_json)
endif()


if(BIPP_BUILD_TESTS)
  if(BIPP_BUNDLED_GOOGLETEST)
	set(BUILD_GMOCK OFF CACHE BOOL "")
	set(INSTALL_GTEST OFF CACHE BOOL "")
	# add googletest
	FetchContent_Declare(
	  googletest
	  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
	  URL_MD5 95b29f0038ec84a611df951d74d99897
	)
	FetchContent_MakeAvailable(googletest)
  else()
	find_package(googletest CONFIG REQUIRED)
  endif()
  list(APPEND BIPP_TEST_LIBRARIES gtest)
endif()


# check if C api is available for blas and lapack
# include(CheckCXXSymbolExists)
set(CMAKE_REQUIRED_LIBRARIES BLAS::BLAS)
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
