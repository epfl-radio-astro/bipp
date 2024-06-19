#.rst:
# FindFFTWF
# -----------
#
# This module looks for the fftw3f library.
#
# The following variables are set
#
# ::
#
#   FFTWF_FOUND           - True if single precision fftw library is found
#   FFTWF_LIBRARIES       - The required libraries
#   FFTWF_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   FFTWF::FFTWF



# set paths to look for library
set(_FFTWF_PATHS ${FFTW_ROOT} $ENV{FFTW_ROOT} ${FFTWF_ROOT} $ENV{FFTWF_ROOT})
set(_FFTWF_INCLUDE_PATHS)

set(_FFTWF_DEFAULT_PATH_SWITCH)

if(_FFTWF_PATHS)
    # disable default paths if ROOT is set
    set(_FFTWF_DEFAULT_PATH_SWITCH NO_DEFAULT_PATH)
else()
    # try to detect location with pkgconfig
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(PKG_FFTWF QUIET "fftw3")
    endif()
    set(_FFTWF_PATHS ${PKG_FFTWF_LIBRARY_DIRS})
    set(_FFTWF_INCLUDE_PATHS ${PKG_FFTWF_INCLUDE_DIRS})
endif()


find_library(
    FFTWF_LIBRARIES
    NAMES "fftw3f"
    HINTS ${_FFTWF_PATHS}
    PATH_SUFFIXES "lib" "lib64"
    ${_FFTWF_DEFAULT_PATH_SWITCH}
)
find_library(
    FFTWF_OMP_LIBRARIES
    NAMES "fftw3f_omp"
    HINTS ${_FFTWF_PATHS}
    PATH_SUFFIXES "lib" "lib64"
    ${_FFTWF_DEFAULT_PATH_SWITCH}
)
find_path(FFTWF_INCLUDE_DIRS
    NAMES "fftw3.h"
    HINTS ${_FFTWF_PATHS} ${_FFTWF_INCLUDE_PATHS}
    PATH_SUFFIXES "include" "include/fftw"
    ${_FFTWF_DEFAULT_PATH_SWITCH}
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTWF REQUIRED_VARS FFTWF_INCLUDE_DIRS FFTWF_LIBRARIES )


# add target to link against
if(FFTWF_FOUND)
    if(NOT TARGET FFTWF::FFTWF)
        add_library(FFTWF::FFTWF INTERFACE IMPORTED)
    endif()
    set_property(TARGET FFTWF::FFTWF PROPERTY INTERFACE_LINK_LIBRARIES ${FFTWF_LIBRARIES})
    set_property(TARGET FFTWF::FFTWF PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FFTWF_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(FFTWF_FOUND FFTWF_LIBRARIES FFTWF_INCLUDE_DIRS pkgcfg_lib_PKG_FFTWF_fftw3)
