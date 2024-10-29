#.rst:
# FindFFTW
# -----------
#
# This module looks for the fftw3 library.
#
# The following variables are set
#
# ::
#
#   FFTW_FOUND           - True if double precision fftw library is found
#   FFTW_LIBRARIES       - The required libraries
#   FFTW_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   FFTW::FFTW



# set paths to look for library
set(_FFTW_PATHS ${FFTW_ROOT} $ENV{FFTW_ROOT})
set(_FFTW_INCLUDE_PATHS)

set(_FFTW_DEFAULT_PATH_SWITCH)

if(_FFTW_PATHS)
    # disable default paths if ROOT is set
    set(_FFTW_DEFAULT_PATH_SWITCH NO_DEFAULT_PATH)
else()
    # try to detect location with pkgconfig
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(PKG_FFTW QUIET "fftw3")
    endif()
    set(_FFTW_PATHS ${PKG_FFTW_LIBRARY_DIRS})
    set(_FFTW_INCLUDE_PATHS ${PKG_FFTW_INCLUDE_DIRS})
endif()


find_library(
    FFTW_LIBRARIES
    NAMES "fftw3"
    HINTS ${_FFTW_PATHS}
    PATH_SUFFIXES "lib" "lib64"
    ${_FFTW_DEFAULT_PATH_SWITCH}
)
find_library(
    FFTW_OMP_LIBRARIES
    NAMES "fftw3_omp"
    HINTS ${_FFTW_PATHS}
    PATH_SUFFIXES "lib" "lib64"
    ${_FFTW_DEFAULT_PATH_SWITCH}
)
find_path(FFTW_INCLUDE_DIRS
    NAMES "fftw3.h"
    HINTS ${_FFTW_PATHS} ${_FFTW_INCLUDE_PATHS}
    PATH_SUFFIXES "include" "include/fftw"
    ${_FFTW_DEFAULT_PATH_SWITCH}
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW REQUIRED_VARS FFTW_INCLUDE_DIRS FFTW_LIBRARIES )

# add target to link against
if(FFTW_FOUND)
    if(NOT TARGET FFTW::FFTW)
        add_library(FFTW::FFTW INTERFACE IMPORTED)
    endif()
    set_property(TARGET FFTW::FFTW PROPERTY INTERFACE_LINK_LIBRARIES ${FFTW_LIBRARIES})
    set_property(TARGET FFTW::FFTW PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FFTW_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(FFTW_FOUND FFTW_LIBRARIES FFTW_INCLUDE_DIRS pkgcfg_lib_PKG_FFTW_fftw3)
