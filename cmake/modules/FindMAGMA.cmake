#.rst:
# FindMAGMA
# -----------
#
# This module looks for the MAGMA3 library.
#
# The following variables are set
#
# ::
#
#   MAGMA_FOUND           - True if double precision MAGMA library is found
#   MAGMA_LIBRARIES       - The required libraries
#   MAGMA_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   MAGMA::MAGMA



# set paths to look for library
set(_MAGMA_PATHS ${MAGMA_ROOT} $ENV{MAGMA_ROOT})
set(_MAGMA_INCLUDE_PATHS)

set(_MAGMA_DEFAULT_PATH_SWITCH)

if(_MAGMA_PATHS)
    # disable default paths if ROOT is set
    set(_MAGMA_DEFAULT_PATH_SWITCH NO_DEFAULT_PATH)
endif()


find_library(
    MAGMA_LIBRARIES
    NAMES "magma"
    HINTS ${_MAGMA_PATHS}
    PATH_SUFFIXES "lib" "lib64"
    ${_MAGMA_DEFAULT_PATH_SWITCH}
)
find_path(MAGMA_INCLUDE_DIRS
    NAMES "magma.h"
    HINTS ${_MAGMA_PATHS} ${_MAGMA_INCLUDE_PATHS}
    PATH_SUFFIXES "include" "include/MAGMA"
    ${_MAGMA_DEFAULT_PATH_SWITCH}
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MAGMA REQUIRED_VARS MAGMA_INCLUDE_DIRS MAGMA_LIBRARIES )

# add target to link against
if(MAGMA_FOUND)
    if(NOT TARGET MAGMA::MAGMA)
        add_library(MAGMA::MAGMA INTERFACE IMPORTED)
    endif()
    set_property(TARGET MAGMA::MAGMA PROPERTY INTERFACE_LINK_LIBRARIES ${MAGMA_LIBRARIES})
    set_property(TARGET MAGMA::MAGMA PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MAGMA_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(MAGMA_FOUND MAGMA_LIBRARIES MAGMA_INCLUDE_DIRS)
