#.rst:
# FindCUFINUFFT
# -----------
#
# This module tries to find the CUFINUFFT library.
#
# The following variables are set
#
# ::
#
#   CUFINUFFT_FOUND           - True if cufinufft is found
#   CUFINUFFT_LIBRARIES       - The required libraries
#   CUFINUFFT_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   CUFINUFFT::cufinufft

#set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
if(NOT POLICY CMP0074)
    set(_CUFINUFFT_PATHS ${CUFINUFFT_ROOT} $ENV{CUFINUFFT_ROOT})
endif()

find_library(
    CUFINUFFT_LIBRARIES
    NAMES "cufinufft"
    HINTS ${_CUFINUFFT_PATHS}
    PATH_SUFFIXES "cufinufft/lib" "cufinufft/lib64" "cufinufft"
)
find_path(
    CUFINUFFT_INCLUDE_DIRS
    NAMES "cufinufft.h"
    HINTS ${_CUFINUFFT_PATHS}
    PATH_SUFFIXES "cufinufft" "cufinufft/include" "include/cufinufft"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUFINUFFT REQUIRED_VARS CUFINUFFT_INCLUDE_DIRS CUFINUFFT_LIBRARIES)

# add target to link against
if(CUFINUFFT_FOUND)
    if(NOT TARGET CUFINUFFT::cufinufft)
        add_library(CUFINUFFT::cufinufft INTERFACE IMPORTED)
    endif()
    set_property(TARGET CUFINUFFT::cufinufft PROPERTY INTERFACE_LINK_LIBRARIES ${CUFINUFFT_LIBRARIES})
    set_property(TARGET CUFINUFFT::cufinufft PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUFINUFFT_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(CUFINUFFT_FOUND CUFINUFFT_LIBRARIES CUFINUFFT_INCLUDE_DIRS)
