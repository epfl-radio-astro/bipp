#.rst:
# FindFINUFFT
# -----------
#
# This module tries to find the FINUFFT library.
#
# The following variables are set
#
# ::
#
#   FINUFFT_FOUND           - True if finufft is found
#   FINUFFT_LIBRARIES       - The required libraries
#   FINUFFT_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   FINUFFT::finufft

#set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
if(NOT POLICY CMP0074)
    set(_FINUFFT_PATHS ${FINUFFT_ROOT} $ENV{FINUFFT_ROOT})
endif()

find_library(
    FINUFFT_LIBRARIES
    NAMES "finufft"
    HINTS ${_FINUFFT_PATHS}
    PATH_SUFFIXES "finufft/lib" "finufft/lib64" "finufft"
)
find_path(
    FINUFFT_INCLUDE_DIRS
    NAMES "finufft.h"
    HINTS ${_FINUFFT_PATHS}
    PATH_SUFFIXES "finufft" "finufft/include" "include/finufft"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FINUFFT REQUIRED_VARS FINUFFT_INCLUDE_DIRS FINUFFT_LIBRARIES)

# add target to link against
if(FINUFFT_FOUND)
    if(NOT TARGET FINUFFT::finufft)
        add_library(FINUFFT::finufft INTERFACE IMPORTED)
    endif()
    set_property(TARGET FINUFFT::finufft PROPERTY INTERFACE_LINK_LIBRARIES ${FINUFFT_LIBRARIES})
    set_property(TARGET FINUFFT::finufft PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FINUFFT_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(FINUFFT_FOUND FINUFFT_LIBRARIES FINUFFT_INCLUDE_DIRS)
