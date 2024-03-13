include(CMakeFindDependencyMacro)
macro(find_dependency_components)
	if(${ARGV0}_FOUND AND ${CMAKE_VERSION} VERSION_LESS "3.15.0")
		# find_dependency does not handle new components correctly before 3.15.0
		set(${ARGV0}_FOUND FALSE)
	endif()
	find_dependency(${ARGV})
endmacro()

# options used for building library
set(BIPP_GPU @BIPP_GPU@)
set(BIPP_CUDA @BIPP_CUDA@)
set(BIPP_ROCM @BIPP_ROCM@)
set(BIPP_MPI @BIPP_MPI@)

if(BIPP_MPI)
	find_dependency(MPI COMPONENTS CXX)
endif()

# find_dependency may set bipp_FOUND to false, so only add bipp if everything required was found
if(NOT DEFINED bipp_FOUND OR bipp_FOUND)
	# add version of package
	include("${CMAKE_CURRENT_LIST_DIR}/bippSharedConfigVersion.cmake")

	# add library target
	include("${CMAKE_CURRENT_LIST_DIR}/bippSharedTargets.cmake")
endif()
