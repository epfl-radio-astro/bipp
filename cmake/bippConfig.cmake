include(CMakeFindDependencyMacro)
macro(find_dependency_components)
	if(${ARGV0}_FOUND AND ${CMAKE_VERSION} VERSION_LESS "3.15.0")
		# find_dependency does not handle new components correctly before 3.15.0
		set(${ARGV0}_FOUND FALSE)
	endif()
	find_dependency(${ARGV})
endmacro()

# Only look for modules we installed and save value
set(_CMAKE_MODULE_PATH_SAVE ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/modules")

# options used for building library
set(BIPP_GPU @BIPP_GPU@)
set(BIPP_CUDA @BIPP_CUDA@)
set(BIPP_ROCM @BIPP_ROCM@)
set(BIPP_BUNDLED_SPDLOG @BIPP_BUNDLED_SPDLOG@)
set(BIPP_MPI @BIPP_MPI@)

if(BIPP_ROCM)
	find_dependency(hip CONFIG)
	find_dependency(rocblas CONFIG)
endif()


if(BIPP_CUDA)
	if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0") 
		find_dependency(CUDAToolkit)
	else()
		enable_language(CUDA)
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
	endif()
endif()

find_dependency(BLAS)
if(NOT TARGET BLAS::blas)
	# target is only available with CMake 3.18.0 and later
	add_library(BLAS::blas INTERFACE IMPORTED)
	set_property(TARGET BLAS::blas PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif()

if(BIPP_MPI)
	find_dependency(MPI COMPONENTS CXX)
endif()


set(CMAKE_MODULE_PATH ${_CMAKE_MODULE_PATH_SAVE}) # restore module path

# find_dependency may set bipp_FOUND to false, so only add bipp if everything required was found
if(NOT DEFINED bipp_FOUND OR bipp_FOUND)
	# add version of package
	include("${CMAKE_CURRENT_LIST_DIR}/bippConfigVersion.cmake")

	# add library target
	include("${CMAKE_CURRENT_LIST_DIR}/bippStaticTargets.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/bippSharedTargets.cmake")
endif()
