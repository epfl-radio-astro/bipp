set(BIPP_SOURCE_FILES
	context_internal.cpp
	communicator.cpp
	context.cpp
	logger.cpp
	rt_graph.cpp
	dataset_file.cpp
	eigensolver.cpp
	context.cpp
	memory/allocator_factory.cpp
	host/eigensolver.cpp
	host/gram_matrix.cpp
	host/nufft_synthesis.cpp
	host/virtual_vis.cpp
	host/kernels/gemmexp.cpp
	host/kernels/nuft_sum.cpp
	host/standard_synthesis.cpp
	image_synthesis.cpp
	image_file.cpp
	)

if(BIPP_CUDA OR BIPP_ROCM)
	list(APPEND BIPP_SOURCE_FILES
		gpu/kernels/add_vector.cu
		)
endif()

# if(BIPP_CUDA OR BIPP_ROCM)
#         list(APPEND BIPP_SOURCE_FILES
#                 gpu/eigensolver.cpp
#                 gpu/gram_matrix.cpp
#                 gpu/nufft_3d3.cpp
#                 gpu/nufft_synthesis.cpp
#                 gpu/virtual_vis.cpp
#                 gpu/standard_synthesis.cpp
#                 gpu/domain_partition.cu
#                 gpu/collector.cpp
#                 gpu/util/solver_api.cpp
#                 gpu/kernels/gram.cu
#                 gpu/kernels/scale_matrix.cu
#                 gpu/kernels/add_vector.cu
#                 gpu/kernels/gemmexp.cu
#                 gpu/kernels/center_vector.cu
#                 gpu/kernels/min_max_element.cu
#                 gpu/kernels/nuft_sum.cu
#                 gpu/kernels/copy_at_indices.cu
#                 gpu/kernels/scale_vector.cu
#                 )
# endif()

if(BIPP_MPI)
	list(APPEND BIPP_SOURCE_FILES
		mpi_util/mpi_init_guard.cpp
		)
endif()

# set all files to CXX, such that .cu files are compiled by the CXX compiler with ROCM enabled
if(BIPP_ROCM)
    set_source_files_properties(${BIPP_SOURCE_FILES} PROPERTIES LANGUAGE CXX)
endif()



add_library(bipp_objects OBJECT ${BIPP_SOURCE_FILES})

# All .cu files are self-contained. Device linking can have issues with propageted linker flags of other targets like MPI.
if(BIPP_CUDA)
	set_property(TARGET bipp_objects PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
	set_property(TARGET bipp_objects PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
endif()
# make static libraries linkable into shared ones
set_property(TARGET bipp_objects PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_options(bipp_objects PRIVATE ${BIPP_FLAGS})
target_include_directories(bipp_objects PRIVATE ${BIPP_INCLUDE_DIRS})
target_link_libraries(bipp_objects PRIVATE ${BIPP_EXTERNAL_LIBS})
target_link_libraries(bipp_objects PRIVATE ${BIPP_EXTERNAL_LIBS_PUBLIC})
target_include_directories(bipp_objects INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)


#shared library
add_library(bipp $<TARGET_OBJECTS:bipp_objects>)
add_library(bipp::bipp ALIAS bipp)

set_property(TARGET bipp PROPERTY VERSION ${BIPP_VERSION})
set_property(TARGET bipp PROPERTY SOVERSION ${BIPP_SO_VERSION})

target_link_libraries(bipp PRIVATE ${BIPP_EXTERNAL_LIBS})
target_link_libraries(bipp PUBLIC ${BIPP_EXTERNAL_LIBS_PUBLIC})
target_include_directories(bipp INTERFACE $<INSTALL_INTERFACE:include>) # for install(EXPORT ...)
target_include_directories(bipp INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>) # for export(...)
set_target_properties(bipp PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)

# generate export header to control symbol visibility
include(GenerateExportHeader)
generate_export_header(bipp)
configure_file("${CMAKE_CURRENT_BINARY_DIR}/bipp_export.h"
	"${PROJECT_BINARY_DIR}/bipp/bipp_export.h"
	COPYONLY
)

# set packge config names
get_target_property(_LIB_TYPE bipp TYPE)
if(_LIB_TYPE STREQUAL "STATIC_LIBRARY")
	set(BIPP_VERSION_FILE "bippStaticConfigVersion.cmake")
	set(BIPP_CONFIG_FILE "bippStaticConfig.cmake")
	set(BIPP_TARGETS_FILE "bippStaticTargets.cmake")
else()
	set(BIPP_VERSION_FILE "bippSharedConfigVersion.cmake")
	set(BIPP_CONFIG_FILE "bippSharedConfig.cmake")
	set(BIPP_TARGETS_FILE "bippSharedTargets.cmake")
endif()


# generate cmake package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	"${PROJECT_BINARY_DIR}/${BIPP_VERSION_FILE}"
	VERSION ${Upstream_VERSION}
	COMPATIBILITY AnyNewerVersion
)
export(TARGETS bipp NAMESPACE bipp:: FILE ${PROJECT_BINARY_DIR}/${BIPP_TARGETS_FILE})
configure_file(${PROJECT_SOURCE_DIR}/cmake/${BIPP_CONFIG_FILE}
	"${PROJECT_BINARY_DIR}/${BIPP_CONFIG_FILE}"
	@ONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/bippConfig.cmake
	"${PROJECT_BINARY_DIR}/bippConfig.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/bippConfigVersion.cmake
	"${PROJECT_BINARY_DIR}/bippConfigVersion.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/bippTargets.cmake
	"${PROJECT_BINARY_DIR}/bippTargets.cmake"
	COPYONLY
)

# installation commands
if(BIPP_INSTALL_LIB)
	install(TARGETS bipp DESTINATION ${CMAKE_INSTALL_LIBDIR} EXPORT bippTargets)
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/bipp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.f90")
	install(FILES ${PROJECT_BINARY_DIR}/bipp/config.h "${PROJECT_BINARY_DIR}/bipp/bipp_export.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bipp)
	install(EXPORT bippTargets NAMESPACE bipp:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bipp FILE ${BIPP_TARGETS_FILE})
	install(
	  FILES
		"${PROJECT_BINARY_DIR}/bippConfig.cmake"
		"${PROJECT_BINARY_DIR}/bippTargets.cmake"
		"${PROJECT_BINARY_DIR}/bippConfigVersion.cmake"
		"${PROJECT_BINARY_DIR}/${BIPP_CONFIG_FILE}"
		"${PROJECT_BINARY_DIR}/${BIPP_VERSION_FILE}"
	  DESTINATION
	      ${CMAKE_INSTALL_LIBDIR}/cmake/bipp
	)

	install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/modules"
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/bipp"
        FILES_MATCHING PATTERN "*.cmake")
endif()

if(BIPP_BUILD_APPS)
    add_subdirectory(apps)
endif()
