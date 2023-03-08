set(BIPP_SOURCE_FILES
	eigensolver.cpp
	gram_matrix.cpp
	standard_synthesis.cpp
	context.cpp
	nufft_synthesis.cpp
	memory/allocator_factory.cpp
	host/eigensolver.cpp
	host/gram_matrix.cpp
	host/nufft_synthesis.cpp
	host/nufft_3d3.cpp
	host/virtual_vis.cpp
	host/kernels/gemmexp.cpp
	host/standard_synthesis.cpp
	)

if(BIPP_CUDA OR BIPP_ROCM)
	list(APPEND BIPP_SOURCE_FILES
		gpu/eigensolver.cpp
		gpu/gram_matrix.cpp
		gpu/nufft_3d3.cpp
		gpu/nufft_synthesis.cpp
		gpu/virtual_vis.cpp
		gpu/standard_synthesis.cpp
		gpu/domain_partition.cu
		gpu/util/solver_api.cpp
		gpu/kernels/reverse.cu
		gpu/kernels/gram.cu
		gpu/kernels/scale_matrix.cu
		gpu/kernels/apply_filter.cu
		gpu/kernels/add_vector.cu
		gpu/kernels/gemmexp.cu
		gpu/kernels/center_vector.cu
		gpu/kernels/min_max_element.cu
		)
endif()

# set all files to CXX, such that .cu files are compiled by the CXX compiler with ROCM enabled
if(BIPP_ROCM)
    set_source_files_properties(${BIPP_SOURCE_FILES} PROPERTIES LANGUAGE CXX)
endif()


# Creates bipp library with given name. All common target modifications should be done here.
macro(bipp_create_library _TARGET_NAME)
	add_library(${_TARGET_NAME} ${BIPP_LIBRARY_TYPE}
		${BIPP_SOURCE_FILES}
		)

	set_property(TARGET ${_TARGET_NAME} PROPERTY VERSION ${BIPP_VERSION})
	set_property(TARGET ${_TARGET_NAME} PROPERTY SOVERSION ${BIPP_SO_VERSION})


	# Don't export any symbols of external static libaries. Only works on linux.
	if(UNIX AND NOT APPLE)
		if(${CMAKE_VERSION} VERSION_LESS "3.13.5") 
			target_link_libraries(${_TARGET_NAME} PRIVATE "-Wl,--exclude-libs,ALL")
		else()
			target_link_options(${_TARGET_NAME} PRIVATE "-Wl,--exclude-libs,ALL")
		endif()
	endif()

	# All .cu files are self-contained. Device linking can have issues with propageted linker flags of other targets like MPI.
	if(BIPP_CUDA)
		set_property(TARGET ${_TARGET_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
		set_property(TARGET ${_TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
	endif()
	# make static libraries linkable into shared ones
	set_property(TARGET ${_TARGET_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

	target_compile_options(${_TARGET_NAME} PRIVATE ${BIPP_FLAGS})
	target_include_directories(${_TARGET_NAME} PRIVATE ${BIPP_INCLUDE_DIRS})
	target_link_libraries(${_TARGET_NAME} PRIVATE ${BIPP_EXTERNAL_LIBS})

	target_include_directories(${_TARGET_NAME} INTERFACE $<INSTALL_INTERFACE:include>) # for install(EXPORT ...)
	target_include_directories(${_TARGET_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>) # for export(...)

endmacro()

bipp_create_library(bipp)
set_target_properties(bipp PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)

if(BIPP_BUILD_TESTS)
	# create library with default visibility if tests are build, to allow linking to internal symbols
	bipp_create_library(bipp_test)
	set_target_properties(bipp_test PROPERTIES VISIBILITY_INLINES_HIDDEN FALSE CXX_VISIBILITY_PRESET default)
	target_include_directories(bipp_test INTERFACE ${BIPP_INCLUDE_DIRS})
	target_link_libraries(bipp_test INTERFACE ${BIPP_EXTERNAL_LIBS})
	target_compile_options(bipp_test PUBLIC -DBIPP_STATIC_DEFINE)
endif()

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
if(BIPP_INSTALL STREQUAL "LIB")
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
