if (BIPP_BUNDLED_PYBIND11)
	FetchContent_Declare(
		pybind11
		URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
		URL_MD5 a04dead9c83edae6d84e2e343da7feeb
	)
	FetchContent_MakeAvailable(pybind11)
else()
	# Check if available as python package
	if (Python_EXECUTABLE)
		execute_process(
			COMMAND "${Python_EXECUTABLE}" -c
							"import pybind11; print(pybind11.get_cmake_dir())"
			OUTPUT_VARIABLE _pybind_dir
			OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
	endif()

	find_package(pybind11 CONFIG REQUIRED HINTS ${_pybind_dir})
endif()


pybind11_add_module(pybipp pybipp.cpp)
target_link_libraries(pybipp PRIVATE bipp_objects ${BIPP_EXTERNAL_LIBS_PUBLIC})
set_target_properties(pybipp PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bipp"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bipp"
	)


set(BIPP_PYTHON_SRC
	__init__.py
	__main__.py
	gram.py
	source.py
	parameter_estimator.py
	array.py
	instrument.py
	core.py
	frame.py
	statistics.py
	beamforming.py
	filter.py
	measurement_set.py
	selection.py
	imot_tools/io/s2image.py
	imot_tools/io/plot.py
	imot_tools/io/fits.py
	imot_tools/io/__init__.py
	imot_tools/__init__.py
	imot_tools/math/linalg.py
	imot_tools/math/special.py
	imot_tools/math/__init__.py
	imot_tools/math/stat.py
	imot_tools/math/func.py
	imot_tools/math/sphere/transform.py
	imot_tools/math/sphere/__init__.py
	imot_tools/math/sphere/interpolate.py
	imot_tools/math/sphere/grid.py
	imot_tools/util/argcheck.py
	imot_tools/util/array.py
	imot_tools/util/__init__.py
	imot_tools/phased_array/__init__.py
	imot_tools/data/io/colormap/aqua.csv
	imot_tools/data/io/colormap/lightbulb.csv
	imot_tools/data/io/colormap/lava.csv
	imot_tools/data/io/colormap/algue.csv
	imot_tools/data/io/colormap/jet_alt.csv
	imot_tools/data/io/colormap/fire.csv
	imot_tools/data/math/special/jv_threshold.csv
	imot_tools/data/math/special/jv_series_threshold.csv
	imot_tools/data/math/special/ive_threshold.csv
	imot_tools/data/math/special/spherical_jn_series_threshold.csv
	imot_tools/data/math/special/spherical_jn_threshold.csv
	data/instrument/LOFAR.csv
	data/instrument/MWA.csv
	apps/cli.py
	apps/create_dataset.py
	apps/create_image_prop.py
	apps/create_selection.py
	apps/image_synthesis.py
	apps/plot_images.py
	numpy_compat.py
	)


foreach(src IN LISTS BIPP_PYTHON_SRC)
	configure_file("${CMAKE_CURRENT_SOURCE_DIR}/bipp/${src}"
		"${CMAKE_CURRENT_BINARY_DIR}/bipp/${src}"
		COPYONLY
	)
endforeach()

if(BIPP_INSTALL_PYTHON)

	if(NOT BIPP_INSTALL_PYTHON_MODE STREQUAL "skbuild")
		foreach(src IN LISTS BIPP_PYTHON_SRC)
			get_filename_component(dir ${BIPP_INSTALL_PYTHON_PATH}/bipp/${src} DIRECTORY )
			install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/bipp/${src} DESTINATION ${dir})
		endforeach()
	endif()

	# make sure the module can find the library
	if(APPLE)
		set(BIPP_PYTHON_RPATH "@loader_path/_libs")
	else()
		target_link_options(pybipp PRIVATE "-Wl,--disable-new-dtags")
		set(BIPP_PYTHON_RPATH "$ORIGIN/_libs")
	endif()

	# destiantion dir must match module name in setup.py
	install(TARGETS pybipp DESTINATION ${BIPP_INSTALL_PYTHON_PATH})
endif()
