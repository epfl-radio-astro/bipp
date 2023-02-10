set(BIPP_TEST_LIBRARIES)
set(BIPP_TEST_INCLUDE_DIRS)

set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
mark_as_advanced(BUILD_GMOCK INSTALL_GTEST)

include(FetchContent)

# add googletest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  message(STATUS "Downloading Google Test repository...")
  FetchContent_Populate(googletest)
endif()
add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
list(APPEND BIPP_TEST_LIBRARIES gtest_main)

# add json parser
set(JSON_Install OFF CACHE BOOL "")
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.10.5
)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  message(STATUS "Downloading json repository...")
  FetchContent_Populate(json)
endif()
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR})
list(APPEND BIPP_TEST_LIBRARIES nlohmann_json::nlohmann_json)


# test executables
add_executable(run_tests
  run_tests.cpp
  test_nufft_synthesis_lofar.cpp
  test_standard_synthesis_lofar.cpp
  test_domain_partition.cpp
)
target_link_libraries(run_tests PRIVATE ${BIPP_TEST_LIBRARIES} bipp_test)
target_compile_options(run_tests PRIVATE -DBIPP_TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data")

