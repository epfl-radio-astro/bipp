set(BIPP_TEST_LIBRARIES)
set(BIPP_TEST_INCLUDE_DIRS)

set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
mark_as_advanced(BUILD_GMOCK INSTALL_GTEST)


set(JSON_Install OFF CACHE BOOL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")

if(BIPP_BUNDLED_GOOGLETEST)
  # add googletest
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
    URL_MD5 95b29f0038ec84a611df951d74d99897
  )
  FetchContent_MakeAvailable(googletest)
else()
  find_package(googletest CONFIG REQUIRED)
endif()

if(BIPP_BUNDLED_JSON)
  # add json parser
  FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz
    URL_MD5 e8d56bc54621037842ee9f0aeae27746
  )
  FetchContent_MakeAvailable(json)
else()
  find_package(nlohmann_json CONFIG REQUIRED)
endif()

list(APPEND BIPP_TEST_LIBRARIES gtest_main nlohmann_json::nlohmann_json)

# test executables
add_executable(run_tests
  run_tests.cpp
  test_nufft_synthesis_lofar.cpp
  test_standard_synthesis_lofar.cpp
  test_domain_partition.cpp
)
target_link_libraries(run_tests PRIVATE ${BIPP_TEST_LIBRARIES} bipp_static)
target_compile_options(run_tests PRIVATE -DBIPP_TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data")

