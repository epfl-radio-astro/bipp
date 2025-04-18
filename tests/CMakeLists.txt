# test executables
add_executable(run_tests
  run_tests.cpp
  test_nufft_synthesis_lofar.cpp
  test_domain_partition.cpp
)
target_link_libraries(run_tests PRIVATE ${BIPP_TEST_LIBRARIES} bipp_objects ${BIPP_EXTERNAL_LIBS} ${BIPP_EXTERNAL_LIBS_PUBLIC})
target_include_directories(run_tests PRIVATE ${BIPP_INCLUDE_DIRS})
target_compile_options(run_tests PRIVATE -DBIPP_TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data")

