add_executable(bipp_synthesis bipp_synthesis.cpp)
target_link_libraries(bipp_synthesis PRIVATE bipp::bipp ${BIPP_APP_LIBRARIES})

if(BIPP_INSTALL_APPS)
	install(TARGETS bipp_synthesis DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()


set_target_properties(bipp_synthesis
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
