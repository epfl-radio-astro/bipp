# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/bippSharedConfigVersion.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/bippSharedConfigVersion.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/bippStaticConfigVersion.cmake")
endif()
