# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/bippSharedTargets.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/bippSharedTargets.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/bippStaticTargets.cmake")
endif()
