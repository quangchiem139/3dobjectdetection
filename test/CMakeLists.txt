cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
find_package(PCL 1.6 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(pcd_write pcd_write.cpp)
target_link_libraries(pcd_write ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})

