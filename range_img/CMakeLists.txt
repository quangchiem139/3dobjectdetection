cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(range_image_based_object_recognition)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (range_image_based_object_recognition range_image_visualization.cpp)
target_link_libraries (range_image_based_object_recognition ${PCL_LIBRARIES})
