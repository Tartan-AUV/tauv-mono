cmake_minimum_required(VERSION 3.0.2)
project(compressed_streaming)

## PkgConfig and gstreamer
find_package(PkgConfig)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs image_transport)

## Declare ROS messages and services

## Generate added messages and services

## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(src include ${catkin_INCLUDE_DIRS} ${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS})

add_executable(compressed_streaming_node src/compressed_streaming_node.cpp src/Pipeline.cpp src/Pipeline.h)
target_link_libraries(compressed_streaming_node ${catkin_LIBRARIES} ${GST_LIBRARIES} ${GST_APP_LIBRARIES})
#add_dependencies(compressed_streaming_node )