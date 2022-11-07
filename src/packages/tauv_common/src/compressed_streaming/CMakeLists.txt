cmake_minimum_required(VERSION 3.0.2)
project(compressed_streaming)

## PkgConfig and gstreamer
find_package(PkgConfig)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

include_directories(${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS})
node(compressed_streaming main.cpp)
target_link_libraries(compressed_streaming ${GST_LIBRARIES} ${GST_APP_LIBRARIES})
