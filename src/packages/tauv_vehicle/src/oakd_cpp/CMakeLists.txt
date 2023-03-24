set (CMAKE_CXX_STANDARD 17)

find_package(PkgConfig)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

include_directories(./ ${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS})
library(pipeline_lib Pipeline.cpp)

node(oakd_cpp main.cpp pipeline_lib)

target_link_libraries(compressed_streaming ${GST_LIBRARIES} ${GST_APP_LIBRARIES})
