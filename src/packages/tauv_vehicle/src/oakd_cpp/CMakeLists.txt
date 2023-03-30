set (CMAKE_CXX_STANDARD 17)

set(depthai_DIR /depthai-core/build)
find_package(depthai CONFIG REQUIRED)

find_package(PkgConfig)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

include_directories(./ ${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS})
library(pipeline_lib Pipeline.cpp)
# library(oakd_node OAKDNode.cpp)

# node(oakd_cpp main.cpp oakd_node pipeline_lib)
node(oakd_cpp main.cpp pipeline_lib)

target_link_libraries(oakd_cpp ${GST_LIBRARIES} ${GST_APP_LIBRARIES} depthai::opencv)
