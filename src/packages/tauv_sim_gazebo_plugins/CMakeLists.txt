cmake_minimum_required(VERSION 2.8.3)
project(tauv_sim_gazebo_plugins)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS gazebo_dev roscpp std_msgs tauv_msgs)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    tauv_thruster_plugin
    tauv_dvl_plugin
  CATKIN_DEPENDS
    std_msgs
    roscpp
)

include_directories(${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS})

link_directories(${catkin_LIBRARY_DIRS})

add_library(tauv_thruster_plugin SHARED src/thruster_plugin.cpp)
add_library(tauv_dvl_plugin SHARED src/dvl_plugin.cpp)
target_link_libraries(tauv_thruster_plugin ${catkin_LIBRARIES})
target_link_libraries(tauv_dvl_plugin ${catkin_LIBRARIES})
add_dependencies(tauv_thruster_plugin ${catkin_EXPORTED_TARGETS})
add_dependencies(tauv_dvl_plugin ${catkin_EXPORTED_TARGETS})

install(TARGETS tauv_thruster_plugin
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
