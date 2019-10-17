cmake_minimum_required(VERSION 2.8.3)
project(tauv_common)

add_compile_options(-Wall -Werror -std=c++11)

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    message_generation
    sensor_msgs
    geometry_msgs
    message_filters
    tf
  )
find_package(yaml-cpp REQUIRED)

catkin_python_setup()

## Generate services in the 'srv' folder
add_service_files(
    FILES
    GetThrusterCurve.srv
    ThrusterManagerInfo.srv
    SetThrusterManagerConfig.srv
    GetThrusterManagerConfig.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
)

catkin_package(CATKIN_DEPENDS
    message_runtime
    std_msgs
    roscpp
    rospy
    sensor_msgs
    geometry_msgs
    tf
    tauv_config
    message_filters
  )

include_directories(
        src
        ${catkin_INCLUDE_DIRS}
)

#catkin_install_python(PROGRAMS scripts/thruster_allocator
#                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        PATTERN "*~" EXCLUDE)

# Macros for declaring ros nodes and libraries
include(cmake/ros-node.cmake)
include(cmake/cppcheck.cmake)
include(cmake/clang-format.cmake)
include(cmake/pylint.cmake)

# define nodes here
# eg: add_subdirectory(src/imu_calibrator)


