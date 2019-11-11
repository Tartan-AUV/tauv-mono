cmake_minimum_required(VERSION 2.8.3)
project(tauv_msgs)

find_package(catkin REQUIRED geometry_msgs sensor_msgs std_msgs message_generation)

add_message_files(
  FILES
  FluidDepth.msg
  ControllerList.msg
  ControllerStatus.msg
)

add_service_files(
  FILES
  SetController.srv
)


generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES
  CATKIN_DEPENDS
  geometry_msgs
  sensor_msgs
  std_msgs
  message_runtime
# DEPENDS
)
