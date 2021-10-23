cmake_minimum_required(VERSION 3.0.2)
project(tauv_msgs)

find_package(catkin REQUIRED geometry_msgs sensor_msgs std_msgs vision_msgs jsk_recognition_msgs message_generation)

add_message_files(
  FILES
  BucketDetection.msg
  BucketList.msg
  FluidDepth.msg
  ControllerCmd.msg
  InertialVals.msg
  PidVals.msg
  SonarPulse.msg
  PoseGraphMeasurement.msg
)

add_service_files(
  FILES
  TuneInertial.srv
  TunePid.srv
  GetTraj.srv
)


generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
  vision_msgs
  jsk_recognition_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES
  CATKIN_DEPENDS
  geometry_msgs
  sensor_msgs
  vision_msgs
  jsk_recognition_msgs
  std_msgs
  message_runtime
# DEPENDS
)
