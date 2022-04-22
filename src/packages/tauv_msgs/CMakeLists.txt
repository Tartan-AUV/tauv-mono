cmake_minimum_required(VERSION 2.8.3)
project(tauv_msgs)

find_package(catkin REQUIRED geometry_msgs sensor_msgs std_msgs vision_msgs jsk_recognition_msgs message_generation)

add_message_files(
  FILES
  AlarmReport.msg
  AlarmWithMessage.msg
  BucketDetection.msg
  BucketList.msg
  FluidDepth.msg
  ControllerCmd.msg
  DynamicsTunings.msg
  ControlsTunings.msg
  SonarPulse.msg
  PoseGraphMeasurement.msg
  Thrust.msg
  Battery.msg
  XsensImuSync.msg
  XsensImuData.msg
  TeledyneDvlData.msg
  Pose.msg
)

add_service_files(
  FILES
  SyncAlarms.srv
  TuneDynamics.srv
  TuneControls.srv
  GetTraj.srv
  HoldPose.srv
  SetPose.srv
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
