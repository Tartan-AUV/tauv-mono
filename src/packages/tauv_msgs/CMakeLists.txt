cmake_minimum_required(VERSION 2.8.3)
project(tauv_msgs)

find_package(catkin REQUIRED geometry_msgs sensor_msgs std_msgs vision_msgs jsk_recognition_msgs message_generation)

add_message_files(
  FILES
  AlarmReport.msg
  AlarmWithMessage.msg
  ControllerCmd.msg
  ControllerDebug.msg
  ControlsTunings.msg
  FeatureDetection.msg
  FluidDepth.msg
  DynamicsTunings.msg
  GateDetection.msg
  Message.msg
  SonarPulse.msg
  Servos.msg
  RegisterMeasurement.msg
  FeatureDetections.msg
  PoseGraphMeasurement.msg
  ReadableAlarmReport.msg
  Thrust.msg
  TrajPoint.msg
  Battery.msg
  XsensImuSync.msg
  XsensImuData.msg
  TeledyneDvlData.msg
  Pose.msg
)

add_service_files(
  FILES
  MapFind.srv
  MapFindClosest.srv
  SyncAlarms.srv
  TuneDynamics.srv
  TuneControls.srv
  GetTraj.srv
  SetTargetPose.srv
  SetPose.srv
  RunBasicMission.srv
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
