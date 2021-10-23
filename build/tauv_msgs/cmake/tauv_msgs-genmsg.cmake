# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "tauv_msgs: 8 messages, 3 services")

set(MSG_I_FLAGS "-Itauv_msgs:/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Ivision_msgs:/opt/ros/noetic/share/vision_msgs/cmake/../msg;-Ijsk_recognition_msgs:/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg;-Ipcl_msgs:/opt/ros/noetic/share/pcl_msgs/cmake/../msg;-Ijsk_footstep_msgs:/opt/ros/noetic/share/jsk_footstep_msgs/cmake/../msg;-Iactionlib_msgs:/opt/ros/noetic/share/actionlib_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(tauv_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg" "geometry_msgs/Quaternion:jsk_recognition_msgs/BoundingBox:geometry_msgs/Vector3:std_msgs/Header:geometry_msgs/Pose2D:geometry_msgs/Pose:vision_msgs/BoundingBox2D:geometry_msgs/Point:sensor_msgs/Image"
)

get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg" "geometry_msgs/Quaternion:jsk_recognition_msgs/BoundingBox:geometry_msgs/Vector3:std_msgs/Header:geometry_msgs/Pose2D:geometry_msgs/Pose:tauv_msgs/BucketDetection:geometry_msgs/Point:sensor_msgs/Image:vision_msgs/BoundingBox2D"
)

get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg" ""
)

get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg" ""
)

get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg" ""
)

get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" "std_msgs/Header:geometry_msgs/Point"
)

get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv" "tauv_msgs/InertialVals"
)

get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv" "tauv_msgs/PidVals"
)

get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv" "geometry_msgs/Quaternion:geometry_msgs/Vector3:geometry_msgs/Twist:std_msgs/Header:geometry_msgs/Pose:geometry_msgs/Point"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg/BoundingBox.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/vision_msgs/cmake/../msg/BoundingBox2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg/BoundingBox.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/vision_msgs/cmake/../msg/BoundingBox2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)

### Generating Services
_generate_srv_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv"
  "${MSG_I_FLAGS}"
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv"
  "${MSG_I_FLAGS}"
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)

### Generating Module File
_generate_module_cpp(tauv_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(tauv_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(tauv_msgs_generate_messages tauv_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_msgs_gencpp)
add_dependencies(tauv_msgs_gencpp tauv_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg/BoundingBox.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/vision_msgs/cmake/../msg/BoundingBox2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg/BoundingBox.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/vision_msgs/cmake/../msg/BoundingBox2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)

### Generating Services
_generate_srv_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv"
  "${MSG_I_FLAGS}"
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv"
  "${MSG_I_FLAGS}"
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)

### Generating Module File
_generate_module_eus(tauv_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(tauv_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(tauv_msgs_generate_messages tauv_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_msgs_geneus)
add_dependencies(tauv_msgs_geneus tauv_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg/BoundingBox.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/vision_msgs/cmake/../msg/BoundingBox2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg/BoundingBox.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/vision_msgs/cmake/../msg/BoundingBox2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)

### Generating Services
_generate_srv_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv"
  "${MSG_I_FLAGS}"
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv"
  "${MSG_I_FLAGS}"
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)

### Generating Module File
_generate_module_lisp(tauv_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(tauv_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(tauv_msgs_generate_messages tauv_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_msgs_genlisp)
add_dependencies(tauv_msgs_genlisp tauv_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg/BoundingBox.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/vision_msgs/cmake/../msg/BoundingBox2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg/BoundingBox.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/vision_msgs/cmake/../msg/BoundingBox2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)

### Generating Services
_generate_srv_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv"
  "${MSG_I_FLAGS}"
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv"
  "${MSG_I_FLAGS}"
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)

### Generating Module File
_generate_module_nodejs(tauv_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(tauv_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(tauv_msgs_generate_messages tauv_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_msgs_gennodejs)
add_dependencies(tauv_msgs_gennodejs tauv_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg/BoundingBox.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/vision_msgs/cmake/../msg/BoundingBox2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg/BoundingBox.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose2D.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/vision_msgs/cmake/../msg/BoundingBox2D.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)

### Generating Services
_generate_srv_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv"
  "${MSG_I_FLAGS}"
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv"
  "${MSG_I_FLAGS}"
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)

### Generating Module File
_generate_module_py(tauv_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(tauv_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(tauv_msgs_generate_messages tauv_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/BucketList.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/ControllerCmd.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/InertialVals.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PidVals.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TuneInertial.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/TunePid.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/srv/GetTraj.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_msgs_genpy)
add_dependencies(tauv_msgs_genpy tauv_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(tauv_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(tauv_msgs_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(tauv_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET vision_msgs_generate_messages_cpp)
  add_dependencies(tauv_msgs_generate_messages_cpp vision_msgs_generate_messages_cpp)
endif()
if(TARGET jsk_recognition_msgs_generate_messages_cpp)
  add_dependencies(tauv_msgs_generate_messages_cpp jsk_recognition_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(tauv_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(tauv_msgs_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(tauv_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET vision_msgs_generate_messages_eus)
  add_dependencies(tauv_msgs_generate_messages_eus vision_msgs_generate_messages_eus)
endif()
if(TARGET jsk_recognition_msgs_generate_messages_eus)
  add_dependencies(tauv_msgs_generate_messages_eus jsk_recognition_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(tauv_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(tauv_msgs_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(tauv_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET vision_msgs_generate_messages_lisp)
  add_dependencies(tauv_msgs_generate_messages_lisp vision_msgs_generate_messages_lisp)
endif()
if(TARGET jsk_recognition_msgs_generate_messages_lisp)
  add_dependencies(tauv_msgs_generate_messages_lisp jsk_recognition_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(tauv_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(tauv_msgs_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(tauv_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET vision_msgs_generate_messages_nodejs)
  add_dependencies(tauv_msgs_generate_messages_nodejs vision_msgs_generate_messages_nodejs)
endif()
if(TARGET jsk_recognition_msgs_generate_messages_nodejs)
  add_dependencies(tauv_msgs_generate_messages_nodejs jsk_recognition_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(tauv_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(tauv_msgs_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(tauv_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET vision_msgs_generate_messages_py)
  add_dependencies(tauv_msgs_generate_messages_py vision_msgs_generate_messages_py)
endif()
if(TARGET jsk_recognition_msgs_generate_messages_py)
  add_dependencies(tauv_msgs_generate_messages_py jsk_recognition_msgs_generate_messages_py)
endif()
