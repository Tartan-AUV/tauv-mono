# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(WARNING "Invoking generate_messages() without having added any message or service file before.
You should either add add_message_files() and/or add_service_files() calls or remove the invocation of generate_messages().")
message(STATUS "tauv_vehicle: 0 messages, 0 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Itauv_msgs:/home/tom/workspaces/tauv_ws/src/packages/tauv_msgs/msg;-Ivision_msgs:/opt/ros/noetic/share/vision_msgs/cmake/../msg;-Ijsk_recognition_msgs:/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg;-Ipcl_msgs:/opt/ros/noetic/share/pcl_msgs/cmake/../msg;-Ijsk_footstep_msgs:/opt/ros/noetic/share/jsk_footstep_msgs/cmake/../msg;-Iactionlib_msgs:/opt/ros/noetic/share/actionlib_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(tauv_vehicle_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services

### Generating Module File
_generate_module_cpp(tauv_vehicle
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_vehicle
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(tauv_vehicle_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(tauv_vehicle_generate_messages tauv_vehicle_generate_messages_cpp)

# add dependencies to all check dependencies targets

# target for backward compatibility
add_custom_target(tauv_vehicle_gencpp)
add_dependencies(tauv_vehicle_gencpp tauv_vehicle_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_vehicle_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services

### Generating Module File
_generate_module_eus(tauv_vehicle
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_vehicle
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(tauv_vehicle_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(tauv_vehicle_generate_messages tauv_vehicle_generate_messages_eus)

# add dependencies to all check dependencies targets

# target for backward compatibility
add_custom_target(tauv_vehicle_geneus)
add_dependencies(tauv_vehicle_geneus tauv_vehicle_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_vehicle_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services

### Generating Module File
_generate_module_lisp(tauv_vehicle
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_vehicle
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(tauv_vehicle_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(tauv_vehicle_generate_messages tauv_vehicle_generate_messages_lisp)

# add dependencies to all check dependencies targets

# target for backward compatibility
add_custom_target(tauv_vehicle_genlisp)
add_dependencies(tauv_vehicle_genlisp tauv_vehicle_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_vehicle_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services

### Generating Module File
_generate_module_nodejs(tauv_vehicle
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_vehicle
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(tauv_vehicle_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(tauv_vehicle_generate_messages tauv_vehicle_generate_messages_nodejs)

# add dependencies to all check dependencies targets

# target for backward compatibility
add_custom_target(tauv_vehicle_gennodejs)
add_dependencies(tauv_vehicle_gennodejs tauv_vehicle_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_vehicle_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services

### Generating Module File
_generate_module_py(tauv_vehicle
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_vehicle
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(tauv_vehicle_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(tauv_vehicle_generate_messages tauv_vehicle_generate_messages_py)

# add dependencies to all check dependencies targets

# target for backward compatibility
add_custom_target(tauv_vehicle_genpy)
add_dependencies(tauv_vehicle_genpy tauv_vehicle_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_vehicle_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_vehicle)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_vehicle
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(tauv_vehicle_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(tauv_vehicle_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(tauv_vehicle_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET tauv_msgs_generate_messages_cpp)
  add_dependencies(tauv_vehicle_generate_messages_cpp tauv_msgs_generate_messages_cpp)
endif()
if(TARGET std_srvs_generate_messages_cpp)
  add_dependencies(tauv_vehicle_generate_messages_cpp std_srvs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_vehicle)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_vehicle
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(tauv_vehicle_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(tauv_vehicle_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(tauv_vehicle_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET tauv_msgs_generate_messages_eus)
  add_dependencies(tauv_vehicle_generate_messages_eus tauv_msgs_generate_messages_eus)
endif()
if(TARGET std_srvs_generate_messages_eus)
  add_dependencies(tauv_vehicle_generate_messages_eus std_srvs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_vehicle)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_vehicle
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(tauv_vehicle_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(tauv_vehicle_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(tauv_vehicle_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET tauv_msgs_generate_messages_lisp)
  add_dependencies(tauv_vehicle_generate_messages_lisp tauv_msgs_generate_messages_lisp)
endif()
if(TARGET std_srvs_generate_messages_lisp)
  add_dependencies(tauv_vehicle_generate_messages_lisp std_srvs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_vehicle)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_vehicle
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(tauv_vehicle_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(tauv_vehicle_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(tauv_vehicle_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET tauv_msgs_generate_messages_nodejs)
  add_dependencies(tauv_vehicle_generate_messages_nodejs tauv_msgs_generate_messages_nodejs)
endif()
if(TARGET std_srvs_generate_messages_nodejs)
  add_dependencies(tauv_vehicle_generate_messages_nodejs std_srvs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_vehicle)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_vehicle\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_vehicle
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(tauv_vehicle_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(tauv_vehicle_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(tauv_vehicle_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET tauv_msgs_generate_messages_py)
  add_dependencies(tauv_vehicle_generate_messages_py tauv_msgs_generate_messages_py)
endif()
if(TARGET std_srvs_generate_messages_py)
  add_dependencies(tauv_vehicle_generate_messages_py std_srvs_generate_messages_py)
endif()
