# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "tauv_joy: 0 messages, 1 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(tauv_joy_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv" NAME_WE)
add_custom_target(_tauv_joy_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_joy" "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(tauv_joy
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_joy
)

### Generating Module File
_generate_module_cpp(tauv_joy
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_joy
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(tauv_joy_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(tauv_joy_generate_messages tauv_joy_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv" NAME_WE)
add_dependencies(tauv_joy_generate_messages_cpp _tauv_joy_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_joy_gencpp)
add_dependencies(tauv_joy_gencpp tauv_joy_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_joy_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(tauv_joy
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_joy
)

### Generating Module File
_generate_module_eus(tauv_joy
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_joy
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(tauv_joy_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(tauv_joy_generate_messages tauv_joy_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv" NAME_WE)
add_dependencies(tauv_joy_generate_messages_eus _tauv_joy_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_joy_geneus)
add_dependencies(tauv_joy_geneus tauv_joy_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_joy_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(tauv_joy
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_joy
)

### Generating Module File
_generate_module_lisp(tauv_joy
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_joy
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(tauv_joy_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(tauv_joy_generate_messages tauv_joy_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv" NAME_WE)
add_dependencies(tauv_joy_generate_messages_lisp _tauv_joy_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_joy_genlisp)
add_dependencies(tauv_joy_genlisp tauv_joy_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_joy_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(tauv_joy
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_joy
)

### Generating Module File
_generate_module_nodejs(tauv_joy
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_joy
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(tauv_joy_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(tauv_joy_generate_messages tauv_joy_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv" NAME_WE)
add_dependencies(tauv_joy_generate_messages_nodejs _tauv_joy_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_joy_gennodejs)
add_dependencies(tauv_joy_gennodejs tauv_joy_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_joy_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(tauv_joy
  "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_joy
)

### Generating Module File
_generate_module_py(tauv_joy
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_joy
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(tauv_joy_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(tauv_joy_generate_messages tauv_joy_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/tom/workspaces/tauv_ws/src/packages/tauv_gui/ext/joystick_drivers/tauv_joy/srv/JoyConnect.srv" NAME_WE)
add_dependencies(tauv_joy_generate_messages_py _tauv_joy_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_joy_genpy)
add_dependencies(tauv_joy_genpy tauv_joy_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_joy_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_joy)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_joy
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(tauv_joy_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_joy)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_joy
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(tauv_joy_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_joy)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_joy
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(tauv_joy_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_joy)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_joy
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(tauv_joy_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_joy)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_joy\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_joy
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(tauv_joy_generate_messages_py std_msgs_generate_messages_py)
endif()
