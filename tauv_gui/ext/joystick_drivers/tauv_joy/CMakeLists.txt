# http://ros.org/doc/groovy/api/catkin/html/user_guide/supposed.html
cmake_minimum_required(VERSION 2.8.3)
project(tauv_joy)

# Load catkin and all dependencies required for this package
set(CATKIN_DEPS roscpp diagnostic_updater std_msgs sensor_msgs message_generation message_runtime)
find_package(catkin REQUIRED ${CATKIN_DEPS})

add_service_files(
  FILES
  JoyConnect.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package(DEPENDS ${CATKIN_DEPS})

# Look for <linux/joystick.h>
include(CheckIncludeFiles)
check_include_files(linux/joystick.h HAVE_LINUX_JOYSTICK_H)

if(HAVE_LINUX_JOYSTICK_H)
  include_directories(msg/cpp ${catkin_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
  add_executable(tauv_joy_node src/tauv_joy_node.cpp)
  add_dependencies(tauv_joy_node tauv_joy_generate_messages_cpp)
  target_link_libraries(tauv_joy_node ${catkin_LIBRARIES})
else(HAVE_LINUX_JOYSTICK_H)
  message("Warning: no <linux/joystick.h>; won't build joy node")
endif(HAVE_LINUX_JOYSTICK_H)

#catkin_add_nosetests(test/test_joy_msg_migration.py)

# Install targets
if(HAVE_LINUX_JOYSTICK_H)
install(TARGETS tauv_joy_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endif(HAVE_LINUX_JOYSTICK_H)

install(DIRECTORY migration_rules scripts config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS)
