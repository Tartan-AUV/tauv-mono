cmake_minimum_required(VERSION 3.8)
project(tauv_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/WaterlinkedDvlFrame.msg"
  "msg/ThrusterSetpoint.msg"
  "msg/EscTelemetry.msg"
  "msg/DepthSensorFrame.msg"
  "msg/Depth.msg"
  "msg/TargetThrust.msg"
  "msg/NavigationState.msg"
  "msg/ControllerCommand.msg"
  "msg/CommanderDebug.msg"
  "msg/ControllerDebug.msg"
  "srv/Goto.srv"
  "srv/SetTrajectory.srv"
  "action/GotoVelocity.action"
  DEPENDENCIES std_msgs geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
