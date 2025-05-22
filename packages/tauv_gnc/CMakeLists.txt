cmake_minimum_required(VERSION 3.22)
project(tauv_gnc)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(gtsam)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tauv_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(state_estimation
        src/state_estimation/StateEstimation.cpp
        src/state_estimation/StateEstimation.h
        src/state_estimation/DVLFactor.cpp
        src/state_estimation/DVLFactor.h
        src/state_estimation/DepthFactor.cpp
        src/state_estimation/DepthFactor.h)

add_dependencies(state_estimation gtsam)

target_include_directories(state_estimation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/state_estimation>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_compile_features(state_estimation PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17


ament_target_dependencies(
  state_estimation
  "rclcpp"
  "rclpy"
  "std_msgs"
  "tauv_msgs"
  "nav_msgs"
  "sensor_msgs"
)

target_link_libraries(state_estimation gtsam)

# Install executables
install(TARGETS state_estimation
  DESTINATION lib/${PROJECT_NAME})

# ENABLE FOR PYTHON
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/emulator.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
