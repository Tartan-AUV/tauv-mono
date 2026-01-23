cmake_minimum_required(VERSION 3.8)
project(tauv_mission)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

set(STONEFISH_MIN_VERSION 1.6.0)
set(STONEFISH_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/stonefish")
option(TAUV_SIM_USE_VENDORED_STONEFISH "Build Stonefish from the bundled submodule" ON)

# C++ dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(tauv_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# python dependencies
# find_package(rclpy REQUIRED)

# Include Cpp "include" directory
include_directories(include)

# Create Cpp executable
add_executable(trajectory_manager src/trajectory_manager.cpp)

target_compile_features(trajectory_manager PUBLIC c_std_99 cxx_std_20)

target_link_libraries(trajectory_manager PUBLIC
      ${geometry_msgs_TARGETS}
      ${sensor_msgs_TARGETS}
      ${tauv_msgs_TARGETS}
      rclcpp_action::rclcpp_action
      rclcpp::rclcpp
      sensor_msgs::sensor_msgs_library
)

install(TARGETS
        trajectory_manager
        DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
        launch
        DESTINATION share/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  tauv_mission/teleop.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()