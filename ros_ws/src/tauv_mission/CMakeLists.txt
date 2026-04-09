cmake_minimum_required(VERSION 3.8)
project(tauv_mission)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# Python dependencies
# find_package(rclpy REQUIRED)

include_directories(include)

add_executable(trajectory_manager src/trajectory_manager.cpp)

target_compile_features(trajectory_manager PUBLIC c_std_99 cxx_std_20)

ament_target_dependencies(trajectory_manager
    rclcpp
    nav_msgs
    geometry_msgs
    ament_index_cpp
)

target_link_libraries(trajectory_manager
    yaml-cpp
)

install(TARGETS
        trajectory_manager
        DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
        launch
        config
        DESTINATION share/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  tauv_mission/teleop.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
