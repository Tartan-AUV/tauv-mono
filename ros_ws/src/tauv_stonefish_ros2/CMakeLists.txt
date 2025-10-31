cmake_minimum_required(VERSION 3.8)
project(tauv_stonefish_ros2)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Stonefish REQUIRED)

add_executable(tauv_sim 
  src/tauv_sim.cpp
  src/tauv_stonefish_ros2/ROS2GraphicalSimulationApp.cpp
  src/tauv_stonefish_ros2/ROS2SimulationManager.cpp
)
ament_target_dependencies(tauv_sim rclcpp std_msgs)

target_include_directories(tauv_sim PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(tauv_sim PRIVATE /usr/local/include/Stonefish)

link_directories(/usr/local/lib)
target_link_libraries(tauv_sim Stonefish::Stonefish)

install(TARGETS
  tauv_sim
  DESTINATION lib/${PROJECT_NAME})

ament_package()