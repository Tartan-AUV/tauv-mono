cmake_minimum_required(VERSION 3.22)
project(tauv_gnc)
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

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
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(state_estimator
        src/state_estimator/state_estimator.cpp
)
add_executable(depth_estimator
        src/depth_estimator/depth_estimator.cpp
)

add_dependencies(state_estimator gtsam)

target_include_directories(state_estimator PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/state_estimation>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_include_directories(depth_estimator PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_compile_features(state_estimator PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(depth_estimator PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(
        state_estimator
        "rclcpp"
        "rclpy"
        "std_msgs"
        "tauv_msgs"
        "nav_msgs"
        "sensor_msgs"
        "tf2"
        "tf2_ros"
        "tf2_geometry_msgs"
        "geometry_msgs"
)
ament_target_dependencies(depth_estimator
        "rclcpp"
        "std_msgs"
        "tauv_msgs"
        "std_srvs"
)

target_link_libraries(state_estimator gtsam)

# Install executables
install(TARGETS state_estimator depth_estimator
        DESTINATION lib/${PROJECT_NAME})

# ENABLE FOR PYTHON
#ament_python_install_package(${PROJECT_NAME})
#install(PROGRAMS
#        DESTINATION lib/${PROJECT_NAME}
#)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()
