cmake_minimum_required(VERSION 3.8)
project(tauv_common)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tauv_msgs REQUIRED)

# depth_converter executable
add_executable(depth_converter
    src/depth_converter.cpp
)
target_include_directories(depth_converter PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_compile_features(depth_converter PUBLIC c_std_99 cxx_std_20)

ament_target_dependencies(depth_converter
    rclcpp
    sensor_msgs
    nav_msgs
)

# dvl_converter executable
add_executable(dvl_converter
    src/dvl_converter.cpp
)
target_include_directories(dvl_converter PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_compile_features(dvl_converter PUBLIC c_std_99 cxx_std_20)

ament_target_dependencies(dvl_converter
    rclcpp
    tauv_msgs
    nav_msgs
)

install(TARGETS
    depth_converter
    dvl_converter
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    config
    DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
