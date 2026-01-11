cmake_minimum_required(VERSION 3.8)
project(tauv_sim)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

find_package(Stonefish REQUIRED 1.6.0)
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tauv_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(tauv_sim
        src/tauv_sim.cpp
        src/tauv_simulation_manager.cpp
        src/osprey.cpp
        src/pressure_sensor_bridge.cpp
        src/context.cpp
        src/config_loader.cpp
        src/util.cpp
        src/thruster_bridge.cpp
)
target_include_directories(tauv_sim PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(tauv_sim PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(tauv_sim
        ament_index_cpp
        rclcpp
        tauv_msgs
)

target_link_libraries(tauv_sim
        Stonefish::Stonefish
        Eigen3::Eigen)

install(TARGETS
        tauv_sim
        DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
        assets
        launch
        config
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

