cmake_minimum_required(VERSION 3.8)
project(tauv_sim)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

set(STONEFISH_MIN_VERSION 1.6.0)
set(STONEFISH_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/stonefish")
option(TAUV_SIM_USE_VENDORED_STONEFISH "Build Stonefish from the bundled submodule" ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tauv_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(build_stonefish_from_source ${TAUV_SIM_USE_VENDORED_STONEFISH})
if (build_stonefish_from_source AND NOT EXISTS "${STONEFISH_SOURCE_DIR}/CMakeLists.txt")
    set(build_stonefish_from_source OFF)
endif ()
if (TAUV_SIM_USE_VENDORED_STONEFISH AND NOT build_stonefish_from_source)
    message(FATAL_ERROR "Requested vendored Stonefish, but the submodule was not found at ${STONEFISH_SOURCE_DIR}.")
endif ()

if (build_stonefish_from_source)
    # Build the vendored Stonefish submodule alongside tauv_sim so the target is always available.
    set(BUILD_TESTS OFF CACHE BOOL "Disable Stonefish test builds when vendored" FORCE)
    set(EMBED_RESOURCES OFF CACHE BOOL "Use packaged shaders instead of embedding resources" FORCE)
    add_subdirectory("${STONEFISH_SOURCE_DIR}" stonefish)
    target_include_directories(Stonefish PUBLIC
        $<BUILD_INTERFACE:${STONEFISH_SOURCE_DIR}/3rdparty>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/stonefish>
    )
    # Point Stonefish at the installed shader bundle (colcon creates the symlinks during install).
    target_compile_definitions(Stonefish PUBLIC SHADER_DIR_PATH=\"${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/shaders/\")
    if (NOT TARGET Stonefish::Stonefish AND TARGET Stonefish)
        add_library(Stonefish::Stonefish ALIAS Stonefish)
    endif ()
else ()
    find_package(Stonefish REQUIRED ${STONEFISH_MIN_VERSION})
endif ()

add_executable(tauv_sim
        src/tauv_sim.cpp
        src/tauv_simulation_manager.cpp
        src/osprey.cpp
        src/pressure_sensor_bridge.cpp
        src/imu_bridge.cpp
        src/context.cpp
        src/config_loader.cpp
        src/util.cpp
        src/thruster_bridge.cpp
        src/trajectory_loader.cpp
        src/osprey_sensors.cpp
        src/kinematic_osprey.cpp
)
target_include_directories(tauv_sim PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(tauv_sim PUBLIC c_std_99 cxx_std_20)  # Require C99 and C++20

ament_target_dependencies(tauv_sim
        ament_index_cpp
        rclcpp
        tauv_msgs
        sensor_msgs
)

target_link_libraries(tauv_sim
        Stonefish::Stonefish
        Eigen3::Eigen
        yaml-cpp)

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
