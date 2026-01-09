cmake_minimum_required(VERSION 3.22)
project(tauv_vehicle)
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tauv_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vpi REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

# Check for OpenCV CUDA support
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs cudaimgproc cudawarping)

set(CMAKE_CXX_STANDARD 17)

# Add C++ executable for lucid_driver
add_executable(lucid_driver src/lucid_driver.cpp)

target_include_directories(lucid_driver PRIVATE 
            /opt/arena/ArenaSDK_Linux_ARM64/include/Arena
            /opt/arena/ArenaSDK_Linux_ARM64/GenICam/library/CPP/include)

ament_target_dependencies(lucid_driver
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  vpi
  Boost
)

# Add Arena SDK library directories
target_link_directories(lucid_driver PRIVATE
  /opt/arena/ArenaSDK_Linux_ARM64/lib
  /opt/arena/ArenaSDK_Linux_ARM64/GenICam/library/lib/Linux64_ARM
  /opt/arena/ArenaSDK_Linux_ARM64/ffmpeg/lib
)

# Link Arena SDK libraries
target_link_libraries(lucid_driver
  arena
  arenac
  savec
  save
  lucidlog
  gentl
  GenApi_gcc54_v3_3_LUCID
  GCBase_gcc54_v3_3_LUCID
  Log_gcc54_v3_3_LUCID
  MathParser_gcc54_v3_3_LUCID
  NodeMapData_gcc54_v3_3_LUCID
  XmlParser_gcc54_v3_3_LUCID
  log4cpp_gcc54_v3_3_LUCID
  vpi
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_cudaimgproc
  opencv_cudawarping
)

# Install the executable
install(TARGETS
  lucid_driver
  DESTINATION lib/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
        scripts/waterlinked_driver
        scripts/thruster_controller
        scripts/depth_actuators_i2c
        scripts/can_driver
        DESTINATION lib/${PROJECT_NAME}
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
