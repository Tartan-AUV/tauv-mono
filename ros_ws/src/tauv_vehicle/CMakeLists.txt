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
find_package(Boost REQUIRED COMPONENTS system thread)

set(CMAKE_CXX_STANDARD 17)

# Build flatbuffers
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fb/build)
execute_process(COMMAND
        ${CMAKE_COMMAND}
        -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_CURRENT_BINARY_DIR}/fb/include
        -DCMAKE_INSTALL_BINDIR=${CMAKE_CURRENT_BINARY_DIR}/fb/bin
        -DCMAKE_INSTALL_LIBDIR=${CMAKE_CURRENT_BINARY_DIR}/fb/lib
        -DFLATBUFFERS_BUILD_FLATLIB=NO
        -DFLATBUFFERS_BUILD_TESTS=NO
        -DFLATBUFFERS_USE_STD_SPAN=TRUE
        ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fb/build)
execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/fb/build --target install)

# Set up schema compilation
set(SCHEMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/schemas)
set(GENERATED_INCLUDES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/generated)

# Create directory if it doesn't exist
file(MAKE_DIRECTORY ${GENERATED_INCLUDES_DIR})


# List all schema files
set(SCHEMA_FILES
        ${SCHEMA_DIR}/util.fbs
        ${SCHEMA_DIR}/eth_msg_rtvc_jetson_100.fbs
        ${SCHEMA_DIR}/eth_msg_rtvc_jetson_50.fbs
        ${SCHEMA_DIR}/eth_msg_jetson_rtvc_50.fbs
)
# List expected output headers for dependency tracking
set(GENERATED_HEADERS "")
foreach (SCHEMA ${SCHEMA_FILES})
    get_filename_component(SCHEMA_NAME ${SCHEMA} NAME_WE)
    list(APPEND GENERATED_HEADERS "${GENERATED_INCLUDES_DIR}/${SCHEMA_NAME}_generated.h")
endforeach ()
#file(MAKE_DIRECTORY ${GENERATED_INCLUDES_DIR})
message(${SCHEMA_FILES})
add_custom_command(
        OUTPUT ${GENERATED_HEADERS}
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/fb/bin/flatc
        --cpp
        --cpp-std c++17
        --gen-all
        --gen-object-api
        -I ${SCHEMA_DIR}
        -o ${GENERATED_INCLUDES_DIR}
        ${SCHEMA_FILES}
        COMMENT "Generating FlatBuffers headers for all schemas"
        VERBATIM
)

add_custom_target(generate_flatbuffers_headers ALL DEPENDS ${GENERATED_HEADERS})

# Executables
add_executable(rtvc src/rtvc.cpp ${GENERATED_HEADERS})

add_dependencies(rtvc generate_flatbuffers_headers)

target_include_directories(rtvc PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
        ${CMAKE_CURRENT_BINARY_DIR}/fb/include
        ${Boost_INCLUDE_DIRS})

target_compile_features(rtvc PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(
        rtvc
        "rclcpp"
        "rclpy"
        "std_msgs"
        "tauv_msgs"
        "sensor_msgs"
)

target_link_libraries(rtvc ${Boost_LIBRARIES})

# Install executables
install(TARGETS rtvc
        DESTINATION lib/${PROJECT_NAME})

# Install generated headers
install(
        DIRECTORY ${GENERATED_INCLUDES_DIR}/
        DESTINATION include/${PROJECT_NAME}/generated
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
        scripts/waterlinked_driver
        scripts/lucid_driver
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
