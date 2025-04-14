cmake_minimum_required(VERSION 3.8)
project(tauv_vehicle)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# Add FlatBuffers
find_path(FLATBUFFERS_INCLUDE_DIR NAMES flatbuffers/flatbuffers.h)
find_library(FLATBUFFERS_LIBRARY NAMES flatbuffers)
find_program(FLATBUFFERS_FLATC_EXECUTABLE NAMES flatc)

if(NOT FLATBUFFERS_INCLUDE_DIR OR NOT FLATBUFFERS_LIBRARY)
  message(FATAL_ERROR "FlatBuffers not found")
endif()

if(NOT FLATBUFFERS_FLATC_EXECUTABLE)
  message(FATAL_ERROR "FlatBuffers compiler (flatc) not found")
endif()

# Set up schema compilation
set(SCHEMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/schemas)
set(GENERATED_INCLUDES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/generated)

# Create directory if it doesn't exist
file(MAKE_DIRECTORY ${GENERATED_INCLUDES_DIR})

# List all schema files
set(SCHEMA_FILES
  ${SCHEMA_DIR}/rtvc.fbs
  # Add more schema files as needed
)

# Generate custom command to compile schemas
set(GENERATED_HEADERS "")
foreach(SCHEMA ${SCHEMA_FILES})
  get_filename_component(SCHEMA_NAME ${SCHEMA} NAME_WE)
  set(GENERATED_HEADER "${GENERATED_INCLUDES_DIR}/${SCHEMA_NAME}_generated.h")
  list(APPEND GENERATED_HEADERS ${GENERATED_HEADER})
  
  add_custom_command(
    OUTPUT ${GENERATED_HEADER}
    COMMAND ${FLATBUFFERS_FLATC_EXECUTABLE} 
            --cpp 
            -o ${GENERATED_INCLUDES_DIR}
            --scoped-enums
            ${SCHEMA}
    COMMENT "Generating FlatBuffers header for ${SCHEMA}"
    VERBATIM
  )
endforeach()

# Create a custom target to ensure header generation happens
add_custom_target(generate_flatbuffers DEPENDS ${GENERATED_HEADERS})

add_executable(rtvc src/rtvc.cpp ${GENERATED_HEADERS})
add_dependencies(rtvc generate_flatbuffers)

target_include_directories(rtvc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${FLATBUFFERS_INCLUDE_DIR})

target_compile_features(rtvc PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(rtvc ${FLATBUFFERS_LIBRARY})
ament_target_dependencies(
  rtvc
  "rclcpp"
  "rclpy"
  "std_msgs"
)

# Install executables
install(TARGETS rtvc
  DESTINATION lib/${PROJECT_NAME})

# Install generated headers
install(
  DIRECTORY ${GENERATED_INCLUDES_DIR}/
  DESTINATION include/${PROJECT_NAME}/generated
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