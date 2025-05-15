cmake_minimum_required(VERSION 3.22)

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME rtvc)

# Include toolchain file
include("cmake/gcc-arm-none-eabi.cmake")

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME} C CXX ASM)
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for C, C++, and ASM
enable_language(C CXX ASM)

# Compile schemas using system compiler
find_program(FLATBUFFERS_FLATC_EXECUTABLE NAMES flatc)
if(NOT FLATBUFFERS_FLATC_EXECUTABLE)
    message(FATAL_ERROR "FlatBuffers compiler (flatc) not found")
endif()
set(SCHEMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../packages/tauv_vehicle/schemas)
set(GENERATED_INCLUDES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/FlatBuffers/)
file(MAKE_DIRECTORY ${GENERATED_INCLUDES_DIR})

set(SCHEMA_FILES
        ${SCHEMA_DIR}/eth_msg_rtvc_jetson.fbs
        ${SCHEMA_DIR}/eth_msg_jetson_rtvc.fbs
)

set(GENERATED_HEADERS ""
)
foreach(SCHEMA ${SCHEMA_FILES})
    get_filename_component(SCHEMA_NAME ${SCHEMA} NAME_WE)
    set(GENERATED_HEADER "${GENERATED_INCLUDES_DIR}/${SCHEMA_NAME}_generated.h")
    list(APPEND GENERATED_HEADERS ${GENERATED_HEADER})

    add_custom_command(
            OUTPUT ${GENERATED_HEADER}
            BYPRODUCTS ${GENERATED_HEADER}
            DEPENDS ${SCHEMA}
            COMMAND ${FLATBUFFERS_FLATC_EXECUTABLE}
            --cpp
            --cpp-std c++17
            --gen-all
            -o ${GENERATED_INCLUDES_DIR}
            --scoped-enums
            ${SCHEMA}
            COMMENT "Generating FlatBuffers header for ${SCHEMA}"
            VERBATIM
    )
endforeach()

message(${GENERATED_HEADERS})

# Create a custom target to ensure header generation happens
add_custom_target(generate_flatbuffers DEPENDS ${GENERATED_HEADERS})

# Create the executable target
add_executable(${CMAKE_PROJECT_NAME}
        # C source files
        Modules/Eth50HzModule.cpp
        Modules/Eth50HzModule.hpp
        Util/Singleton.hpp
        Util/StaticQueue.hpp
        Tasks/IntervalTask.hpp
        Tasks/Task50Hz.hpp
        Tasks/Task50Hz.cpp
        Util/UdpSocket.hpp
        Tasks/Task.hpp
        Modules/ESCModule.cpp
        Modules/ESCModule.hpp
        Drivers/VESC/buffer.cpp
        Drivers/VESC/crc.cpp
        Drivers/VESC/VESC_UART.cpp
        ${GENERATED_HEADERS}
)

add_dependencies(${CMAKE_PROJECT_NAME} generate_flatbuffers)

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
        # Add user defined library search paths
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        Modules
        Util
        Tasks
        Drivers/VESC
        ${GENERATED_INCLUDES_DIR}
        Middlewares/Third_Party/flatbuffers/include
)

# Add preprocessor definitions
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        FLATBUFFERS_GENERAL_HEAP_ALLOC_OK=0
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti>
)

# Link libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
        stm32cubemx
)
