cmake_minimum_required(VERSION 3.15)

project(flatccrt_arm VERSION 1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_AR arm-none-eabi-ar)
set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
set(CMAKE_SIZE arm-none-eabi-size)

set(MCU_FLAGS "-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard")
set(CMAKE_C_FLAGS "${MCU_FLAGS} -O2 -ffunction-sections -fdata-sections -Wall")
set(CMAKE_EXE_LINKER_FLAGS "${MCU_FLAGS} -Wl,--gc-sections")

add_library(flatccrt_arm STATIC
        ../../../flatcc/src/runtime/builder.c
        ../../../flatcc/src/runtime/emitter.c
        ../../../flatcc/src/runtime/refmap.c
        ../../../flatcc/src/runtime/verifier.c
        ../../../flatcc/src/runtime/json_parser.c
        ../../../flatcc/src/runtime/json_printer.c
)

target_include_directories(flatccrt_arm PUBLIC
        ../../../flatcc/include)

target_link_libraries(flatccrt_arm INTERFACE)

install(TARGETS flatccrt_arm
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
