project(FallingTest)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    ${CMAKE_SOURCE_DIR}/../../Library/include
)

add_executable(FallingTest
    MySimulationManager.cpp
    FallingTest.cpp
)

target_link_libraries(FallingTest
    Stonefish
    GL X11 Xi Xrandr
    SDL2 SDL2_image freetype
)

install(TARGETS FallingTest DESTINATION bin)
