cmake_minimum_required(VERSION 3.4)

# Add depthai-core dependency
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/depthai-core EXCLUDE_FROM_ALL)

# Create a project with name 'rgb_decoder'
set(TARGET_NAME rgb_decoder)
project(${TARGET_NAME})

# Dependencies
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(gstreamer REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.4)
pkg_search_module(gstreamer-sdp REQUIRED IMPORTED_TARGET gstreamer-sdp-1.0>=1.4)
pkg_search_module(gstreamer-app REQUIRED IMPORTED_TARGET gstreamer-app-1.0>=1.4)
pkg_search_module(gstreamer-video REQUIRED IMPORTED_TARGET gstreamer-video-1.0>=1.4)

include_directories(
        /usr/lib/x86_64-linux-gnu/glib-2.0/include
        /usr/include/glib-2.0
        /usr/local/include/gstreamer-1.0
        /usr/local/include/libsoup-2.4
        /usr/local/include/json-glib-1.0)

# Add source files
add_executable("${TARGET_NAME}"
    src/decoder.cpp
)

# Link with libraries
target_link_libraries(${TARGET_NAME}
    PUBLIC
        PkgConfig::gstreamer
        PkgConfig::gstreamer-sdp
        PkgConfig::gstreamer-app
        PkgConfig::gstreamer-video
        ${GSTLIBS_LIBRARIES}
        depthai::opencv
        ${OpenCV_LIBS} # optional, used for example
)


# Set some errors
if(NOT MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>)
endif()

# Set compiler features (c++17)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)

