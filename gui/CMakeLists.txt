cmake_minimum_required(VERSION 3.21)
project(ttslab_gui LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Multimedia Concurrent)

# The Zig backend: build it first with `zig build -Doptimize=ReleaseFast`
# in the project root, which produces zig-out/lib/libttslab.so.
get_filename_component(TTSLAB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(TTSLAB_LIB "${TTSLAB_ROOT}/zig-out/lib/libttslab.so")
if(NOT EXISTS "${TTSLAB_LIB}")
  message(FATAL_ERROR "libttslab.so not found: run `zig build -Doptimize=ReleaseFast` in ${TTSLAB_ROOT} first")
endif()

add_executable(ttslab-gui main.cpp)
target_include_directories(ttslab-gui PRIVATE "${TTSLAB_ROOT}/include")
target_compile_definitions(ttslab-gui PRIVATE TTSLAB_ROOT="${TTSLAB_ROOT}")
target_link_libraries(ttslab-gui PRIVATE Qt6::Widgets Qt6::Multimedia Qt6::Concurrent "${TTSLAB_LIB}")
set_target_properties(ttslab-gui PROPERTIES BUILD_RPATH "${TTSLAB_ROOT}/zig-out/lib")
