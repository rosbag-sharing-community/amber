cmake_minimum_required(VERSION 3.22)

project(amber_mcap)
find_package(PythonExtensions REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wnon-virtual-dtor
                      -Woverloaded-virtual)
endif()

set(MESSAGES_DIR amber_mcap/tf2/messages)
set(ROSIDL_RUNTIME_CPP_DIR amber_mcap/tf2/rosidl_runtime_cpp)
set(CONSOLE_BRIDGE_DIR amber_mcap/tf2/console_bridge)
set(TF2_DIR amber_mcap/tf2/geometry2/tf2)
set(TF2_PY_DIR amber_mcap/tf2/geometry2/tf2_py)

find_package(pybind11 REQUIRED)
pybind11_add_module(
  tf2_amber
  ${TF2_DIR}/src/time.cpp 
  ${TF2_DIR}/src/cache.cpp
  ${TF2_DIR}/src/static_cache.cpp 
  ${TF2_DIR}/src/buffer_core.cpp
  ${TF2_DIR}/src/pybind.cpp
  ${CONSOLE_BRIDGE_DIR}/src/console.cpp)

target_include_directories(
  tf2_amber PUBLIC ${TF2_DIR}/include ${ROSIDL_RUNTIME_CPP_DIR}/include
             ${CONSOLE_BRIDGE_DIR}/include ${MESSAGES_DIR})

install(TARGETS tf2_amber LIBRARY DESTINATION .)
