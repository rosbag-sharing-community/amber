cmake_minimum_required(VERSION 3.22)

project(amber)
find_package(PythonExtensions REQUIRED)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wnon-virtual-dtor
                      -Woverloaded-virtual)
endif()

set(MESSAGES_DIR amber/tf2/messages)
set(ROSIDL_RUNTIME_CPP_DIR amber/tf2/rosidl_runtime_cpp)
set(CONSOLE_BRIDGE_DIR amber/tf2/console_bridge)
set(TF2_DIR amber/tf2/geometry2/tf2)
set(TF2_PY_DIR amber/tf2/geometry2/tf2_py)

add_library(
  tf2 STATIC
  ${TF2_DIR}/src/time.cpp ${TF2_DIR}/src/cache.cpp
  ${TF2_DIR}/src/static_cache.cpp ${TF2_DIR}/src/buffer_core.cpp
  ${CONSOLE_BRIDGE_DIR}/src/console.cpp)

target_include_directories(
  tf2 PUBLIC ${TF2_DIR}/include ${ROSIDL_RUNTIME_CPP_DIR}/include
             ${CONSOLE_BRIDGE_DIR}/include ${MESSAGES_DIR})

add_library(tf2_py STATIC
  ${TF2_PY_DIR}/src/tf2_py.cpp)
target_include_directories(
  tf2_py
  PUBLIC ${TF2_DIR}/include ${ROSIDL_RUNTIME_CPP_DIR}/include
         ${CONSOLE_BRIDGE_DIR}/include ${MESSAGES_DIR} ${PYTHON_INCLUDE_DIRS})

install(TARGETS tf2 LIBRARY DESTINATION tf2)
install(TARGETS tf2_py LIBRARY DESTINATION tf2_py)
python_extension_module(tf2)
python_extension_module(tf2_py)

find_package(pybind11 REQUIRED)
pybind11_add_module(geometry_msgs ${MESSAGES_DIR}/geometry_msgs.cpp)
target_include_directories(
  geometry_msgs
  PUBLIC ${MESSAGES_DIR})

# test settings
find_package(GTest)
