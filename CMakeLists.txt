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
  tf2
  ${TF2_DIR}/src/time.cpp ${TF2_DIR}/src/cache.cpp
  ${TF2_DIR}/src/static_cache.cpp ${TF2_DIR}/src/buffer_core.cpp
  ${CONSOLE_BRIDGE_DIR}/src/console.cpp)

target_include_directories(
  tf2 PUBLIC ${TF2_DIR}/include ${ROSIDL_RUNTIME_CPP_DIR}/include
             ${CONSOLE_BRIDGE_DIR}/include ${MESSAGES_DIR})

add_library(tf2_py ${TF2_PY_DIR}/src/tf2_py.cpp)
target_include_directories(
  tf2_py
  PUBLIC ${TF2_DIR}/include ${ROSIDL_RUNTIME_CPP_DIR}/include
         ${CONSOLE_BRIDGE_DIR}/include ${MESSAGES_DIR} ${PYTHON_INCLUDE_DIRS})

install(TARGETS tf2 LIBRARY DESTINATION tf2)
install(TARGETS tf2_py LIBRARY DESTINATION tf2_py)
python_extension_module(tf2)
python_extension_module(tf2_py)

# test settings
find_package(GTest)

# if(GTest_FOUND)
#   # tf2_unittest: add testcase for tf2
#   function(tf2_unittest test_name)
#     add_executable(${test_name} ${TF2_DIR}/test/${test_name}.cpp)
#     target_include_directories(
#       ${test_name} PUBLIC ${TF2_DIR}/include ${ROSIDL_RUNTIME_CPP_DIR}/include
#                           ${CONSOLE_BRIDGE_DIR}/include ${MESSAGES_DIR})
#     target_link_libraries(${test_name} GTest::GTest GTest::Main tf2)
#   endfunction()

#   tf2_unittest(cache_unittest)
#   tf2_unittest(simple_tf2_core)
#   tf2_unittest(static_cache_test)
#   tf2_unittest(test_time)
# endif()
