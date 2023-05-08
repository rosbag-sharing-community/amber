cmake_minimum_required(VERSION 3.22)

project(rosbag2_pytorch_data_loader)
find_package(PythonExtensions REQUIRED)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wnon-virtual-dtor
                      -Woverloaded-virtual)
endif()

find_package(PythonExtensions REQUIRED)
include(FindProtobuf REQUIRED)

set(PROTO_FILES "rosbag2_pytorch_data_loader/proto/annotation.proto")
protobuf_generate_python(PROTO_PY ${PROTO_FILES})

set(MESSAGES_DIR rosbag2_pytorch_data_loader/tf2/messages)
set(ROSIDL_RUNTIME_CPP_DIR rosbag2_pytorch_data_loader/tf2/rosidl_runtime_cpp)
set(CONSOLE_BRIDGE_DIR rosbag2_pytorch_data_loader/tf2/console_bridge)
set(TF2_DIR rosbag2_pytorch_data_loader/tf2/geometry2/tf2)
set(TF2_PY_DIR rosbag2_pytorch_data_loader/tf2/geometry2/tf2_py)

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

# Generate python files from .proto
add_custom_target(
  generate_proto ALL
  DEPENDS ${PROTO_PY}
  COMMENT ${PROTO_PY}" was generated")
file(COPY ${PROTO_PY}
     DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/rosbag2_pytorch_data_loader/proto)

# test settings
find_package(GTest)

if(GTest_FOUND)
  # tf2_unittest: add testcase for tf2
  function(tf2_unittest test_name)
    add_executable(${test_name} ${TF2_DIR}/test/${test_name}.cpp)
    target_include_directories(
      ${test_name} PUBLIC ${TF2_DIR}/include ${ROSIDL_RUNTIME_CPP_DIR}/include
                          ${CONSOLE_BRIDGE_DIR}/include ${MESSAGES_DIR})
    target_link_libraries(${test_name} GTest::GTest GTest::Main tf2)
  endfunction()

  tf2_unittest(cache_unittest)
  tf2_unittest(simple_tf2_core)
  tf2_unittest(static_cache_test)
  tf2_unittest(test_time)
endif()
