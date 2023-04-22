cmake_minimum_required(VERSION 3.22)

project(rosbag2_pytorch_data_loader)
find_package(PythonExtensions REQUIRED)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wnon-virtual-dtor -Woverloaded-virtual)
endif()

find_package(PythonExtensions REQUIRED)

set(messages_dir rosbag2_pytorch_data_loader/messages)
set(rosidl_runtime_cpp_dir rosbag2_pytorch_data_loader/rosidl_runtime_cpp)
set(console_bridge_dir rosbag2_pytorch_data_loader/console_bridge)
set(tf2_dir rosbag2_pytorch_data_loader/geometry2/tf2)
set(tf2_py_dir rosbag2_pytorch_data_loader/geometry2/tf2_py)

add_library(tf2 
  ${tf2_dir}/src/time.cpp
  ${tf2_dir}/src/cache.cpp
  ${tf2_dir}/src/static_cache.cpp
  ${tf2_dir}/src/buffer_core.cpp
  ${console_bridge_dir}/src/console.cpp
)

target_include_directories(tf2
  PUBLIC 
    ${tf2_dir}/include
    ${rosidl_runtime_cpp_dir}/include
    ${console_bridge_dir}/include
    ${messages_dir}
)

add_library(tf2_py
  ${tf2_py_dir}/src/tf2_py.cpp
)
target_include_directories(tf2_py
  PUBLIC 
    ${tf2_dir}/include
    ${rosidl_runtime_cpp_dir}/include
    ${console_bridge_dir}/include
    ${messages_dir}
    ${PYTHON_INCLUDE_DIRS}
)

install(TARGETS tf2 LIBRARY DESTINATION tf2)
install(TARGETS tf2_py LIBRARY DESTINATION tf2_py)

# test settings
find_package(GTest REQUIRED)

function(tf2_unittest test_name)
  add_executable(${test_name} ${tf2_dir}/test/${test_name}.cpp)
  target_include_directories(${test_name}
    PUBLIC
      ${tf2_dir}/include
      ${rosidl_runtime_cpp_dir}/include
      ${console_bridge_dir}/include
      ${messages_dir}
  )
  target_link_libraries(${test_name} GTest::GTest GTest::Main tf2)
endfunction()

tf2_unittest(cache_unittest)
tf2_unittest(simple_tf2_core)
tf2_unittest(static_cache_test)
tf2_unittest(test_time)
