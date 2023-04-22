cmake_minimum_required(VERSION 3.22)

project(rosbag2_pytorch_data_loader)
find_package(PythonExtensions REQUIRED)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wnon-virtual-dtor -Woverloaded-virtual)
endif()

set(messages_dir rosbag2_pytorch_data_loader/messages)
set(tf2_dir rosbag2_pytorch_data_loader/geometry2/tf2)
add_library(tf2 
  ${tf2_dir}/src/time.cpp
  ${tf2_dir}/src/cache.cpp
  ${tf2_dir}/src/static_cache.cpp
  ${tf2_dir}/src/buffer_core.cpp
)

target_include_directories(tf2
  PUBLIC ${tf2_dir}/include ${messages_dir}
)
install(TARGETS tf2 LIBRARY DESTINATION tf2)