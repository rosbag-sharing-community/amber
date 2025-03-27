// Copyright (c) 2023 OUXT Polaris
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GEOMETRY_MSGS__MSG__TRANSFORM_STAMPED_HPP_
#define GEOMETRY_MSGS__MSG__TRANSFORM_STAMPED_HPP_

#include <geometry_msgs/msg/transform.hpp>
#include <std_msgs/msg/header.hpp>

namespace geometry_msgs {
namespace msg {
struct TransformStamped {
  std_msgs::msg::Header header;
  std::string child_frame_id;
  geometry_msgs::msg::Transform transform;

  TransformStamped() : header(), child_frame_id(""), transform() {}
  TransformStamped(const std_msgs::msg::Header &header,
                   const std::string &child_frame_id,
                   const geometry_msgs::msg::Transform &transform)
      : header(header), child_frame_id(child_frame_id), transform(transform) {}
};
} // namespace msg
} // namespace geometry_msgs

#endif // GEOMETRY_MSGS__MSG__TRANSFORM_STAMPED_HPP_
