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

#ifndef GEOMETRY_MSGS__MSG__TRANSFORM_HPP_
#define GEOMETRY_MSGS__MSG__TRANSFORM_HPP_

#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>

namespace geometry_msgs {
namespace msg {
struct Transform {
  geometry_msgs::msg::Vector3 translation;
  geometry_msgs::msg::Quaternion rotation;

  Transform() : translation(), rotation() {}
  Transform(const geometry_msgs::msg::Vector3 &translation,
            const geometry_msgs::msg::Quaternion &rotation)
      : translation(translation), rotation(rotation) {}
};
} // namespace msg
} // namespace geometry_msgs

#endif // GEOMETRY_MSGS__MSG__TRANSFORM_HPP_
