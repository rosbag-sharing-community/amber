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

#ifndef GEOMETRY_MSGS__MSG__VECTOR3_HPP_
#define GEOMETRY_MSGS__MSG__VECTOR3_HPP_

#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>

namespace geometry_msgs {
namespace msg {
struct Vector3 {
  double x = 0;
  double y = 0;
  double z = 0;

  Vector3() : x(0), y(0), z(0) {}
  Vector3(double x, double y, double z) : x(x), y(y), z(z) {}
};
} // namespace msg
} // namespace geometry_msgs

#endif // GEOMETRY_MSGS__MSG__VECTOR3_HPP_
