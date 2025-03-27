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

#ifndef GEOMETRY_MSGS__MSG__QUATERNION_HPP_
#define GEOMETRY_MSGS__MSG__QUATERNION_HPP_

namespace geometry_msgs {
namespace msg {
struct Quaternion {
  double x = 0;
  double y = 0;
  double z = 0;
  double w = 1;

  Quaternion() : x(0), y(0), z(0), w(1) {}
  Quaternion(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {}
};
} // namespace msg
} // namespace geometry_msgs

#endif // GEOMETRY_MSGS__MSG__QUATERNION_HPP_
