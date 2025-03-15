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

#ifndef BUILTIN_INTERFACES__MSG__TIME_HPP_
#define BUILTIN_INTERFACES__MSG__TIME_HPP_

#include <cstdint>

namespace builtin_interfaces {
namespace msg {
struct Time {
  std::int32_t sec;
  std::uint32_t nanosec;

  Time(std::int32_t sec, std::uint32_t nanosec) : sec(sec), nanosec(nanosec) {}
  Time() : sec(0), nanosec(0) {}
};
} // namespace msg
} // namespace builtin_interfaces

#endif // BUILTIN_INTERFACES__MSG__TIME_HPP_
