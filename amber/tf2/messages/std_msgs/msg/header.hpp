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

#ifndef STD_MSGS__MSG__HEADER_HPP_
#define STD_MSGS__MSG__HEADER_HPP_

#include <builtin_interfaces/msg/time.hpp>
#include <cstdint>
#include <string>

namespace std_msgs {
namespace msg {
struct Header {
  builtin_interfaces::msg::Time stamp;
  std::string frame_id;

  Header() : stamp(), frame_id("") {}
  Header(const builtin_interfaces::msg::Time &stamp,
         const std::string &frame_id)
      : stamp(stamp), frame_id(frame_id) {}
};
} // namespace msg
} // namespace std_msgs

#endif // STD_MSGS__MSG__HEADER_HPP_
