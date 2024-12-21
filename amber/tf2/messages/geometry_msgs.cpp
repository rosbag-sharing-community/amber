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

#include <geometry_msgs/msg/transform_stamped.hpp>

#include <pybind11/pybind11.h>

PYBIND11_MODULE(geometry_msgs, m)
{
    m.doc() = "python package compatible with geometry_msgs";
    namespace py = pybind11;
    py::class_<geometry_msgs::msg::TransformStamped>(m, "TransformStamped");
}
