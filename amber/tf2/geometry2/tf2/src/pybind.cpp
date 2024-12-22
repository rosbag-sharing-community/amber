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

#include <tf2/buffer_core.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(tf2_amber, m)
{
    m.doc() = "python package compatible with tf2";
    namespace py = pybind11;

    // Functions/Classes related to time.
    py::class_<tf2::Duration>(m, "Duration")
        .def("count", &tf2::Duration::count);
    m.def("durationFromSec", &tf2::durationFromSec, "Construct tf2::Duration from second");
    
    // ROS 2 compatible messages
    py::class_<builtin_interfaces::msg::Time>(m, "Time")
        .def_readwrite("sec", &builtin_interfaces::msg::Time::sec)
        .def_readwrite("nanosec", &builtin_interfaces::msg::Time::nanosec);  

    py::class_<std_msgs::msg::Header>(m, "Header")
        .def_readwrite("seq", &std_msgs::msg::Header::seq)
        .def_readwrite("stamp", &std_msgs::msg::Header::stamp)
        .def_readwrite("frame_id", &std_msgs::msg::Header::frame_id);

    py::class_<geometry_msgs::msg::Vector3>(m, "Vector3")
        .def_readwrite("x", &geometry_msgs::msg::Vector3::x)
        .def_readwrite("y", &geometry_msgs::msg::Vector3::y)
        .def_readwrite("z", &geometry_msgs::msg::Vector3::z);

    py::class_<geometry_msgs::msg::Quaternion>(m, "Quaternion")
        .def_readwrite("x", &geometry_msgs::msg::Quaternion::x)
        .def_readwrite("y", &geometry_msgs::msg::Quaternion::y)
        .def_readwrite("z", &geometry_msgs::msg::Quaternion::z)
        .def_readwrite("z", &geometry_msgs::msg::Quaternion::w);

    py::class_<geometry_msgs::msg::Transform>(m, "Transform")
        .def_readwrite("translation", &geometry_msgs::msg::Transform::translation)
        .def_readwrite("rotation", &geometry_msgs::msg::Transform::rotation);

    py::class_<geometry_msgs::msg::TransformStamped>(m, "TransformStamped")
        .def_readwrite("header", &geometry_msgs::msg::TransformStamped::header)
        .def_readwrite("child_frame_id", &geometry_msgs::msg::TransformStamped::child_frame_id)
        .def_readwrite("transform", &geometry_msgs::msg::TransformStamped::transform);
}

