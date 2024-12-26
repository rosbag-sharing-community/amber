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

#include <pybind11/pybind11.h>
#include <tf2/buffer_core.h>

PYBIND11_MODULE(tf2_amber, m) {
  m.doc() = "python package compatible with tf2";
  namespace py = pybind11;

  // Functions/Classes related to time.
  py::class_<tf2::Duration>(m, "Duration").def("count", &tf2::Duration::count);
  py::class_<tf2::TimePoint>(m, "TimePoint").def(py::init<tf2::Duration>());
  m.def("get_now", &tf2::get_now, "Get tf2::TimePoint of the now.");
  m.def("durationFromSec", &tf2::durationFromSec,
        "Construct tf2::Duration from second");
  m.def("timeFromSec", &tf2::timeFromSec,
        "Construct tf2::TimePoint from second");
  m.def("durationToSec", &tf2::durationToSec,
        "convert tf2::Duration to double value");
  m.def("timeToSec", &tf2::durationToSec,
        "convert tf2::TimePoint to double value");
  m.def("displayTimePoint", &tf2::displayTimePoint,
        "convert tf2::TimePoint to string");

  // ROS 2 compatible messages.
  py::class_<builtin_interfaces::msg::Time>(m, "Time")
      .def(py::init([](int sec, unsigned int nanosec) {
        return builtin_interfaces::msg::Time(sec, nanosec);
      }))
      .def(py::init([]() { return builtin_interfaces::msg::Time(); }))
      .def_readwrite("sec", &builtin_interfaces::msg::Time::sec)
      .def_readwrite("nanosec", &builtin_interfaces::msg::Time::nanosec);

  py::class_<std_msgs::msg::Header>(m, "Header")
      .def(py::init([](const builtin_interfaces::msg::Time &stamp,
                       const std::string &frame_id) {
        return std_msgs::msg::Header(stamp, frame_id);
      }))
      .def(py::init([]() { return std_msgs::msg::Header(); }))
      .def_readwrite("stamp", &std_msgs::msg::Header::stamp)
      .def_readwrite("frame_id", &std_msgs::msg::Header::frame_id);

  py::class_<geometry_msgs::msg::Vector3>(m, "Vector3")
      .def(py::init([](double x, double y, double z) {
        return geometry_msgs::msg::Vector3(x, y, z);
      }))
      .def(py::init([]() { return geometry_msgs::msg::Vector3(); }))
      .def_readwrite("x", &geometry_msgs::msg::Vector3::x)
      .def_readwrite("y", &geometry_msgs::msg::Vector3::y)
      .def_readwrite("z", &geometry_msgs::msg::Vector3::z);

  py::class_<geometry_msgs::msg::Quaternion>(m, "Quaternion")
      .def(py::init([](double x, double y, double z, double w) {
        return geometry_msgs::msg::Quaternion(x, y, z, w);
      }))
      .def(py::init([]() { return geometry_msgs::msg::Quaternion(); }))
      .def_readwrite("x", &geometry_msgs::msg::Quaternion::x)
      .def_readwrite("y", &geometry_msgs::msg::Quaternion::y)
      .def_readwrite("z", &geometry_msgs::msg::Quaternion::z)
      .def_readwrite("w", &geometry_msgs::msg::Quaternion::w);

  py::class_<geometry_msgs::msg::Transform>(m, "Transform")
      .def(py::init([](const geometry_msgs::msg::Vector3 &translation,
                       const geometry_msgs::msg::Quaternion &rotation) {
        return geometry_msgs::msg::Transform(translation, rotation);
      }))
      .def(py::init([]() { return geometry_msgs::msg::Transform(); }))
      .def_readwrite("translation", &geometry_msgs::msg::Transform::translation)
      .def_readwrite("rotation", &geometry_msgs::msg::Transform::rotation);

  py::class_<geometry_msgs::msg::TransformStamped>(m, "TransformStamped")
      .def(py::init([](const std_msgs::msg::Header &header,
                       const std::string &child_frame_id,
                       const geometry_msgs::msg::Transform &transform) {
        return geometry_msgs::msg::TransformStamped(header, child_frame_id,
                                                    transform);
      }))
      .def(py::init([]() { return geometry_msgs::msg::TransformStamped(); }))
      .def_readwrite("header", &geometry_msgs::msg::TransformStamped::header)
      .def_readwrite("child_frame_id",
                     &geometry_msgs::msg::TransformStamped::child_frame_id)
      .def_readwrite("transform",
                     &geometry_msgs::msg::TransformStamped::transform);

  // Functions/Classes related to BufferCore
  py::class_<tf2::BufferCore>(m, "BufferCore")
      .def(py::init<tf2::Duration>())
      .def("setTransform", &tf2::BufferCore::setTransform)
      .def("lookupTransform",
           [](const tf2::BufferCore &self, const std::string &target_frame,
              const std::string &source_frame, const tf2::TimePoint &time) {
             return self.lookupTransform(target_frame, source_frame, time);
           });
}
