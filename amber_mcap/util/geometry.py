import numpy as np
import cv2
import sys
from mcap.reader import NonSeekingReader
import amber_mcap.tf2_amber
from amber_mcap.dataset.topic_config import TfTopicConfig
from amber_mcap.dataset.conversion import build_transform_stamped_message
from amber_mcap.exception import TaskDescriptionError, RuntimeError
from typing import List, Tuple, Optional, Dict, cast
from pathlib import Path
import xml.etree.ElementTree as ET
import quaternion


def build_tf_buffer(
    rosbag_files: List[str],
    topic_config: TfTopicConfig = TfTopicConfig(),
    compressed: bool = False,
) -> Tuple[amber_mcap.tf2_amber.BufferCore, Optional[float], Optional[float]]:
    class Origin:
        rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    class FixedJoint:
        origin: Origin = Origin()
        parent: str
        child: str

        def to_tf_message(
            self,
            timestamp: int,
        ) -> amber_mcap.tf2_amber.TransformStamped:
            quat = quaternion.from_euler_angles(
                np.array(
                    [
                        float(self.origin.rpy[0]),
                        float(self.origin.rpy[1]),
                        float(self.origin.rpy[2]),
                    ]
                )
            )
            tf_amber_message = amber_mcap.tf2_amber.TransformStamped(
                amber_mcap.tf2_amber.Header(
                    amber_mcap.tf2_amber.Time(
                        timestamp // 10**9, timestamp % 10**9
                    ),
                    self.parent,
                ),
                self.child,
                amber_mcap.tf2_amber.Transform(
                    amber_mcap.tf2_amber.Vector3(
                        float(self.origin.xyz[0]),
                        float(self.origin.xyz[1]),
                        float(self.origin.xyz[2]),
                    ),
                    amber_mcap.tf2_amber.Quaternion(quat.x, quat.y, quat.z, quat.real),
                ),
            )
            return tf_amber_message

    def load_static_tf_from_urdf_string(
        tf_buffer: amber_mcap.tf2_amber.BufferCore, urdf: str, timestamp: int
    ) -> None:
        def get_fixed_joint(xml: ET.Element) -> FixedJoint:
            joint = FixedJoint()
            for child in xml:
                if child.tag == "origin":
                    if "rpy" in child.attrib:
                        joint.origin.rpy = cast(
                            Tuple[float, float, float],
                            tuple(map(float, child.attrib["rpy"].split(" "))),
                        )
                    if "xyz" in child.attrib:
                        joint.origin.xyz = cast(
                            Tuple[float, float, float],
                            tuple(map(float, child.attrib["xyz"].split(" "))),
                        )
                if child.tag == "parent":
                    joint.parent = child.attrib["link"]
                if child.tag == "child":
                    joint.child = child.attrib["link"]
            return joint

        joints = ET.fromstring(urdf).findall(".//joint")
        for joint in joints:
            if joint.attrib["type"] == "fixed":
                tf_buffer.setTransform(
                    get_fixed_joint(joint).to_tf_message(timestamp),
                    "Authority undetectable",
                    True,
                )

    if topic_config.urdf_path and topic_config.robot_description_topic:
        raise TaskDescriptionError(
            "Do not specidy urdf and robot_description topic at the same time."
        )
    first_timestamp: Optional[int] = None
    last_timestamp: Optional[int] = None
    urdf_string: Optional[str] = None
    tf_buffer = amber_mcap.tf2_amber.BufferCore(
        amber_mcap.tf2_amber.durationFromSec(sys.float_info.max)
    )

    for rosbag_file in rosbag_files:
        reader = NonSeekingReader(rosbag_file)
        for schema, channel, message in reader.iter_messages():
            timestamp = message.log_time

            if first_timestamp is None or timestamp < first_timestamp:
                first_timestamp = timestamp
            if last_timestamp is None or timestamp > last_timestamp:
                last_timestamp = timestamp

            if channel.topic == topic_config.topic_name:
                for tf_amber_message in build_transform_stamped_message(
                    message, schema, compressed
                ):
                    tf_buffer.setTransform(
                        tf_amber_message, "Authority undetectable", False
                    )
            if channel.topic == topic_config.static_tf_topic_name:
                for tf_amber_message in build_transform_stamped_message(
                    message, schema, compressed
                ):
                    tf_buffer.setTransform(
                        tf_amber_message, "Authority undetectable", True
                    )

            if channel.topic == topic_config.robot_description_topic:
                urdf_string = message.data

    if topic_config.urdf_path:
        with open(topic_config.urdf_path, "r", encoding="utf-8") as f:
            urdf_string = f.read()

    if urdf_string:
        if first_timestamp:
            load_static_tf_from_urdf_string(tf_buffer, urdf_string, first_timestamp)
        else:
            raise RuntimeError(
                "Failed to get first timestmp. Please check tf data in rosbag data is exist or not."
            )

    return (tf_buffer, first_timestamp, last_timestamp)
