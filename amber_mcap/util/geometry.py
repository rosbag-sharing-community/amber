import numpy as np
import cv2
import sys
from mcap.reader import NonSeekingReader
from amber_mcap.tf2_amber import (
    BufferCore,
    timeFromSec,
    durationFromSec,
    TransformStamped,
)
from amber_mcap.dataset.topic_config import TfTopicConfig
from amber_mcap.dataset.conversion import build_transform_stamped_message
from amber_mcap.exception import TaskDescriptionError
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import xml.etree.ElementTree as ET


def build_tf_buffer(
    rosbag_files: List[str],
    topic_config: TfTopicConfig = TfTopicConfig(),
    compressed=False,
) -> Tuple[BufferCore, Optional[float], Optional[float]]:
    def load_static_tf_from_urdf_string(urdf: str):
        class Origin:
            rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        class FixedJoint:
            origin: Origin = Origin()
            parent: str
            child: str

        def get_fixed_joint(xml: ET) -> FixedJoint:
            joint = FixedJoint()
            for child in xml:
                if child.tag == "origin":
                    if "rpy" in child.attrib:
                        joint.origin.rpy = tuple(child.attrib["rpy"].split(" "))
                    if "xyz" in child.attrib:
                        joint.origin.xyz = tuple(child.attrib["xyz"].split(" "))
                if child.tag == "parent":
                    joint.parent = child.attrib["link"]
                if child.tag == "child":
                    joint.child = child.attrib["link"]
            return joint


        joints = ET.fromstring(urdf).findall(".//joint")
        for joint in joints:
            if joint.attrib["type"] == "fixed":
                fixed_joint = get_fixed_joint(joint)
                # print(joint.attrib)
        raise RuntimeError("Hoge")

    if topic_config.urdf_path and topic_config.robot_description_topic:
        raise TaskDescriptionError(
            "Do not specidy urdf and robot_description topic at the same time."
        )
    first_timestamp = None
    last_timestamp = None
    tf_buffer = BufferCore(durationFromSec(sys.float_info.max))

    if topic_config.urdf_path:
        with open(topic_config.urdf_path, "r", encoding="utf-8") as f:
            load_static_tf_from_urdf_string(f.read())

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
                    message, schema, ompressed
                ):
                    tf_buffer.setTransform(
                        tf_amber_message, "Authority undetectable", True
                    )

            if channel.topic == topic_config.robot_description_topic:
                load_static_tf_from_urdf_string(message.data)
    return (tf_buffer, first_timestamp, last_timestamp)


def project_3d_points_to_image(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
):
    """
    Projects 3D points to the image coordinate system.

    Args:
        points_3d (numpy.ndarray): 3D coordinates of the points (N x 3)
        camera_matrix (numpy.ndarray): Camera intrinsic matrix (3 x 3)
        dist_coeffs (numpy.ndarray): Lens distortion coefficients (1 x 5 or 1 x 8)
        rvec (numpy.ndarray): Rotation vector (3 x 1)
        tvec (numpy.ndarray): Translation vector (3 x 1)

    Returns:
        numpy.ndarray: Projected 2D coordinates of the points (N x 2)
    """

    image_points, _ = cv2.projectPoints(
        points_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    return image_points.reshape(-1, 2)
