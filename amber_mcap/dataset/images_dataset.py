from amber_mcap.dataset.conversion import decode_image_message, decode_message
from amber_mcap.dataset.rosbag2_dataset import Rosbag2Dataset, MessageMetaData
from amber_mcap.dataset.topic_config import ImageTopicConfig, TfTopicConfig
from amber_mcap.exception import TaskDescriptionError, RuntimeError
from amber_mcap.unit.time import Time, TimeUnit
from amber_mcap.util.geometry import build_tf_buffer
from amber_mcap.tf2_amber import (
    BufferCore,
    timeFromSec,
    displayTimePoint,
    durationFromSec,
    TransformStamped,
)
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass, field
from mcap.reader import NonSeekingReader
from typing import Any, List, Optional, Tuple
import numpy as np
import datetime
import quaternion
import torch


@dataclass
class ReadImagesConfig(YAMLWizard):  # type: ignore
    image_topics: List[ImageTopicConfig] = field(default_factory=list)
    compressed: bool = True
    tf_topic: TfTopicConfig = TfTopicConfig()

    def get_image_topics(self) -> List[str]:
        topics: List[str] = []
        for topic in self.image_topics:
            topics.append(topic.topic_name)
        return topics

    def get_camera_info_topics(self) -> List[str]:
        topics: List[str] = []
        for topic in self.image_topics:
            if topic.camera_info_topic_name != None:
                topics.append(topic.camera_info_topic_name)
        return topics


class ImagesDataset(Rosbag2Dataset):  # type: ignore
    # images: List[torch.Tensor] = []
    num_images = 0
    config: ReadImagesConfig = ReadImagesConfig()
    tf_buffer: Optional[BufferCore] = None

    def __init__(
        self,
        rosbag_path: str,
        config: ReadImagesConfig,
        transform: Any = None,
        target_transform: Any = None,
    ) -> None:
        # self.images.clear()
        self.config = config
        print(self.config)
        super().__init__(
            rosbag_path,
            self.config.compressed,
            transform,
            target_transform,
        )
        self.tf_buffer, _, _ = build_tf_buffer(
            self.rosbag_files, self.config.tf_topic, self.config.compressed
        )
        self.count_images()

    def count_images(self) -> None:
        self.num_images = 0
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.get_image_topics():
                    self.num_images = self.num_images + 1
                    self.message_metadata.append(
                        MessageMetaData.from_dict(
                            {
                                "publish_time": datetime.datetime.fromtimestamp(
                                    Time(message.publish_time, TimeUnit.NANOSECOND).get(
                                        TimeUnit.SECOND
                                    ),
                                    tz=datetime.timezone.utc,
                                ),
                                "topic": channel.topic,
                                "rosbag_path": rosbag_file,
                            }
                        )
                    )

    def __len__(self) -> int:
        return self.num_images

    def __iter__(self) -> torch.Tensor:
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.get_image_topics():
                    yield decode_image_message(message, schema, self.config.compressed)

    def get_camera_info(self):
        if (
            len(self.config.get_image_topics()) != 1
            or len(self.config.get_camera_info_topics()) != 1
        ):
            raise TaskDescriptionError(
                "Multiple camera with camera_info does not supported."
            )
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.get_camera_info_topics():
                    return decode_message(message, schema, self.config.compressed)

    def get_camera_image_by_index(self, index: int):
        if index >= self.num_images or index < 0:
            raise RuntimeError("Index of the image was invalid. Index : " + str(index))
        current_index = 0
        for rosbag_file in self.rosbag_files:
            reader = NonSeekingReader(rosbag_file)
            for schema, channel, message in reader.iter_messages():
                if channel.topic in self.config.get_image_topics():
                    if current_index == index:
                        return decode_image_message(
                            message, schema, self.config.compressed
                        )
                    current_index = current_index + 1
        raise RuntimeError(
            "Failed to get camera image by index. Index "
            + str(index)
            + " was not found."
        )

    def transform_3d_point_to_image_coordinate(
        self,
        index: int,
        point_3d: Tuple[float, float, float],
        map_frame_id: str = "map",
    ) -> Tuple[float, float]:
        camera_info = self.get_camera_info()
        transform = self.tf_buffer.lookupTransform(
            map_frame_id,
            camera_info.header.frame_id,
            timeFromSec(self.get_metadata(index).publish_time.timestamp()),
        )
        rotation_quaternion = np.quaternion(
            transform.transform.rotation.w,
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
        ).conjugate()
        translation_vector = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=float,
        ).reshape(3)
        rotated_point = -quaternion.rotate_vectors(
            rotation_quaternion, np.array(point_3d, dtype=float).reshape(3)
        )
        rotated_translation = quaternion.rotate_vectors(
            rotation_quaternion, translation_vector
        )
        transformed_point = rotated_point + rotated_translation
        # # See also https://github.com/ros-perception/vision_opencv/blob/27de9ecf9862e6fba509b7e49e3c2511c7d11627/image_geometry/src/pinhole_camera_model.cpp#L299-L308
        fx = np.array(camera_info.p, dtype=float).reshape(3, 4)[0][0]
        fy = np.array(camera_info.p, dtype=float).reshape(3, 4)[1][1]
        cx = np.array(camera_info.p, dtype=float).reshape(3, 4)[0][2]
        cy = np.array(camera_info.p, dtype=float).reshape(3, 4)[1][2]
        tx = np.array(camera_info.p, dtype=float).reshape(3, 4)[0][3]
        ty = np.array(camera_info.p, dtype=float).reshape(3, 4)[1][3]
        ux = (fx * -transformed_point[1] + tx) / transformed_point[0] + cx
        uy = (fy * transformed_point[2] + ty) / transformed_point[0] + cy
        return (ux, uy)


if __name__ == "__main__":
    pass
