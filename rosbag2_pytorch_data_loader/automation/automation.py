from typing import Any
from abc import ABC, abstractmethod
from rosbag2_pytorch_data_loader.dataset.rosbag2_pytorch_dataset import Rosbag2Dataset
from mcap.writer import CompressionType, Writer
from typing import List
import json
from mcap.reader import NonSeekingReader
from mcap_ros2.writer import Writer


class Automation(ABC):
    @abstractmethod
    def __init__(self, yaml_path: str) -> None:
        pass

    @abstractmethod
    def inference(self, dataset: Rosbag2Dataset) -> Any:
        pass

    def write(
        self,
        dataset: Rosbag2Dataset,
        topic: str,
        annotation_data: List[Any],
        output_rosbag_path: str,
    ) -> None:
        rosbag_file = open(output_rosbag_path, "w+b")
        writer = Writer(rosbag_file)
        reader = NonSeekingReader(dataset.rosbag_path)
        topics = []
        # Copy all topics
        for schema, channel, message in reader.iter_messages():
            if channel.topic not in topics:
                writer.register_msgdef(schema.name, schema.data.decode("utf-8"))
                topics.append(channel.topic)
            writer.write_message(
                topic=channel.topic,
                schema=schema,
                message=message,
                log_time=message.log_time,
                publish_time=message.publish_time,
                sequence=message.sequence,
            )
        # Append annotation data
        annotation_schema = writer.register_msgdef(
            "std_msgs/msg/String", "string annotation_json"
        )
        writer.write_message(
            topic=topic,
            schema=annotation_schema,
            message={"annotations": annotation_data},
            sequence=0,
        )
        writer.finish()
        rosbag_file.close()
