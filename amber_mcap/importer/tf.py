from amber_mcap.dataset.schema import TFMessageSchema
from amber_mcap.dataset.conversion import build_message_from_tf
from tf2_amber import TransformStamped
from mcap_ros2.writer import Writer as McapWriter
from dataclasses import dataclass
from dataclass_wizard import YAMLWizard
import math


@dataclass
class TfImporterConfig(YAMLWizard):  # type: ignore
    rosbag_path: str = "output.mcap"


class TfImporter:
    def __init__(self, config):
        self.config: TfImporterConfig = config
        self.file = open(self.config.rosbag_path, "wb")
        self.writer = McapWriter(self.file)
        self.schema = self.writer.register_msgdef(
            TFMessageSchema.name, TFMessageSchema.schema_text
        )
        self.index = 0

    # self.writer.finish throw error when we use it in destructor.
    def finish(self):
        self.writer.finish()
        self.file.close()

    def write(self, transform: TransformStamped):
        nanoseconds = int(
            transform.header.stamp.sec * math.pow(10, 9)
            + transform.header.stamp.nanosec
        )
        self.writer.write_message(
            topic="/tf",
            schema=self.schema,
            message=build_message_from_tf([transform]),
            log_time=nanoseconds,
            publish_time=nanoseconds,
            sequence=self.index,
        )
        self.index = self.index + 1


if __name__ == "__main__":
    pass
    # importer = TfImporter(TfImporterConfig())
    # importer.write(TransformStamped())
    # importer.write(TransformStamped())
    # importer.write(TransformStamped())
    # importer.finish()
