from amber.dataset.schema import TFMessageSchema
from tf2_amber import TransformStamped


@dataclass
class TfImporterConfig(YAMLWizard):  # type: ignore
    rosbag_path: str = "output.mcap"


class TfImporter:
    def __init__(self):
        self.config: TfImporterConfig = config
        self.file = open(self.config.rosbag_path, "wb")
        self.writer = McapWriter(self.file)
        self.schema = writer.register_msgdef(
            TFMessageSchema.name, TFMessageSchema.schema_text
        )

    def __del__(self):
        self.writer.finish()
        self.file.close()

    def write(self, transform: TransformStamped):
        get_nanoseconds_timestamp: Callable[[], int] = lambda: math.floor(
            Time(self.capture.get(cv2.CAP_PROP_POS_MSEC), TimeUnit.MILLISECOND).get(
                TimeUnit.NANOSECOND
            )
        )
        self.writer.write_message(
            topic="/tf",
            schema=self.schema,
            message="",
            log_time=get_nanoseconds_timestamp(),
            publish_time=get_nanoseconds_timestamp(),
            sequence=i,
        )
