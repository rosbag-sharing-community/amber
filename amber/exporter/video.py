from dataclasses import dataclass
from dataclass_wizard import YAMLWizard

@dataclass
class VideoExportConfig(YAMLWizard):  # type: ignore
    topic_name: str = "/camera/image_raw"
    video_path: str = "output.mp4"


class VideoExporter:
    def __init__(self, mcap_path: str, config: VideoExportConfig) -> None:
        self.config: VideoImporterConfig = config
        self.capture = cv2.VideoCapture(video_path)
