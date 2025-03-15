from amber.importer.video import VideoImporter, VideoImporterConfig
from amber.importer.tf import TfImporter, TfImporterConfig
from pathlib import Path
import os


def test_video_importer() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    importer = VideoImporter(
        str(current_path / "video" / "soccer_goal.mp4"),
        VideoImporterConfig.from_yaml_file(
            str(current_path / "video" / "video_importer.yaml")
        ),
    )
    importer.write()


def test_tf_importer() -> None:
    pass
