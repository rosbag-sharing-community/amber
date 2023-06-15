from amber.importer.video import VideoImporter, VideoImporterConfig
from pathlib import Path
import os


def test_video_importer() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    importer = VideoImporter(
        str(current_path / "video" / "soccer_goal.mp4"),
        str(current_path / "video" / "video_importer.yaml"),
    )
    importer.write()
