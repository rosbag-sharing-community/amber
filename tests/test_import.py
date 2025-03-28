from amber_mcap.importer.video import VideoImporter, VideoImporterConfig
from amber_mcap.importer.tf import TfImporter, TfImporterConfig
from amber_mcap.dataset.tf_dataset import TfDataset, ReadTfTopicConfig
from amber_mcap.tf2_amber import TransformStamped
from pathlib import Path
from torch.utils.data import DataLoader
import os
import copy


# def test_video_importer() -> None:
#     current_path = Path(os.path.dirname(os.path.realpath(__file__)))
#     importer = VideoImporter(
#         str(current_path / "video" / "soccer_goal.mp4"),
#         VideoImporterConfig.from_yaml_file(
#             str(current_path / "video" / "video_importer.yaml")
#         ),
#     )
#     importer.write()


def test_tf_importer() -> None:
    importer = TfImporter(TfImporterConfig())
    sample_data = TransformStamped()
    sample_data.child_frame_id = "base_link"
    sample_data.header.frame_id = "map"
    sample_data.header.stamp.nanosec = 0
    sample_data.header.stamp.sec = 0
    sample_data.transform.translation.x = 1
    importer.write(sample_data)
    sample_data.header.stamp.sec = 10
    importer.write(sample_data)
    importer.finish()
