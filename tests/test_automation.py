from amber.automation.detic_image_labeler import DeticImageLabeler
from amber.automation.nerf_3d_reconstruction import Nerf3DReconstruction
from pathlib import Path
import os
from amber.dataset.rosbag2_dataset import Rosbag2Dataset


# def test_detic_auto_labeler() -> None:
#     current_path = Path(os.path.dirname(os.path.realpath(__file__)))
#     labeler = DeticImageLabeler(str(current_path / "detic_image_labeler.yaml"))
#     dataset = Rosbag2Dataset(
#         str(current_path / "rosbag" / "ford" / "ford.mcap"),
#         str(current_path / "read_image_ford.yaml"),
#     )
#     labeler.inference(dataset)


def test_colmap_pose_estimation() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    labeler = Nerf3DReconstruction(str(current_path / "nerf_3d_reconstruction.yaml"))
    dataset = Rosbag2Dataset(
        str(current_path / "rosbag" / "soccer_goal"),
        str(current_path / "read_images_soccer_goal.yaml"),
    )
    labeler.inference(dataset)
