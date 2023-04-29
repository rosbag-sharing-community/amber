from rosbag2_pytorch_data_loader.dataset.rosbag2_pytorch_dataset import Rosbag2Dataset
from rosbag2_pytorch_data_loader.dataset.conversion import image_to_tensor
from torch.utils.data import DataLoader
import torch
import tests
import os
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path


def test_read_images() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    dataset = Rosbag2Dataset(
        str(current_path / "rosbag" / "vrx" / "vrx.mcap"),
        str(current_path / "read_image.yaml"),
    )
    assert len(dataset) == 2
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    transform = transforms.ToPILImage()
    count = 0
    for i_batch, sample_batched in enumerate(dataloader):
        for sample in sample_batched:
            image = str(count) + ".png"
            assert torch.equal(
                sample,
                image_to_tensor(
                    Image.open(
                        str(current_path / "images" / image),
                    )
                ),
            )
            count = count + 1
