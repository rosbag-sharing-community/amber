from rosbag2_pytorch_data_loader.rosbag2_pytorch_dataset import Rosbag2Dataset
from rosbag2_pytorch_data_loader.conversion import image_to_tensor
from torch.utils.data import DataLoader
import torch
import tests
import os
import torchvision.transforms as transforms
from PIL import Image


def test_read_images() -> None:
    dataset = Rosbag2Dataset(
        os.path.join(os.path.dirname(tests.__file__), "rosbag/vrx.mcap"),
        os.path.join(os.path.dirname(tests.__file__), "read_image.yaml"),
    )
    assert len(dataset) == 2
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    transform = transforms.ToPILImage()
    count = 0
    for i_batch, sample_batched in enumerate(dataloader):
        for sample in sample_batched:
            assert torch.equal(
                sample,
                image_to_tensor(
                    Image.open(
                        os.path.join(
                            os.path.dirname(tests.__file__),
                            "images",
                            str(count) + ".png",
                        )
                    )
                ),
            )
            count = count + 1
