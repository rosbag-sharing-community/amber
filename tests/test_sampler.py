from amber.sampler.timestamp_sampler import TimestampSampler
from amber.dataset.images_dataset import ImagesDataset
import os
from pathlib import Path
from amber.unit.time import Time, TimeUnit
import torch


def test_read_images_ford_with_timestamp_sampler() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    dataset = ImagesDataset(
        str(current_path / "rosbag" / "ford" / "ford.mcap"),
        str(current_path / "rosbag" / "ford" / "read_image.yaml"),
    )
    assert len(dataset) == 39
    sampler = TimestampSampler(dataset, Time(5, TimeUnit.SECOND))
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
        pass
