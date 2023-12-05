from amber.sampler.timestamp_sampler import TimestampSampler
from amber.dataset.images_dataset import ImagesDataset, ReadImagesConfig
import os
from pathlib import Path
from amber.unit.time import Time, TimeUnit
import torch


def test_read_images_ford_with_timestamp_sampler() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    dataset = ImagesDataset(
        str(current_path / "rosbag" / "ford" / "ford.mcap"),
        ReadImagesConfig.from_yaml_file(
            str(current_path / "rosbag" / "ford" / "read_image.yaml")
        ),
    )
    assert len(dataset) == 39
    dataloader_no_batched = torch.utils.data.DataLoader(
        dataset, batch_sampler=TimestampSampler(dataset, Time(5, TimeUnit.SECOND))
    )
    for i_batch, sample_batched in enumerate(dataloader_no_batched):
        if i_batch == 0:
            assert len(sample_batched) == 5
        assert i_batch == 0
    dataloader_batched = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=TimestampSampler(
            dataset, Time(1, TimeUnit.SECOND), Time(5, TimeUnit.SECOND)
        ),
    )
    for i_batch, sample_batched in enumerate(dataloader_batched):
        assert len(sample_batched) <= 4
