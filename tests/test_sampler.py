from amber.sampler.timestamp_sampler import TimestampSampler
from amber.dataset.images_dataset import ImagesDataset
import os
from pathlib import Path
from amber.unit.time import Time, TimeUnit


def test_read_images_ford_with_timestamp_sampler() -> None:
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    dataset = ImagesDataset(
        str(current_path / "rosbag" / "ford" / "ford.mcap"),
        str(current_path / "rosbag" / "ford" / "read_image.yaml"),
    )
    assert len(dataset) == 39
    sampler = TimestampSampler(dataset, Time(5, TimeUnit.SECOND))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
