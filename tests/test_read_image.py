from rosbag2_pytorch_data_loader.rosbag2_pytorch_dataset import Rosbag2Dataset
from torch.utils.data import DataLoader
import tests
import os
import torchvision.transforms as T


def test_read_images() -> None:
    dataset = Rosbag2Dataset(
        os.path.join(os.path.dirname(tests.__file__), "rosbag/vrx.mcap"),
        os.path.join(os.path.dirname(tests.__file__), "read_image.yaml"),
    )
    assert len(dataset) == 2
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    transform = T.ToPILImage()
    count = 0
    for i_batch, sample_batched in enumerate(dataloader):
        for sample in sample_batched:
            # transform(sample_batched[0]).save(str(count) + ".png", quality=100)
            print(count)
            count = count + 1
