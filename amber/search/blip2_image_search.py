from amber.automation.blip2_encoder import Blip2Encoder
from amber.sampler.timestamp_sampler import TimestampSampler
from pathlib import Path
from glob import glob
from typing import List
import os
import torch
from amber.unit.time import Time, TimeUnit


class Blip2ImageSearch:
    def __init__(self, rosbag_directory: Path) -> None:
        self.encoder = Blip2Encoder()
        self.mcap_files: List[Path] = []
        for mcap_file in glob(
            "**/*.mcap",
            root_dir=rosbag_directory.absolute().as_posix(),
            recursive=True,
        ):
            self.mcap_files.append(
                Path(os.path.join(rosbag_directory.absolute().as_posix(), mcap_file))
            )

    def preprocess_video(self, target_topics: str) -> None:
        for mcap_file in self.mcap_files:
            pass
            # dataloader_no_batched = torch.utils.data.DataLoader(
            #     dataset,
            #     batch_sampler=TimestampSampler(dataset, Time(5, TimeUnit.SECOND)),
            # )


if __name__ == "__main__":
    pass
