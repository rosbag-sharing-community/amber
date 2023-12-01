from torch.utils.data.sampler import BatchSampler
from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from amber.unit.time import Time, TimeUnit
from torch.utils.data import DataLoader
from typing import Any, Optional, List, Iterator
from datetime import timedelta, datetime


class TimestampSampler(BatchSampler):  # type: ignore
    def __init__(
        self,
        dataset: Rosbag2Dataset,
        data_timestamp_duration: Time = Time(0, TimeUnit.SECOND),
        batch_duration: Optional[Time] = None,
    ):
        self.data_timestamp_duration = data_timestamp_duration
        self.batch_duration = batch_duration
        self.dataset = dataset

    def __iter__(self) -> Iterator[List[int]]:
        if self.batch_duration == None:
            return iter(
                [
                    self.dataset.get_sample_data_index_by_timestamp(
                        self.dataset.get_first_timestamp(),
                        self.dataset.get_last_timestamp(),
                        timedelta(
                            seconds=self.data_timestamp_duration.get(TimeUnit.SECOND)
                        ),
                    )
                ]
            )
        else:
            return iter([])

    def __len__(self) -> int:
        if self.batch_duration == None:
            return len(
                self.dataset.get_sample_data_index_by_timestamp(
                    self.dataset.get_first_timestamp(),
                    self.dataset.get_last_timestamp(),
                    timedelta(
                        seconds=self.data_timestamp_duration.get(TimeUnit.SECOND)
                    ),
                )
            )
        else:
            return 0
