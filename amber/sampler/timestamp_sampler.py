from torch.utils.data.sampler import BatchSampler
from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from amber.unit.time import Time, TimeUnit
from torch.utils.data import DataLoader
from typing import Any, Optional, List, Iterator
from datetime import timedelta, datetime
import pandas


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
            if len(self) <= 0:
                raise RuntimeError(
                    "Something completely unexpected thing happened in making batch from timestamp."
                )
            start_timestamps = pandas.date_range(
                start=self.dataset.get_first_timestamp(),
                periods=len(self),
                freq=str(int(self.data_timestamp_duration.get(TimeUnit.SECOND))) + "S",
            )
            indices: List[List[int]] = []
            for start_timestamp in start_timestamps:
                indices.append(
                    self.dataset.get_sample_data_index_by_timestamp(
                        start_timestamp,
                        start_timestamp
                        + timedelta(seconds=self.batch_duration.get(TimeUnit.SECOND)),
                        timedelta(
                            seconds=self.data_timestamp_duration.get(TimeUnit.SECOND)
                        ),
                    )
                )
            return iter(indices)

    def __len__(self) -> int:
        if self.batch_duration == None:
            return 1
        else:
            return (
                self.dataset.get_last_timestamp() - self.dataset.get_first_timestamp()
            ).seconds
