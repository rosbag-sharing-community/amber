from torch.utils.data.sampler import BatchSampler
from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from amber.unit.time import Time, TimeUnit
from torch.utils.data import DataLoader
from typing import Any, Optional, List, Tuple
from datetime import timedelta, datetime


class TimestampSampler(BatchSampler):  # type: ignore
    def __init__(
        self,
        dataset: Rosbag2Dataset,
        data_timestamp_duration: Time = Time(0, TimeUnit.SECOND),
        batch_duration: Optional[Time] = None,
    ):
        # self.data_timestamp_duration = data_timestamp_duration
        # self.batch_duration = batch_duration
        self.batched_data: List[List[Any]] = []
        self.dataset = dataset
        if batch_duration == None:
            sampled_data = self.dataset.sample_data_by_timestamp(
                self.dataset.get_first_timestamp(),
                self.dataset.get_last_timestamp(),
                timedelta(seconds=data_timestamp_duration.get(TimeUnit.SECOND)),
            )
        else:
            pass
            # batch_start_timestamp = self.dataset.get_first_timestamp()
            # self.dataset.get_data_in_target_duration(
            #     batch_start_timestamp,
            #     batch_start_timestamp
            #     + datetime.timedelta(
            #         seconds=data_timestamp_duration.get(TimeUnit.SECOND)
            #     ),
            # )

    def __iter__(self) -> None:
        first_timestamp = self.dataset.get_first_timestamp()
        # for index, data in enumerate(self.dataset):
        #     self.dataset.get_metadata(index).publish_time
