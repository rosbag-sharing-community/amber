import os
from torch.utils.data import IterableDataset
from typing import Any, List, Optional
from yaml import safe_load  # type: ignore
from amber_mcap.exception import CertificationError
from dataclasses import dataclass
from dataclass_wizard import JSONWizard
import glob
import boto3
import requests  # type: ignore
import datetime


@dataclass
class MessageMetaData(JSONWizard):  # type: ignore
    topic: str = ""
    rosbag_path: str = ""
    publish_time: datetime.datetime = datetime.datetime.fromtimestamp(
        0, datetime.timezone.utc
    )  # Unix epoch time


class Rosbag2Dataset(IterableDataset):  # type: ignore
    message_metadata: List[MessageMetaData] = []
    compressed: bool = False

    def __init__(
        self,
        rosbag_path: str,
        compressed: bool,
        transform: Any = None,
        target_transform: Any = None,
    ) -> None:
        if os.path.isfile(rosbag_path):
            self.rosbag_files = [rosbag_path]
        else:
            self.rosbag_files = glob.glob(rosbag_path + "/**/*.mcap", recursive=True)
        self.compressed = compressed
        self.transform = transform
        self.target_transform = target_transform
        self.message_metadata.clear()

    def get_sample_data_index_by_timestamp(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        data_duration: datetime.timedelta,
    ) -> List[int]:
        last_publish_timestamp_in_sample: datetime.datetime = (
            datetime.datetime.fromtimestamp(0, datetime.timezone.utc)
        )  # Unix epoch time
        indices: List[int] = []
        assert len(self.message_metadata) == len(self)
        for i in range(len(self.message_metadata)):
            if (
                self.get_metadata(i).publish_time > start_time
                and self.get_metadata(i).publish_time < end_time
            ):
                if len(indices) == 0:
                    indices.append(i)
                    last_publish_timestamp_in_sample = self.get_metadata(i).publish_time
                elif (
                    last_publish_timestamp_in_sample + data_duration
                    <= self.get_metadata(i).publish_time
                ):
                    indices.append(i)
                    last_publish_timestamp_in_sample = self.get_metadata(i).publish_time
        return indices

    def get_metadata(self, index: int) -> MessageMetaData:
        return self.message_metadata[index]

    def get_first_timestamp(self) -> datetime.datetime:
        assert len(self.message_metadata) != 0
        return min(self.message_metadata, key=lambda x: x.publish_time).publish_time

    def get_last_timestamp(self) -> datetime.datetime:
        assert len(self.message_metadata) != 0
        return max(self.message_metadata, key=lambda x: x.publish_time).publish_time


def download_rosbag(
    bucket_name: str,
    remote_rosbag_directory: str,
    remote_rosbag_filename: str,
    endpoint_url: str,
    download_dir: str = "/tmp/amber/remote_bags",
    is_public: bool = True,
    false_overwrite: bool = False,
    aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY"),
) -> str:
    if aws_access_key_id is None:
        raise CertificationError("You should specify aws access key.")
    if aws_secret_access_key is None:
        raise CertificationError("You should specify aws secret access key.")
    if is_public:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        remote_rosbag_path = os.path.join(
            remote_rosbag_directory, remote_rosbag_filename
        )
        local_rosbag_path = os.path.join(download_dir, remote_rosbag_path)
        if os.path.exists(local_rosbag_path) and not false_overwrite:
            return local_rosbag_path
        with open(local_rosbag_path, mode="wb") as f:
            f.write(
                requests.get(
                    s3.generate_presigned_url(
                        ClientMethod="get_object",
                        Params={"Bucket": bucket_name, "Key": remote_rosbag_path},
                        ExpiresIn=3600,
                        HttpMethod="GET",
                    )
                ).content
            )
        return local_rosbag_path
    else:
        s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        remote_rosbag_path = os.path.join(
            remote_rosbag_directory, remote_rosbag_filename
        )
        bucket = s3.Bucket(bucket_name)
        local_rosbag_path = os.path.join(download_dir, remote_rosbag_path)
        if os.path.exists(local_rosbag_path) and not false_overwrite:
            return local_rosbag_path
        bucket.download_file(remote_rosbag_path, local_rosbag_path)
        return local_rosbag_path
