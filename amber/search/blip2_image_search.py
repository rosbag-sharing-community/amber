from amber.automation.blip2_encoder import Blip2Encoder
from amber.dataset.images_dataset import ImagesDataset, ReadImagesConfig
from amber.sampler.timestamp_sampler import TimestampSampler
from amber.unit.time import Time, TimeUnit
from glob import glob
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from typing import List
import os
import shutil
import torch
import torchvision
import uuid


class Blip2ImageSearch:
    def __init__(self, rosbag_directory: Path) -> None:
        self.client = QdrantClient("localhost", port=6333)
        self.encoder = Blip2Encoder()
        self.datasets: List[ImagesDataset] = []
        for mcap_file in glob(
            "**/*.mcap",
            root_dir=rosbag_directory.absolute().as_posix(),
            recursive=True,
        ):
            self.datasets.append(
                ImagesDataset(
                    os.path.join(rosbag_directory.absolute().as_posix(), mcap_file),
                    ReadImagesConfig.from_yaml_file(
                        os.path.join(
                            rosbag_directory.absolute().as_posix(), "dataset.yaml"
                        )
                    ),
                )
            )

    def preprocess_video(self) -> None:
        self.client.recreate_collection(
            collection_name="rosbag",
            vectors_config=VectorParams(size=254, distance=Distance.COSINE),
        )
        image_save_directory = "/tmp/blip2_image_search"
        shutil.rmtree(image_save_directory, ignore_errors=True)
        os.makedirs(image_save_directory)
        for dataset in self.datasets:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=TimestampSampler(dataset, Time(5, TimeUnit.SECOND)),
            )
            for i_batch, sample_batched in enumerate(dataloader):
                for sample in sample_batched:
                    image_features: torch.Tensor = self.encoder.encode_image(sample)
                    points = []
                    image_id = uuid.uuid4()
                    image_path = os.path.join(
                        image_save_directory, str(image_id) + ".png"
                    )
                    torchvision.transforms.functional.to_pil_image(sample).save(
                        image_path
                    )
                    for i in range(image_features[0].shape[0]):
                        points.append(
                            PointStruct(
                                id=str(image_id) + "_" + str(i),
                                vector=image_features[0][i].tolist(),
                                payload={"image_path": image_path},
                            )
                        )
                        self.client.upsert(
                            collection_name="rosbag", wait=True, points=points
                        )


if __name__ == "__main__":
    pass
