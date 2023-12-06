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
        self.mcap_files: List[str] = []
        for mcap_file in glob(
            "**/*.mcap",
            root_dir=rosbag_directory.absolute().as_posix(),
            recursive=True,
        ):
            self.mcap_files.append(
                os.path.join(rosbag_directory.absolute().as_posix(), mcap_file)
            )

    def search_by_text(self, text: str) -> None:
        search_result = self.client.search(
            collection_name="rosbag",
            query_vector=self.encoder.encode_text(text)[0].tolist(),
            limit=10,
        )
        for result in search_result:
            print(result)

    def preprocess_video(self) -> None:
        self.client.recreate_collection(
            collection_name="rosbag",
            vectors_config=VectorParams(size=256, distance=Distance.COSINE),
        )
        image_save_directory = "/tmp/blip2_image_search"
        shutil.rmtree(image_save_directory, ignore_errors=True)
        os.makedirs(image_save_directory)
        for mcap_file in self.mcap_files:
            dataset = ImagesDataset(
                mcap_file,
                ReadImagesConfig.from_yaml_file(
                    os.path.join(
                        os.path.dirname(mcap_file),
                        "dataset.yaml",
                    )
                ),
            )
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
                                id=str(uuid.uuid4()),
                                vector=image_features[0][i].tolist(),
                                payload={
                                    "image_path": image_path,
                                    "image_id": image_id,
                                    "mcap_path": dataset.rosbag_files,
                                },
                            )
                        )
                        self.client.upsert(
                            collection_name="rosbag", wait=True, points=points
                        )


if __name__ == "__main__":
    pass
