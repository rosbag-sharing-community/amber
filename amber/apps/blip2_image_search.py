from amber.automation.blip2_encoder import Blip2Encoder
from amber.dataset.images_dataset import ImagesDataset, ReadImagesConfig
from amber.sampler.timestamp_sampler import TimestampSampler
from amber.unit.time import Time, TimeUnit
from glob import glob
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from typing import List, Any, Optional
import os
import torch
import torchvision
import uuid
import docker
import gradio as gr


class Blip2ImageSearch:
    def __init__(self, qdrant_port: int = 6333) -> None:
        # make data saving directory
        self.data_directory = "/tmp/blip2_image_search"
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        # setup docker container
        container_name = "blip2_image_search"
        self.docker_client = docker.from_env()
        if not self.found_image(image_name="qdrant/qdrant", tag="v1.7.2"):
            self.docker_client.images.pull("qdrant/qdrant", tag="v1.7.2")
        if not self.found_container(container_name):
            self.container = self.docker_client.containers.run(
                image="qdrant/qdrant:v1.7.2",
                ports={qdrant_port: qdrant_port},
                detach=True,
                tty=True,
                name=container_name,
            )

        # setup qdrant client
        self.client = QdrantClient("localhost", port=qdrant_port)
        # load blip2 models
        self.encoder = Blip2Encoder()

    def show_gradio_ui(self) -> None:
        with gr.Blocks() as gradio_ui:
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.ClearButton([msg, chatbot])
            upload_button = gr.UploadButton(
                "Click to upload a directory which contains rosbag in mcap format and dataset.yaml",
                file_count="directory",
            )
            file_output = gr.File()
            upload_button.upload(self.upload_file, upload_button, file_output)

            def respond(message: Any, chat_history: Any) -> Any:
                # chat_history.append((message, ("lion.jpeg",)))
                chat_history.append((message, ("fuga")))
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
        gradio_ui.launch()

    def upload_file(self, files: List[Any]) -> List[str]:
        mcap_path: Optional[str] = None
        yaml_path: Optional[str] = None
        for file in files:
            if os.path.splitext(file.name)[-1] == ".mcap":
                if mcap_path != None:
                    return []
                mcap_path = file.name
            if os.path.basename(file.name) == "dataset.yaml":
                if yaml_path != None:
                    return []
                yaml_path = file.name
        if mcap_path == None or yaml_path == None:
            return []
        return [str(mcap_path), str(yaml_path)]

    def found_container(self, container_name: str) -> bool:
        for container in self.docker_client.containers.list():
            if container.name == container_name:
                return True
        return False

    def found_image(self, image_name: str, tag: str = "latest") -> bool:
        qdrant_images = self.docker_client.images.list(
            filters={"reference": image_name}
        )
        for qdrant_image in qdrant_images:
            for docker_tag in qdrant_image.tags:
                if docker_tag == image_name + ":" + tag:
                    return True
        return False

    def search_by_text(self, text: str) -> None:
        search_result = self.client.search(
            collection_name="rosbag",
            query_vector=self.encoder.encode_text(text)[0].tolist(),
            limit=10,
        )
        for result in search_result:
            print(result)

    def get_rosbag_files(self, rosbag_directory: Path) -> None:
        self.mcap_files: List[str] = []
        for mcap_file in glob(
            "**/*.mcap",
            root_dir=rosbag_directory.absolute().as_posix(),
            recursive=True,
        ):
            self.mcap_files.append(
                os.path.join(rosbag_directory.absolute().as_posix(), mcap_file)
            )

    def preprocess_video(self, dataset: ImagesDataset) -> None:
        self.client.recreate_collection(
            collection_name="rosbag",
            vectors_config=VectorParams(size=256, distance=Distance.COSINE),
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
                image_path = os.path.join(self.data_directory, str(image_id) + ".png")
                torchvision.transforms.functional.to_pil_image(sample).save(image_path)
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
    app = Blip2ImageSearch()
    app.show_gradio_ui()
