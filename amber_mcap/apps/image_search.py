from amber_mcap.automation.blip2_encoder import Blip2Encoder
from amber_mcap.automation.clip_encoder import ClipEncoder
from amber_mcap.dataset.images_dataset import ImagesDataset, ReadImagesConfig
from glob import glob
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from typing import List, Any, Optional, cast
import os
import torch
import torchvision
import uuid
import docker
import gradio as gr
import hashlib
import argparse
from tqdm import tqdm
from torchvision import transforms


class SearchResult:
    duration_from_rosbag_start: float = 0
    mcap_path: str = ""
    image_path: str = ""

    def __init__(
        self, duration_from_rosbag_start: float, mcap_path: str, image_path: str
    ) -> None:
        self.duration_from_rosbag_start = duration_from_rosbag_start
        self.mcap_path = mcap_path
        self.image_path = image_path


class ImageSearch:
    def __init__(
        self,
        qdrant_port: int = 5555,
        preload_rosbag_directory: Optional[str] = None,
        model: str = "blip2",
    ) -> None:
        # make data saving directory
        self.data_directory = "/tmp/image_search"
        self.mcap_hashes: List[str] = []
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        # setup docker container
        container_name = "image_search"
        self.docker_client = docker.from_env()
        if not self.found_image(image_name="qdrant/qdrant", tag="v1.7.2"):
            self.docker_client.images.pull("qdrant/qdrant", tag="v1.7.2")
        if not self.found_container(container_name):
            self.container = self.docker_client.containers.run(
                image="qdrant/qdrant:v1.7.2",
                ports={6333: qdrant_port},
                detach=True,
                tty=True,
                name=container_name,
            )

        # setup qdrant client
        self.client = QdrantClient("localhost", port=qdrant_port)
        self.model = model
        if self.model == "blip2":
            # load blip2 models
            self.encoder = Blip2Encoder()
        elif self.model == "clip":
            self.encoder = ClipEncoder()
        else:
            raise Exception("Model type " + self.model + " does not supported.")

        if preload_rosbag_directory != None:
            self.preload_rosbag_files(Path(str(preload_rosbag_directory)))

    def collection_exists(self, collection_name: str) -> bool:
        for collections in self.client.get_collections():
            for collection in collections[1]:
                if collection.name == collection_name:
                    return True
        return False

    def is_processed_mcap(self, mcap_path: str) -> bool:
        hash = hashlib.sha256()
        with open(mcap_path, "rb") as f:
            while True:
                chunk = f.read(2048 * hash.block_size)
                if len(chunk) == 0:
                    break
                hash.update(chunk)
        digest = hash.hexdigest()
        if digest in self.mcap_hashes:
            return True
        else:
            self.mcap_hashes.append(digest)
            return False

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
                result = self.search_by_text(message)
                if result == None:
                    chat_history.append(
                        (message, ("something wrong happend in search"))
                    )
                else:
                    chat_history.append(
                        (message, (cast(SearchResult, result).image_path,))
                    )
                    chat_history.append(
                        ("Which rosbag?", cast(SearchResult, result).mcap_path)
                    )
                    chat_history.append(
                        (
                            "Around how many seconds after the start?",
                            "About "
                            + str(cast(SearchResult, result).duration_from_rosbag_start)
                            + " sec",
                        )
                    )
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

        if self.is_processed_mcap(str(mcap_path)):
            return []
        self.preprocess(
            ImagesDataset(mcap_path, ReadImagesConfig.from_yaml_file(yaml_path))
        )
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

    def search_by_text(self, text: str) -> Optional[SearchResult]:
        if self.model == "blip2":
            search_result = self.client.search(
                collection_name=self.model,
                query_vector=self.encoder.encode_text(text)[0].tolist(),
                limit=10,
            )
        elif self.model == "clip":
            search_result = self.client.search(
                collection_name=self.model,
                query_vector=self.encoder.get_single_text_embeddings(text)[0].tolist(),
                limit=10,
            )
        else:
            raise Exception("Model type " + self.model + " does not supported.")
        for result in search_result:
            return SearchResult(
                mcap_path=result.payload["mcap_path"],
                image_path=result.payload["image_path"],
                duration_from_rosbag_start=result.payload["duration_from_rosbag_start"],
            )
        return None

    def preload_rosbag_files(self, rosbag_directory: Path) -> None:
        for mcap_file in tqdm(
            glob(
                "**/*.mcap",
                root_dir=rosbag_directory.absolute().as_posix(),
                recursive=True,
            ),
            desc="preloading rosbag",
            postfix="range",
            ncols=80,
        ):
            print(
                "Start loading rosbag : "
                + os.path.join(rosbag_directory.absolute().as_posix(), mcap_file)
            )
            if not self.is_processed_mcap(
                os.path.join(rosbag_directory.absolute().as_posix(), mcap_file)
            ):
                self.preprocess(
                    ImagesDataset(
                        os.path.join(rosbag_directory.absolute().as_posix(), mcap_file),
                        ReadImagesConfig.from_yaml_file(
                            os.path.join(
                                rosbag_directory.absolute().as_posix(), "dataset.yaml"
                            )
                        ),
                    )
                )
            else:
                print("Rosbag data was already processed.")

    def preprocess(self, dataset: ImagesDataset) -> None:
        if not self.collection_exists(self.model):
            if self.model == "blip2":
                self.client.create_collection(
                    collection_name=self.model,
                    vectors_config=VectorParams(size=256, distance=Distance.COSINE),
                )
            else:
                self.client.create_collection(
                    collection_name=self.model,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                )
        batch_size = 1
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        for i_batch, sample_batched in enumerate(dataloader):
            for sample_index, sample in enumerate(
                tqdm(sample_batched, desc="processing images in rosbag.")
            ):
                image_features: torch.Tensor
                if self.model == "blip2":
                    image_features = self.encoder.encode_image(sample)
                elif self.model == "clip":
                    to_pil_image = transforms.ToPILImage()
                    image_features = self.encoder.get_single_image_embeddings(
                        to_pil_image(sample)
                    ).unsqueeze(0)
                else:
                    raise Exception("Model type " + self.model + " does not supported.")
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
                                "duration_from_rosbag_start": (
                                    dataset.get_metadata(
                                        i_batch * batch_size + sample_index
                                    ).publish_time
                                    - dataset.get_first_timestamp()
                                ).total_seconds(),
                            },
                        )
                    )
                    self.client.upsert(
                        collection_name=self.model, wait=True, points=points
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample application of searching image by blip2"
    )
    parser.add_argument("--port", default=5555, help="connection port of the qdrant")
    parser.add_argument("--model", choices=["blip2", "clip"], default="blip2")
    parser.add_argument("--rosbag_directory", default=None)
    args = parser.parse_args()
    app = ImageSearch(
        qdrant_port=args.port,
        preload_rosbag_directory=args.rosbag_directory,
        model=args.model,
    )
    app.show_gradio_ui()
