import docker
from amber.automation.automation import Automation
import shutil
import os
from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from amber.automation.task_description import ColmapPoseEstimationConfig
from typing import List
from torchvision import transforms
import uuid


class Nerf3DReconstruction(Automation):  # type: ignore
    docker_image_name = "dromni/nerfstudio:0.3.1"

    def __init__(self, yaml_path: str) -> None:
        self.config = ColmapPoseEstimationConfig.from_yaml_file(yaml_path)
        self.temporary_image_directory = "/tmp/nerf_3d_reconstruction/" + str(
            uuid.uuid4()
        )
        self.setup_directory()
        self.to_pil_image = transforms.ToPILImage()
        self.docker_client = docker.from_env()
        self.docker_client.images.pull("dromni/nerfstudio", tag="0.3.1")
        self.container = self.docker_client.containers.run(
            image="dromni/nerfstudio:0.3.1",
            volumes={
                self.get_input_directory_path(): {
                    "bind": "/workspace/inputs",
                    "mode": "rw",
                },
                self.get_output_directory_path(): {
                    "bind": "/workspace/outputs",
                    "mode": "rw",
                },
            },
            device_requests=self.build_device_requests(),
            shm_size=self.config.docker_config.shm_size,
            command=["/bin/bash"],
            detach=True,
            tty=True,
            runtime=None,
        )

    def get_input_directory_path(self) -> str:
        return os.path.join(self.temporary_image_directory, "inputs")

    def get_output_directory_path(self) -> str:
        return os.path.join(self.temporary_image_directory, "outputs")

    def get_camera_pose_output_directory_path(self) -> str:
        return os.path.join(self.get_output_directory_path(), "camera_poses")

    def get_checkpoint_output_directory_path(self) -> str:
        return os.path.join(self.get_output_directory_path(), "checkpoints")

    def get_trained_model_output_directory_path(self) -> str:
        return os.path.join(self.get_output_directory_path(), "models")

    def __del__(self) -> None:
        self.container.stop()
        self.container.remove()
        if self.config.docker_config.claenup_image_on_shutdown:
            self.docker_client.images.remove(
                image=self.docker_image_name, force=False, noprune=False
            )

    def build_device_requests(self) -> list[docker.types.DeviceRequest]:
        requests = []
        if self.config.docker_config.use_gpu:
            requests.append(
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            )
        return requests

    def setup_directory(self) -> None:
        if not os.path.exists(self.temporary_image_directory):
            os.makedirs(self.temporary_image_directory)
        os.makedirs(self.get_input_directory_path())

    def cpu_or_gpu(self) -> str:
        if self.config.docker_config.use_gpu:
            return "--gpu"
        else:
            return "--no-gpu"

    def build_post_process_command(self) -> List[str]:
        return [
            "ns-process-data",
            "images",
            "--data",
            "/workspace/inputs",
            "--output-dir",
            "/workspace/outputs/camera_poses",
            self.cpu_or_gpu(),
        ]

    def build_train_command(self) -> List[str]:
        return [
            "ns-train",
            "instant-ngp",
            "--data",
            "/workspace/outputs/camera_poses",
            "--output-dir",
            self.get_checkpoint_output_directory_path(),
        ]

    def run_command(self, command: List[str]) -> None:
        _, stream = self.container.exec_run(command, stream=True)
        for data in stream:
            print(data.decode())

    def inference(self, dataset: Rosbag2Dataset) -> None:
        for index, image in enumerate(dataset):
            self.to_pil_image(image).save(
                os.path.join(
                    self.get_input_directory_path(),
                    "input" + str(index) + ".jpeg",
                )
            )
        self.run_command(self.build_post_process_command())
        self.run_command(self.build_train_command())
