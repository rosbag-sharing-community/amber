import docker
from amber.automation.automation import Automation
import shutil
import os
from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from amber.automation.task_description import ColmapPoseEstimationConfig


class ColmapPoseEstimation(Automation):  # type: ignore
    docker_image_name = "dromni/nerfstudio:0.3.1"

    def __init__(self, yaml_path: str) -> None:
        self.config = ColmapPoseEstimationConfig.from_yaml_file(yaml_path)
        self.temporary_image_directory = "/tmp/colmap_pose_estimation"
        self.docker_client = docker.from_env()
        self.docker_client.images.pull("dromni/nerfstudio", tag="0.3.1")
        self.container = self.docker_client.containers.run(
            image="dromni/nerfstudio:0.3.1",
            volumes={
                os.path.join(self.temporary_image_directory, "inputs"): {
                    "bind": "/workspace/inputs",
                    "mode": "rw",
                },
                os.path.join(self.temporary_image_directory, "outputs"): {
                    "bind": "/workspace/outputs",
                    "mode": "rw",
                },
            },
            device_requests=self.build_device_requests(),
            command=["/bin/sh"],
            detach=True,
            tty=True,
            runtime=None,
        )

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

    def setup_directory(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
        os.makedirs(os.path.join(self.temporary_image_directory, "inputs"))
        os.makedirs(os.path.join(self.temporary_image_directory, "outputs"))

    def inference(self, dataset: Rosbag2Dataset) -> None:
        pass
