from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from amber.exception import TaskDescriptionError
from enum import Enum
from typing import Dict
import os


class AutomationTaskType(Enum):
    DETIC_IMAGE_LABELER = "detic_image_labeler"


class DeticModelType(Enum):
    Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max_size = (
        "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size"
    )


class DeticVocabulary(Enum):
    LVIS = "lvis"
    OPENIMAGES = "openimages"
    OBJECT365 = "objects365"
    COCO = "coco"
    CUSTOM = "custom"


@dataclass
class DockerConfig(YAMLWizard):  # type: ignore
    use_gpu: bool = False


@dataclass
class DeticImageLabalerConfig(YAMLWizard):  # type: ignore
    task_type: AutomationTaskType = AutomationTaskType.DETIC_IMAGE_LABELER
    model: DeticModelType = (
        DeticModelType.Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max_size
    )
    vocabulary: DeticVocabulary = DeticVocabulary.LVIS
    custom_vocabulary: list[str] = field(default_factory=list)
    config_file: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
    confidence_threshold: float = 0.5
    video_output_path: str = ""  # If the text is empty, it means no video output.
    docker_config: DockerConfig = DockerConfig()

    def validate(self) -> None:
        if (
            self.video_output_path != ""
            and os.path.splitext(self.video_output_path)[-1] != ".mp4"
        ):
            raise TaskDescriptionError(
                "Type of the output video should be mp4, you specified "
                + self.video_output_path
            )
