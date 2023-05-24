from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard, JSONWizard
from rosbag2_pytorch_data_loader.exception import TaskDescriptionError
from enum import Enum
from typing import Dict
import os


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
class DeticImageLabalerConfig(YAMLWizard):  # type: ignore
    task_type: str = "detic_image_labaler"
    model: DeticModelType = (
        DeticModelType.Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max_size
    )
    vocabulary: DeticVocabulary = DeticVocabulary.LVIS
    custom_vocabulary: list[str] = field(default_factory=list)
    config_file: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
    confidence_threshold: float = 0.5
    video_output_path: str = ""  # If the text is empty, it means no video output.

    def validate(self) -> None:
        if (
            self.video_output_path != ""
            and os.path.splitext(self.video_output_path)[-1] != ".mp4"
        ):
            raise TaskDescriptionError(
                "Type of the output video should be mp4, you specified "
                + self.video_output_path
            )
