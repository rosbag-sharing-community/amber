from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from amber_mcap.exception import TaskDescriptionError
from enum import Enum
from typing import Dict
import os


class DeticModelType(Enum):
    SwinB_896_4x = "SwinB_896_4x"
    R50_640_4x = "R50_640_4x"


class DeticVocabulary(Enum):
    LVIS = "lvis"
    IMAGENET_21K = "imagenet_21k"


@dataclass
class DockerConfig(YAMLWizard):  # type: ignore
    use_gpu: bool = False
    claenup_image_on_shutdown: bool = False
    shm_size: str = "64M"


@dataclass
class DeticImageLabalerConfig(YAMLWizard):  # type: ignore
    model_type: DeticModelType = DeticModelType.SwinB_896_4x
    vocabulary: DeticVocabulary = DeticVocabulary.LVIS
    video_output_path: str = ""  # If the text is empty, it means no video output.
    min_height: int = 5
    min_width: int = 5
    min_area: int = 50

    def validate(self) -> None:
        if (
            self.video_output_path != ""
            and os.path.splitext(self.video_output_path)[-1] != ".mp4"
        ):
            raise TaskDescriptionError(
                "Type of the output video should be mp4, you specified "
                + self.video_output_path
            )

    def get_onnx_filename(self) -> str:
        if self.model_type == DeticModelType.SwinB_896_4x:
            if self.vocabulary == DeticVocabulary.LVIS:
                return "Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis_op16.onnx"
            elif self.vocabulary == DeticVocabulary.IMAGENET_21K:
                return "Detic_C2_SwinB_896_4x_IN-21K+COCO_in21k_op16.onnx"
        elif self.model_type == DeticModelType.R50_640_4x:
            if self.vocabulary == DeticVocabulary.LVIS:
                return "Detic_C2_R50_640_4x_lvis_op16.onnx"
            elif self.vocabulary == DeticVocabulary.IMAGENET_21K:
                return "Detic_C2_R50_640_4x_in21k_op16.onnx"

        raise TaskDescriptionError(
            "Model type and vocablary is invalid, please check setting. You specified, model_type : "
            + self.model_type.value
            + " , vocabulary : ",
            self.vocablary.value,
        )


class ClipClassifyMethod(Enum):
    CLIP_WITH_LVIS_AND_CUSTOM_VOCABULARY = "clip_with_lvis_and_custom_vocabulary"
    CONSIDER_ANNOTATION_WITH_BERT = "consider_annotation_with_bert"


@dataclass
class ConsiderAnnotationWithBerfConfig(YAMLWizard):  # type: ignore
    positive_nagative_ratio: float = 2.0
    min_clip_cosine_similarity: float = 0.25
    min_clip_cosine_similarity_with_bert: float = 0.5


@dataclass
class ClipImageAnnotationFilterConfig(YAMLWizard):  # type: ignore
    target_objects: list[str] = field(default_factory=list)
    min_height: int = 5
    min_width: int = 5
    min_area: int = 50
    classify_method: ClipClassifyMethod = (
        ClipClassifyMethod.CONSIDER_ANNOTATION_WITH_BERT
    )
    consider_annotation_with_bert_config: ConsiderAnnotationWithBerfConfig = (
        ConsiderAnnotationWithBerfConfig()
    )


@dataclass
class ColmapPoseEstimationConfig(YAMLWizard):  # type: ignore
    docker_config: DockerConfig = DockerConfig()
