from dataclass_wizard import JSONWizard
from typing import List
from dataclasses import dataclass, field


@dataclass
class BoundingBox(JSONWizard):  # type: ignore
    x1: float = 0.0
    x2: float = 0.0
    y1: float = 0.0
    y2: float = 0.0


@dataclass
class BoundingBoxAnnotation(JSONWizard):  # type: ignore
    box: BoundingBox = BoundingBox()
    score: float = 0
    object_class: str = ""


@dataclass
class ImageAnnotations(JSONWizard):  # type: ignore
    image_index: int = 0
    bounding_boxes: List[BoundingBoxAnnotation] = field(default_factory=list)
