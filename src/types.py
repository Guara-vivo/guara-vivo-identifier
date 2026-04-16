from dataclasses import dataclass

import numpy as np


@dataclass
class BirdDetection:
    bbox_xyxy: list[int]
    yolo_confidence: float
    crop_rgb: np.ndarray
    crop_mask: np.ndarray | None = None


@dataclass
class SpeciesPrediction:
    label: str
    confidence: float


@dataclass
class ColorPrediction:
    color_label: str
    life_stage: str
    confidence: float
    dominant_rgb: list[int]
    color_source: str


@dataclass
class DistanceEstimation:
    distance_m: float | None
    uncertainty_m: float | None
    confidence: float
    method: str
    pixel_width: int
    pixel_height: int
    pixel_source: str


@dataclass
class GuaraIdentification:
    detection_id: int
    bbox_xyxy: list[int]
    color: str
    life_stage: str
    color_source: str
    distance_m: float | None
    distance_uncertainty_m: float | None
    distance_confidence: float
    distance_method: str
    distance_pixel_width: int
    distance_pixel_height: int
    distance_pixel_source: str
    yolo_confidence: float
    species_confidence: float
    color_confidence: float
    dominant_rgb: list[int]
