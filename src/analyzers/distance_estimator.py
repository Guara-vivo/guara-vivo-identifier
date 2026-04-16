import numpy as np

from src.config import DistanceEstimatorConfig
from src.types import BirdDetection, DistanceEstimation


class GuaraDistanceEstimator:
    def __init__(self, config: DistanceEstimatorConfig) -> None:
        self._config = config

    def estimate(self, detection: BirdDetection) -> DistanceEstimation:
        width_px, height_px, source = self._extract_object_dimensions(detection)
        if width_px < self._config.min_object_pixels or height_px < self._config.min_object_pixels:
            return DistanceEstimation(
                distance_m=None,
                uncertainty_m=None,
                confidence=0.0,
                method="insufficient_pixels",
                pixel_width=width_px,
                pixel_height=height_px,
                pixel_source=source,
            )

        wingspan_cm = (self._config.adult_wingspan_cm_min + self._config.adult_wingspan_cm_max) / 2.0
        distance_height_m = self._estimate_distance(
            real_size_cm=self._config.adult_height_cm,
            observed_size_px=height_px,
        )
        distance_wingspan_m = self._estimate_distance(
            real_size_cm=wingspan_cm,
            observed_size_px=width_px,
        )

        aspect_ratio = width_px / max(height_px, 1)
        if aspect_ratio >= self._config.wingspan_aspect_ratio_threshold:
            primary_distance = distance_wingspan_m
            method = "wingspan"
        else:
            primary_distance = distance_height_m
            method = "height"

        disagreement = abs(distance_height_m - distance_wingspan_m)
        uncertainty_m = max(
            primary_distance * self._config.min_relative_uncertainty,
            disagreement * 0.5,
        )
        confidence = self._estimate_confidence(
            source=source,
            width_px=width_px,
            height_px=height_px,
            disagreement=disagreement,
            distance=primary_distance,
        )

        return DistanceEstimation(
            distance_m=round(primary_distance, 2),
            uncertainty_m=round(uncertainty_m, 2),
            confidence=round(confidence, 4),
            method=method,
            pixel_width=width_px,
            pixel_height=height_px,
            pixel_source=source,
        )

    def _extract_object_dimensions(self, detection: BirdDetection) -> tuple[int, int, str]:
        x1, y1, x2, y2 = detection.bbox_xyxy
        bbox_width = max(1, x2 - x1)
        bbox_height = max(1, y2 - y1)

        if detection.crop_mask is None:
            return bbox_width, bbox_height, "bbox"

        ys, xs = np.where(detection.crop_mask > 0)
        if ys.size == 0 or xs.size == 0:
            return bbox_width, bbox_height, "bbox_fallback_empty_mask"

        mask_width = int(xs.max() - xs.min() + 1)
        mask_height = int(ys.max() - ys.min() + 1)
        return max(1, mask_width), max(1, mask_height), "mask"

    def _estimate_distance(self, real_size_cm: float, observed_size_px: int) -> float:
        distance_cm = (self._config.focal_length_px * real_size_cm) / max(observed_size_px, 1)
        return distance_cm / 100.0

    def _estimate_confidence(
        self,
        source: str,
        width_px: int,
        height_px: int,
        disagreement: float,
        distance: float,
    ) -> float:
        confidence = 0.75 if source == "mask" else 0.55

        min_side = min(width_px, height_px)
        if min_side >= 120:
            confidence += 0.15
        elif min_side >= 60:
            confidence += 0.08
        else:
            confidence -= 0.1

        disagreement_ratio = disagreement / max(distance, 0.01)
        if disagreement_ratio <= 0.2:
            confidence += 0.08
        elif disagreement_ratio >= 0.5:
            confidence -= 0.12

        return float(np.clip(confidence, 0.0, 0.99))
