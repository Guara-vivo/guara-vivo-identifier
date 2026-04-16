import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.config import ColorAnalyzerConfig
from src.types import ColorPrediction


class GuaraColorAnalyzer:
    def __init__(self, config: ColorAnalyzerConfig) -> None:
        self._config = config
        self._red_references = [
            np.array([255.0, 70.0, 60.0], dtype=np.float32),
            np.array([220.0, 60.0, 50.0], dtype=np.float32),
            np.array([180.0, 50.0, 45.0], dtype=np.float32),
        ]
        self._gray_reference = np.array([140.0, 140.0, 140.0], dtype=np.float32)

    def analyze(self, crop_rgb: np.ndarray, crop_mask: np.ndarray | None = None) -> ColorPrediction:
        if crop_rgb.size == 0:
            return ColorPrediction(
                color_label="indefinido",
                life_stage="indefinido",
                confidence=0.0,
                dominant_rgb=[0, 0, 0],
                color_source="bbox",
            )

        pixels, color_source = self._extract_color_pixels(crop_rgb, crop_mask)
        if pixels.size == 0:
            return ColorPrediction(
                color_label="indefinido",
                life_stage="indefinido",
                confidence=0.0,
                dominant_rgb=[0, 0, 0],
                color_source=color_source,
            )

        dominant_rgb = self._dominant_color(pixels)
        color_label, life_stage, confidence = self._classify_color(dominant_rgb)

        return ColorPrediction(
            color_label=color_label,
            life_stage=life_stage,
            confidence=confidence,
            dominant_rgb=dominant_rgb,
            color_source=color_source,
        )

    def _extract_color_pixels(
        self,
        crop_rgb: np.ndarray,
        crop_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, str]:
        if not self._config.use_mask_for_color or crop_mask is None:
            pixels = crop_rgb.reshape(-1, 3).astype(np.float32)
            return pixels, "bbox"

        mask = self._prepare_mask(crop_mask, crop_rgb.shape[:2])
        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        if coverage < self._config.min_mask_coverage:
            pixels = crop_rgb.reshape(-1, 3).astype(np.float32)
            return pixels, "bbox_fallback_low_coverage"

        masked_pixels = crop_rgb[mask > 0]
        if masked_pixels.shape[0] < self._config.min_mask_pixels:
            pixels = crop_rgb.reshape(-1, 3).astype(np.float32)
            return pixels, "bbox_fallback_low_pixels"

        return masked_pixels.astype(np.float32), "mask"

    def _prepare_mask(self, crop_mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        mask = crop_mask
        if mask.shape[0] != target_shape[0] or mask.shape[1] != target_shape[1]:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        kernel_size = max(1, self._config.mask_erosion_kernel_size)
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        return (mask > 0).astype(np.uint8)

    def _dominant_color(self, pixels: np.ndarray) -> list[int]:
        if pixels.shape[0] == 0:
            return [0, 0, 0]

        if pixels.shape[0] > self._config.resized_width * self._config.resized_height:
            sample_size = self._config.resized_width * self._config.resized_height
            indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
            pixels = pixels[indices]

        n_clusters = min(self._config.kmeans_clusters, max(1, pixels.shape[0]))

        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=42,
        )
        labels = kmeans.fit_predict(pixels)

        counts = np.bincount(labels)
        dominant_index = int(np.argmax(counts))
        center = kmeans.cluster_centers_[dominant_index]
        center = np.clip(center, 0, 255).astype(np.uint8)

        return [int(center[0]), int(center[1]), int(center[2])]

    def _classify_color(self, dominant_rgb: list[int]) -> tuple[str, str, float]:
        pixel = np.array(dominant_rgb, dtype=np.float32)

        max_distance = float(np.sqrt(3.0 * (255.0**2)))
        red_distance = min(float(np.linalg.norm(pixel - ref)) for ref in self._red_references)
        gray_distance = float(np.linalg.norm(pixel - self._gray_reference))

        red_score = max(0.0, 1.0 - (red_distance / max_distance))
        gray_score = max(0.0, 1.0 - (gray_distance / max_distance))

        score_total = red_score + gray_score
        if score_total <= 0.0:
            return "indefinido", "indefinido", 0.0

        if red_score >= gray_score:
            confidence = red_score / score_total
            return "vermelho", "adulto", float(confidence)

        confidence = gray_score / score_total
        return "cinza", "filhote", float(confidence)
