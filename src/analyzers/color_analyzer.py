import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.config import ColorAnalyzerConfig
from src.types import ColorPrediction


WARNING_UNCERTAIN_STAGE = "fase_vida_incerta"
WARNING_SMALL_BIRD = "ave_muito_pequena"
WARNING_NO_MASK = "mascara_indisponivel"


class GuaraColorAnalyzer:
    def __init__(self, config: ColorAnalyzerConfig) -> None:
        self._config = config

    def analyze(self, crop_rgb: np.ndarray, crop_mask: np.ndarray | None = None) -> ColorPrediction:
        if crop_rgb.size == 0:
            return ColorPrediction(
                color_label="indefinido",
                life_stage="indefinido",
                confidence=0.0,
                dominant_rgb=[0, 0, 0],
                color_source="bbox",
                warnings=[WARNING_SMALL_BIRD],
            )

        pixels, color_source = self._extract_color_pixels(crop_rgb, crop_mask)
        if pixels.size == 0:
            return ColorPrediction(
                color_label="indefinido",
                life_stage="indefinido",
                confidence=0.0,
                dominant_rgb=[0, 0, 0],
                color_source=color_source,
                warnings=[WARNING_SMALL_BIRD],
            )

        rng = np.random.default_rng(self._config.random_seed)
        sample = self._sample_pixels(pixels, rng)
        dominant_rgb = self._dominant_color(sample)

        warnings: list[str] = []
        if color_source != "mask":
            # Sem mascara confiavel o recorte inclui fundo e a saturacao nao representa a ave.
            warnings.append(WARNING_NO_MASK)
        if pixels.shape[0] < self._config.min_pixels_analysis:
            warnings.append(WARNING_SMALL_BIRD)

        median_saturation = self._median_saturation(sample)
        color_label, life_stage, confidence, uncertain = self._classify_saturation(median_saturation)
        if uncertain:
            warnings.append(WARNING_UNCERTAIN_STAGE)

        if WARNING_NO_MASK in warnings or WARNING_SMALL_BIRD in warnings or uncertain:
            color_label, life_stage = "indefinido", "indefinido"

        return ColorPrediction(
            color_label=color_label,
            life_stage=life_stage,
            confidence=confidence,
            dominant_rgb=dominant_rgb,
            color_source=color_source,
            warnings=warnings,
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

    def _sample_pixels(self, pixels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sample_size = self._config.resized_width * self._config.resized_height
        if pixels.shape[0] > sample_size:
            indices = rng.choice(pixels.shape[0], sample_size, replace=False)
            return pixels[indices]
        return pixels

    def _dominant_color(self, pixels: np.ndarray) -> list[int]:
        if pixels.shape[0] == 0:
            return [0, 0, 0]

        n_clusters = min(self._config.kmeans_clusters, max(1, pixels.shape[0]))

        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=self._config.random_seed,
        )
        labels = kmeans.fit_predict(pixels)

        counts = np.bincount(labels)
        dominant_index = int(np.argmax(counts))
        center = kmeans.cluster_centers_[dominant_index]
        center = np.clip(center, 0, 255).astype(np.uint8)

        return [int(center[0]), int(center[1]), int(center[2])]

    @staticmethod
    def _median_saturation(pixels: np.ndarray) -> float:
        rgb = np.clip(pixels, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        return float(np.median(hsv[:, 1]))

    def _classify_saturation(self, median_saturation: float) -> tuple[str, str, float, bool]:
        """Retorna (cor, fase, confianca, incerto).

        Confianca e um escore heuristico (nao uma probabilidade): 0,5 no limiar e
        cresce linearmente com a distancia a ele, saturando em 1,0 (a 60 de distancia).
        """
        threshold = self._config.saturation_threshold
        margin = self._config.saturation_uncertainty_margin
        distance = abs(median_saturation - threshold)
        confidence = float(min(1.0, 0.5 + distance / 120.0))
        uncertain = distance < margin

        if median_saturation < threshold:
            return "cinza", "filhote", confidence, uncertain
        return "vermelho", "adulto", confidence, uncertain
