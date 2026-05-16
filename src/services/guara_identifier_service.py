import cv2
import numpy as np

from src.analyzers.color_analyzer import GuaraColorAnalyzer
from src.analyzers.distance_estimator import GuaraDistanceEstimator
from src.classifiers.species_classifier import SpeciesClassifier
from src.config import ColorAnalyzerConfig, DistanceEstimatorConfig, ModelConfig
from src.detectors.yolo_detector import YoloBirdDetector
from src.types import GuaraIdentification


class GuaraIdentifierService:
    def __init__(
        self,
        model_config: ModelConfig | None = None,
        color_config: ColorAnalyzerConfig | None = None,
        distance_config: DistanceEstimatorConfig | None = None,
    ) -> None:
        self._model_config = model_config or ModelConfig()
        self._color_config = color_config or ColorAnalyzerConfig()
        self._distance_config = distance_config or DistanceEstimatorConfig()

        self._detector = YoloBirdDetector(self._model_config)
        self._classifier = SpeciesClassifier(self._model_config)
        self._color_analyzer = GuaraColorAnalyzer(self._color_config)
        self._distance_estimator = GuaraDistanceEstimator(self._distance_config)

    def process_image(self, image_path: str) -> dict:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Nao foi possivel carregar a imagem: {image_path}")

        return self.process_image_bgr(image_bgr, image_path)

    def process_image_bgr(self, image_bgr: np.ndarray, image_source: str = "imagem_enviada") -> dict:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Nao foi possivel processar a imagem recebida")

        detections = self._detector.detect_birds(image_bgr)
        guara_results: list[GuaraIdentification] = []

        for detection in detections:
            species = self._classifier.classify(detection.crop_rgb)
            if species.label != self._classifier.guara_label:
                continue

            color = self._color_analyzer.analyze(detection.crop_rgb, detection.crop_mask)
            distance = self._distance_estimator.estimate(detection)
            guara_results.append(
                GuaraIdentification(
                    detection_id=len(guara_results) + 1,
                    bbox_xyxy=detection.bbox_xyxy,
                    color=color.color_label,
                    life_stage=color.life_stage,
                    color_source=color.color_source,
                    distance_m=distance.distance_m,
                    distance_uncertainty_m=distance.uncertainty_m,
                    distance_confidence=distance.confidence,
                    distance_method=distance.method,
                    distance_pixel_width=distance.pixel_width,
                    distance_pixel_height=distance.pixel_height,
                    distance_pixel_source=distance.pixel_source,
                    yolo_confidence=detection.yolo_confidence,
                    species_confidence=species.confidence,
                    color_confidence=color.confidence,
                    dominant_rgb=color.dominant_rgb,
                )
            )

        return {
            "imagem": image_source,
            "quantidade_guaras": len(guara_results),
            "guaras": [self._serialize_result(item) for item in guara_results],
        }

    @staticmethod
    def _serialize_result(item: GuaraIdentification) -> dict:
        return {
            "id": item.detection_id,
            "bbox_xyxy": item.bbox_xyxy,
            "cor": item.color,
            "fase_vida": item.life_stage,
            "fonte_analise_cor": item.color_source,
            "cor_dominante_rgb": item.dominant_rgb,
            "distancia_estimada_m": item.distance_m,
            "incerteza_distancia_m": item.distance_uncertainty_m,
            "metodo_distancia": item.distance_method,
            "fonte_pixels_distancia": item.distance_pixel_source,
            "medidas_pixels_objeto": {
                "largura_px": item.distance_pixel_width,
                "altura_px": item.distance_pixel_height,
            },
            "acuracia": {
                "deteccao_yolo": round(item.yolo_confidence, 4),
                "classificacao_guara": round(item.species_confidence, 4),
                "classificacao_cor": round(item.color_confidence, 4),
                "classificacao_fase_vida": round(item.color_confidence, 4),
                "estimativa_distancia": round(item.distance_confidence, 4),
            },
        }
