import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import ModelConfig
from src.types import SpeciesPrediction


class SpeciesClassifier:
    def __init__(self, config: ModelConfig) -> None:
        self._labels = config.species_labels
        # O modelo possui camada Lambda 'mobilenet_preprocess' que referencia
        # preprocess_input pelo nome; precisa estar em custom_objects.
        self._model = tf.keras.models.load_model(
            config.species_model_path,
            compile=False,
            custom_objects={"preprocess_input": preprocess_input},
        )

    @property
    def guara_label(self) -> str:
        return "guara_vermelho"

    def _predict_probs(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Redimensiona o recorte e retorna probabilidades preditas.

        O pré-processamento do MobileNetV2 já é feito dentro do modelo
        (camada 'mobilenet_preprocess'); a entrada é float32 em 0-255.
        """
        resized = cv2.resize(crop_rgb, (224, 224))
        model_input = np.expand_dims(resized, axis=0).astype(np.float32)
        return self._model.predict(model_input, verbose=0)[0]

    def classify(self, crop_rgb: np.ndarray) -> SpeciesPrediction:
        predictions = self._predict_probs(crop_rgb)
        best_index = int(np.argmax(predictions))

        return SpeciesPrediction(
            label=self._labels[best_index],
            confidence=float(predictions[best_index]),
        )

    def guara_confidence(self, crop_rgb: np.ndarray) -> float:
        predictions = self._predict_probs(crop_rgb)
        guara_index = self._labels.index(self.guara_label)
        return float(predictions[guara_index])
