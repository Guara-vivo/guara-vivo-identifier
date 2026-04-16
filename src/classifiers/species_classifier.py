import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import ModelConfig
from src.types import SpeciesPrediction


class SpeciesClassifier:
    def __init__(self, config: ModelConfig) -> None:
        self._labels = config.species_labels
        self._model = tf.keras.models.load_model(config.species_model_path)

    @property
    def guara_label(self) -> str:
        return "guara_vermelho"

    def classify(self, crop_rgb: np.ndarray) -> SpeciesPrediction:
        resized = cv2.resize(crop_rgb, (224, 224))
        model_input = np.expand_dims(resized, axis=0)
        model_input = preprocess_input(model_input)

        predictions = self._model.predict(model_input, verbose=0)[0]
        best_index = int(np.argmax(predictions))

        return SpeciesPrediction(
            label=self._labels[best_index],
            confidence=float(predictions[best_index]),
        )

    def guara_confidence(self, crop_rgb: np.ndarray) -> float:
        resized = cv2.resize(crop_rgb, (224, 224))
        model_input = np.expand_dims(resized, axis=0)
        model_input = preprocess_input(model_input)

        predictions = self._model.predict(model_input, verbose=0)[0]
        guara_index = self._labels.index(self.guara_label)
        return float(predictions[guara_index])
