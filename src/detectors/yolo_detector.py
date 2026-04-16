import cv2
import numpy as np
from ultralytics import YOLO

from src.config import ModelConfig
from src.types import BirdDetection


class YoloBirdDetector:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = YOLO(config.yolo_model_path)

    def detect_birds(self, image_bgr: np.ndarray) -> list[BirdDetection]:
        results = self._model(image_bgr, verbose=False)
        image_height, image_width = image_bgr.shape[:2]
        detections: list[BirdDetection] = []

        for result in results:
            masks_data = None
            if result.masks is not None and result.masks.data is not None:
                masks_data = result.masks.data

            for index, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = self._model.names[class_id]
                if class_name != self._config.target_detection_class:
                    continue

                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                x1 = max(0, min(x1, image_width - 1))
                y1 = max(0, min(y1, image_height - 1))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop_bgr = image_bgr[y1:y2, x1:x2]
                if crop_bgr.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                crop_mask = self._extract_crop_mask(
                    masks_data=masks_data,
                    index=index,
                    bbox=[x1, y1, x2, y2],
                    image_width=image_width,
                    image_height=image_height,
                )

                detections.append(
                    BirdDetection(
                        bbox_xyxy=[x1, y1, x2, y2],
                        yolo_confidence=float(box.conf[0]),
                        crop_rgb=crop_rgb,
                        crop_mask=crop_mask,
                    )
                )

        return detections

    @staticmethod
    def _extract_crop_mask(
        masks_data,
        index: int,
        bbox: list[int],
        image_width: int,
        image_height: int,
    ) -> np.ndarray | None:
        if masks_data is None or index >= len(masks_data):
            return None

        mask = masks_data[index].cpu().numpy()
        if mask.shape[0] != image_height or mask.shape[1] != image_width:
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        x1, y1, x2, y2 = bbox
        crop_mask = mask[y1:y2, x1:x2]
        if crop_mask.size == 0:
            return None

        binary_mask = (crop_mask > 0.5).astype(np.uint8)
        if np.count_nonzero(binary_mask) == 0:
            return None

        return binary_mask
