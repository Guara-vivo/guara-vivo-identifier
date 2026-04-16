from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    yolo_model_path: str = "yolo11x-seg.pt"
    species_model_path: str = "modelo_guara_mobilenetv2.keras"
    species_labels: tuple[str, ...] = ("colhereiro", "garca", "guara_vermelho")
    target_detection_class: str = "bird"


@dataclass(frozen=True)
class ColorAnalyzerConfig:
    kmeans_clusters: int = 3
    resized_width: int = 50
    resized_height: int = 50
    use_mask_for_color: bool = True
    min_mask_coverage: float = 0.08
    mask_erosion_kernel_size: int = 3
    min_mask_pixels: int = 100
