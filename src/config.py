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
    random_seed: int = 42
    # Regra de fase de vida: saturacao HSV mediana (0-255) da mascara do YOLO.
    # Abaixo do limiar = filhote (cinza/marrom); acima = adulto (vermelho/rosado).
    saturation_threshold: float = 110.0
    # Faixa +-margem ao redor do limiar: sem sinal claro, fase = "indefinido".
    saturation_uncertainty_margin: float = 20.0
    # Menos pixels de mascara que isso: ave muito pequena, fase = "indefinido".
    min_pixels_analysis: int = 400


@dataclass(frozen=True)
class DistanceEstimatorConfig:
    focal_length_px: float = 1200.0
    adult_height_cm: float = 63.5
    adult_wingspan_cm_min: float = 52.0
    adult_wingspan_cm_max: float = 56.0
    wingspan_aspect_ratio_threshold: float = 1.2
    min_object_pixels: int = 20
    min_relative_uncertainty: float = 0.2
