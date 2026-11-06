import numpy as np

from src.analyzers.color_analyzer import (
    WARNING_NO_MASK,
    WARNING_SMALL_BIRD,
    WARNING_UNCERTAIN_STAGE,
    GuaraColorAnalyzer,
)
from src.config import ColorAnalyzerConfig


def _crop(rgb: tuple[int, int, int], size: int = 60) -> tuple[np.ndarray, np.ndarray]:
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    crop[:, :] = rgb
    mask = np.ones((size, size), dtype=np.uint8)
    return crop, mask


def _analyze(rgb: tuple[int, int, int], size: int = 60, mask: bool = True):
    crop, crop_mask = _crop(rgb, size)
    return GuaraColorAnalyzer(ColorAnalyzerConfig()).analyze(crop, crop_mask if mask else None)


def test_saturated_red_is_adult_without_warnings():
    result = _analyze((220, 30, 40))
    assert result.life_stage == "adulto"
    assert result.color_label == "vermelho"
    assert result.warnings == []


def test_gray_is_juvenile_without_warnings():
    result = _analyze((120, 115, 110))
    assert result.life_stage == "filhote"
    assert result.color_label == "cinza"
    assert result.warnings == []


def test_saturation_near_threshold_is_undefined_with_warning():
    # S ~ 110 em HSV (0-255): rosa palido, na faixa de incerteza
    result = _analyze((220, 124, 124))
    assert result.life_stage == "indefinido"
    assert WARNING_UNCERTAIN_STAGE in result.warnings


def test_tiny_bird_is_undefined_with_warning():
    result = _analyze((220, 30, 40), size=15)  # 225 px < min_pixels_analysis
    assert result.life_stage == "indefinido"
    assert WARNING_SMALL_BIRD in result.warnings


def test_without_mask_is_undefined_with_warning():
    result = _analyze((220, 30, 40), mask=False)
    assert result.life_stage == "indefinido"
    assert WARNING_NO_MASK in result.warnings


def test_empty_crop_is_undefined():
    result = GuaraColorAnalyzer(ColorAnalyzerConfig()).analyze(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result.life_stage == "indefinido"
    assert result.warnings


def test_result_is_deterministic_for_noisy_large_crop():
    rng = np.random.default_rng(1)
    crop = rng.integers(0, 256, size=(120, 120, 3), dtype=np.uint8)
    mask = np.ones((120, 120), dtype=np.uint8)
    analyzer = GuaraColorAnalyzer(ColorAnalyzerConfig())
    first = analyzer.analyze(crop, mask)
    for _ in range(3):
        again = analyzer.analyze(crop, mask)
        assert again == first
