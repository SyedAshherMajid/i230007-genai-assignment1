import numpy as np

from genai_a1.corruptions import LABELS, SEVERITIES, apply_corruption, recipe


def test_corruption_recipes_replay_and_preserve_targets():
    clean = np.random.default_rng(10).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    original = clean.copy()
    for label in LABELS:
        for severity in (SEVERITIES if label != "clean" else (None,)):
            settings = recipe(label, 934, severity)
            first = apply_corruption(clean, settings)
            second = apply_corruption(clean, settings)
            assert first.shape == clean.shape and first.dtype == np.uint8
            assert np.array_equal(first, second)
            assert np.array_equal(clean, original)


def test_occlusion_union_coverage_and_bounds():
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    for severity, expected in (("low", 0.10), ("medium", 0.20), ("high", 0.35)):
        for seed in range(30):
            settings = recipe("occlusion", seed, severity)
            output = apply_corruption(image, settings)
            actual = np.mean(np.all(output == 0, axis=2))
            assert abs(actual - expected) <= 0.01
            assert abs(actual - settings["actual_coverage"]) < 1e-12
            assert len(settings["boxes"]) == settings["rectangles"]
            for x0, y0, x1, y1 in settings["boxes"]:
                assert 0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 128


def test_salt_pepper_changes_entire_rgb_pixel():
    clean = np.full((128, 128, 3), 127, dtype=np.uint8)
    output = apply_corruption(clean, recipe("noise", 42, "high"))
    assert np.all((output == 127).all(axis=2) | (output == 0).all(axis=2) | (output == 255).all(axis=2))
    changed = np.mean(np.any(output != clean, axis=2))
    assert 0.13 <= changed <= 0.17


def test_blur_uses_kernel_size():
    impulse = np.zeros((128, 128, 3), dtype=np.uint8)
    impulse[64, 64] = 255
    narrow = apply_corruption(impulse, {"label": "blur", "kernel": 3, "sigma": 2.0})
    wide = apply_corruption(impulse, {"label": "blur", "kernel": 7, "sigma": 2.0})
    assert narrow[64, 67, 0] == 0
    assert wide[64, 67, 0] > 0
