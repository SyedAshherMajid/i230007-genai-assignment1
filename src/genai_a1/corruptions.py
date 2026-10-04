"""Shared, replayable 128px corruption definitions for tasks 1-3."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import correlate1d

LABELS = ("clean", "noise", "blur", "occlusion")
SEVERITIES = ("low", "medium", "high")
TEST_SETTINGS = {
    "noise": {"low": {"probability": 0.03}, "medium": {"probability": 0.08}, "high": {"probability": 0.15}},
    "blur": {"low": {"kernel": 3, "sigma": 0.7}, "medium": {"kernel": 5, "sigma": 1.5}, "high": {"kernel": 7, "sigma": 2.5}},
    "occlusion": {"low": {"coverage": 0.10, "rectangles": 1}, "medium": {"coverage": 0.20, "rectangles": 2}, "high": {"coverage": 0.35, "rectangles": 3}},
}


def _rectangles(shape: tuple[int, int], coverage: float, count: int, seed: int) -> tuple[list[list[int]], float]:
    height, width = shape
    rng = np.random.default_rng(seed)
    target = round(height * width * coverage)
    # Separate horizontal bands make the union exactly the sum of rectangle areas.
    bands = [(i * height // count, (i + 1) * height // count) for i in range(count)]
    rects: list[list[int]] = []
    total = 0
    for i, (top, bottom) in enumerate(bands):
        area = round(target * (i + 1) / count) - round(target * i / count)
        band_height = bottom - top
        min_width = max(1, math.ceil(area / band_height))
        max_width = min(width, area)
        rect_width = int(rng.integers(min_width, max_width + 1))
        rect_height = min(band_height, max(1, round(area / rect_width)))
        x0 = int(rng.integers(0, width - rect_width + 1))
        y0 = int(rng.integers(top, bottom - rect_height + 1))
        rects.append([x0, y0, x0 + rect_width, y0 + rect_height])
        total += rect_width * rect_height
    achieved = total / (height * width)
    if abs(achieved - coverage) > 0.01:
        raise AssertionError(f"Occlusion coverage {achieved:.4f} differs from target {coverage:.4f}")
    return rects, achieved


def recipe(label: str, seed: int, severity: str | None = None, size: int = 128) -> dict:
    if label not in LABELS:
        raise ValueError(label)
    if severity is not None and severity not in SEVERITIES:
        raise ValueError(severity)
    result: dict = {"label": label, "seed": int(seed), "severity": severity or "train"}
    if label == "clean":
        return result
    rng = np.random.default_rng(seed)
    if severity is None:
        if label == "noise":
            result["probability"] = float(rng.uniform(0.02, 0.15))
        elif label == "blur":
            result["kernel"] = int(rng.choice([3, 5, 7]))
            result["sigma"] = float(rng.uniform(0.5, 2.5))
        else:
            result["coverage"] = float(rng.uniform(0.10, 0.35))
            result["rectangles"] = int(rng.integers(1, 4))
    else:
        result.update(TEST_SETTINGS[label][severity])
    if label == "occlusion":
        rects, actual = _rectangles((size, size), result["coverage"], result["rectangles"], seed)
        result["boxes"] = rects
        result["actual_coverage"] = actual
    return result


def apply_corruption(rgb: np.ndarray, settings: dict) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 uint8 RGB image")
    kind = settings["label"]
    if kind == "clean":
        return rgb.copy()
    if kind == "noise":
        rng = np.random.default_rng(settings["seed"])
        take = rng.random(rgb.shape[:2]) < settings["probability"]
        values = rng.integers(0, 2, size=rgb.shape[:2], dtype=np.uint8) * 255
        output = rgb.copy()
        output[take] = values[take, None]
        return output
    if kind == "blur":
        kernel_size = int(settings["kernel"])
        sigma = float(settings["sigma"])
        coords = np.arange(kernel_size, dtype=np.float32) - kernel_size // 2
        weights = np.exp(-(coords * coords) / (2 * sigma * sigma))
        weights /= weights.sum()
        blurred = correlate1d(rgb.astype(np.float32), weights, axis=0, mode="reflect")
        blurred = correlate1d(blurred, weights, axis=1, mode="reflect")
        return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    if kind == "occlusion":
        output = rgb.copy()
        for x0, y0, x1, y1 in settings["boxes"]:
            output[y0:y1, x0:x1] = 0
        return output
    raise ValueError(kind)

