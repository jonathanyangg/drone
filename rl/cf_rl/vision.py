"""Grayscale conversion and the camera noise applied to each vision frame.

The AI-deck camera is monochrome. A bright red target is only a bright blob
if the room is darker than the target's luminance, so the arena is painted
dark and the target stays red-dominant but high-valued. Photometric gain,
bias, noise, and a 3x3 blur stand in for exposure and the low-quality sensor.
They are not a measured noise model of the Himax sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Rec. 601 luminance. A monochrome sensor does not get a free red channel.
_LUMA_R = 0.299
_LUMA_G = 0.587
_LUMA_B = 0.114


@dataclass
class FrameNoise:
    gain: float = 1.0
    bias: float = 0.0
    sigma: float = 0.0
    blur: bool = False

    @classmethod
    def fixed(cls) -> "FrameNoise":
        return cls()

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "FrameNoise":
        return cls(
            gain=float(rng.uniform(0.60, 1.30)),
            bias=float(rng.uniform(-15.0, 15.0)),
            sigma=float(rng.uniform(2.0, 8.0)),
            blur=bool(rng.random() < 0.5),
        )


def rgb_to_gray(rgba: np.ndarray) -> np.ndarray:
    """``(..., 3 or 4)`` uint8 RGB(A) to a float grayscale image in 0..255."""
    rgb = np.asarray(rgba, dtype=np.float32)
    return _LUMA_R * rgb[..., 0] + _LUMA_G * rgb[..., 1] + _LUMA_B * rgb[..., 2]


def blur3(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    total = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            total += padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
    return total / 9.0


def apply_noise(gray: np.ndarray, noise: FrameNoise, rng: np.random.Generator) -> np.ndarray:
    """Return a uint8 image. ``gray`` is float, roughly 0..255, and is not modified."""
    image = np.asarray(gray, dtype=np.float32) * noise.gain + noise.bias
    if noise.blur:
        image = blur3(image)
    if noise.sigma > 0:
        image = image + rng.normal(0.0, noise.sigma, size=image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)
