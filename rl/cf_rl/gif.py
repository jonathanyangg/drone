"""Write a GIF with Pillow. Pillow is already required by gym-pybullet-drones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("no frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
