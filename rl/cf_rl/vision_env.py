"""Vision environment: 64x64 grayscale from the forward camera.

The target position is not in the observation. Frame stacking is applied
outside this class (see ``cf_rl.vecenv``) so a single step here is one frame.

``proprio=True`` adds height and heading-frame velocity. It does not add the
target position. Training then uses ``MultiInputPolicy`` instead of ``CnnPolicy``.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from cf_rl.base_env import ApproachAviary
from cf_rl.setpoints import world_vector_to_heading
from cf_rl.vision import FrameNoise, apply_noise, rgb_to_gray


class VisionApproachEnv(ApproachAviary):
    """Forward-camera observation of a bright red target."""

    def __init__(self, proprio: bool = False, **kwargs) -> None:
        self.proprio = proprio
        self.frame_noise = FrameNoise.fixed()
        self._last_gray: np.ndarray | None = None
        super().__init__(**kwargs)

    def _on_episode_start(self) -> None:
        if self.randomize:
            rng = self._rng()
            gray = float(rng.uniform(0.08, 0.22))
            green = float(rng.uniform(0.05, 0.28))
            blue = float(rng.uniform(0.05, 0.22))
            self.frame_noise = FrameNoise.sample(rng)
        else:
            gray, green, blue = 0.12, 0.20, 0.12
            self.frame_noise = FrameNoise.fixed()
        for wall in self._wall_ids:
            self._tint(wall, (gray, gray, gray + 0.02, 1.0))
        self._tint(int(self.PLANE_ID), (gray * 0.9, gray * 0.9, gray * 0.9, 1.0))
        self._tint(self._target_id, (1.0, green, blue, 1.0))

    def _observationSpace(self):
        size = int(self.task.image_size)
        image = spaces.Box(low=0, high=255, shape=(1, size, size), dtype=np.uint8)
        if not self.proprio:
            return image
        # height, heading-frame velocity. No target coordinates.
        proprio = spaces.Box(
            low=np.array([0.0, -5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )
        return spaces.Dict({"image": image, "proprio": proprio})

    def _computeObs(self):
        gray = self._gray_frame()
        self._last_gray = gray
        image = gray[None, ...]
        if not self.proprio:
            return image
        return {"image": image, "proprio": self._proprio()}

    def _gray_frame(self) -> np.ndarray:
        rgba = self.grab_forward_rgba()
        gray = rgb_to_gray(rgba)
        return apply_noise(gray, self.frame_noise, self._rng())

    def _proprio(self) -> np.ndarray:
        yaw = float(self.rpy[0, 2])
        velocity = world_vector_to_heading(self.vel[0], yaw)
        return np.array(
            [self.pos[0, 2], velocity[0], velocity[1], velocity[2]],
            dtype=np.float32,
        )

    def onboard_frame(self) -> np.ndarray:
        """Last grayscale observation, HxW uint8. Available after reset or step."""
        if self._last_gray is None:
            self._last_gray = self._gray_frame()
        return self._last_gray
