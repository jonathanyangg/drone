"""State baseline: the standoff point is in the observation.

This is the loop to train first. It does not use the camera. The observation
is in the same heading frame as the hover setpoint, so a positive ``rel_x``
means the standoff is in front of the drone and a positive ``vx`` command
flies toward it when the vehicle is level.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from cf_rl.base_env import ApproachAviary
from cf_rl.setpoints import world_vector_to_heading


# rel (3), heading velocity (3), roll, pitch, height, previous command (4)
STATE_OBS_NAMES = (
    "rel_x",
    "rel_y",
    "rel_z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "z",
    "cmd_vx",
    "cmd_vy",
    "cmd_yaw_rate",
    "cmd_z",
)


class StateApproachEnv(ApproachAviary):
    """Kinematic observation, target position known."""

    def _observationSpace(self):
        low = np.array(
            [-5, -5, -5, -5, -5, -5, -np.pi, -np.pi, 0, -1, -1, -2, 0],
            dtype=np.float32,
        )
        high = np.array(
            [5, 5, 5, 5, 5, 5, np.pi, np.pi, 5, 1, 1, 2, 2],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self) -> np.ndarray:
        yaw = float(self.rpy[0, 2])
        relative = world_vector_to_heading(self.standoff - self.pos[0], yaw)
        velocity = world_vector_to_heading(self.vel[0], yaw)
        obs = np.array(
            [
                relative[0],
                relative[1],
                relative[2],
                velocity[0],
                velocity[1],
                velocity[2],
                self.rpy[0, 0],
                self.rpy[0, 1],
                self.pos[0, 2],
                self.last_command[0],
                self.last_command[1],
                self.last_command[2],
                self.last_command[3],
            ],
            dtype=np.float32,
        )
        return obs
