"""Hover setpoints that map onto cflib without a change of units.

The policy action is four numbers::

    vx, vy, yaw_rate, z

``vx`` and ``vy`` are metres per second in the heading frame: world XY
rotated by yaw, not by roll or pitch. ``yaw_rate`` is radians per second.
``z`` is an absolute height setpoint in metres.

That is the argument list of ``Crazyflie.commander.send_hover_setpoint``,
except the firmware wants yaw rate in degrees per second. ``HoverSetpoint.as_hover``
does that conversion::

    vx, vy, yaw_deg, z = setpoint.as_hover()
    cf.commander.send_hover_setpoint(vx, vy, yaw_deg, z)

Send it every control step. The firmware drops the motors if setpoints stop
(commander watchdog, on the order of a second).

``as_velocity_world`` is the other radio command. Height is still held by
``z``, so the vertical rate in that packet is zero. Yaw is again degrees::

    vx, vy, vz, yaw_deg = setpoint.as_velocity_world(yaw_rad)
    cf.commander.send_velocity_world_setpoint(vx, vy, vz, yaw_deg)

Clip and slew with the same ``HoverLimits`` the environment trains against
before the packet goes out. Battery cutoff, a height ceiling check against
the estimator, and a hand-coded fallback are not in this function; they
belong in the flight script. See ``ROADMAP.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cf_rl.config import HoverLimits


ACTION_NAMES = ("vx", "vy", "yaw_rate", "z")


@dataclass(frozen=True)
class HoverSetpoint:
    vx: float
    vy: float
    yaw_rate: float
    z: float

    @classmethod
    def from_array(cls, action: np.ndarray) -> "HoverSetpoint":
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.size != 4:
            raise ValueError(f"hover setpoint wants 4 values, got shape {np.shape(action)}")
        return cls(float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    def as_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.yaw_rate, self.z], dtype=np.float32)

    def as_hover(self) -> tuple[float, float, float, float]:
        """Arguments for ``cf.commander.send_hover_setpoint(vx, vy, yawrate, zdistance)``.

        ``vx``, ``vy`` are m/s in the heading frame. ``yawrate`` is deg/s.
        ``zdistance`` is metres.
        """
        return (self.vx, self.vy, math.degrees(self.yaw_rate), self.z)

    def as_velocity_world(self, yaw_rad: float) -> tuple[float, float, float, float]:
        """Arguments for ``cf.commander.send_velocity_world_setpoint``.

        Horizontal velocity is rotated into the world frame. ``vz`` is 0
        because this setpoint holds height with ``z`` rather than a climb rate.
        ``yawrate`` is deg/s.
        """
        vx_w, vy_w = heading_to_world(self.vx, self.vy, yaw_rad)
        return (vx_w, vy_w, 0.0, math.degrees(self.yaw_rate))


def clip_hover_action(action: np.ndarray, limits: HoverLimits) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(4)
    return np.array(
        [
            np.clip(values[0], -limits.vx, limits.vx),
            np.clip(values[1], -limits.vy, limits.vy),
            np.clip(values[2], -limits.yaw_rate, limits.yaw_rate),
            np.clip(values[3], limits.z_min, limits.z_max),
        ],
        dtype=np.float32,
    )


class SlewLimiter:
    """Limit how fast a hover command can change from one step to the next."""

    def __init__(self, limits: HoverLimits) -> None:
        self.limits = limits
        self._previous: np.ndarray | None = None

    def reset(self, action: np.ndarray) -> np.ndarray:
        self._previous = clip_hover_action(action, self.limits).copy()
        return self._previous.copy()

    def apply(self, action: np.ndarray, dt: float) -> np.ndarray:
        clipped = clip_hover_action(action, self.limits)
        if self._previous is None:
            return self.reset(clipped)
        max_step = np.array(
            [
                self.limits.max_dv * dt,
                self.limits.max_dv * dt,
                self.limits.max_dyaw * dt,
                self.limits.max_dz * dt,
            ],
            dtype=np.float32,
        )
        delta = np.clip(clipped - self._previous, -max_step, max_step)
        self._previous = (self._previous + delta).astype(np.float32)
        return self._previous.copy()


def heading_to_world(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    """Rotate a heading-frame horizontal velocity into the world frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx - s * vy, s * vx + c * vy


def world_to_heading(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    """Rotate a world-frame horizontal velocity into the heading frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx + s * vy, -s * vx + c * vy


def world_vector_to_heading(vector: np.ndarray, yaw: float) -> np.ndarray:
    x, y = world_to_heading(float(vector[0]), float(vector[1]), yaw)
    return np.array([x, y, float(vector[2])], dtype=np.float64)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
