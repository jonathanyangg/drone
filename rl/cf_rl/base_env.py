"""Shared approach-and-hover aviary.

Subclasses implement the observation. Actions, reward, the target, and the
brushless parameter override live here.

The inner loop is gym-pybullet-drones' DSL PID for the Crazyflie 2.x, fed a
velocity and a height. It is not the Bitcraze firmware controller. The
setpoint that comes out of ``_preprocessAction`` is what a later cflib script
should send.
"""

from __future__ import annotations

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from cf_rl.config import DroneConfig, TaskConfig
from cf_rl.params import apply_brushless
from cf_rl.setpoints import (
    HoverSetpoint,
    SlewLimiter,
    heading_to_world,
    wrap_angle,
)


_DRONE_RADIUS_M = 0.07


class ApproachAviary(BaseAviary):
    """One drone, a bright target, and a hover-setpoint action."""

    def __init__(
        self,
        drone: DroneConfig | None = None,
        task: TaskConfig | None = None,
        ctrl_freq: int = 30,
        pyb_freq: int = 240,
        randomize: bool = True,
        gui: bool = False,
        seed: int | None = None,
    ) -> None:
        self.drone = drone or DroneConfig()
        self.task = task or TaskConfig()
        self.randomize = randomize
        self.ctrl = None
        self.target_pos = np.zeros(3)
        self.standoff = np.zeros(3)
        self.last_command = np.zeros(4, dtype=np.float32)
        self._slew = SlewLimiter(self.task.limits)
        self._failed = False
        self._holding = False
        self._target_id = -1
        self._wall_ids: list[int] = []
        self._rng_fallback = np.random.default_rng(0 if seed is None else seed)
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            physics=Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
            vision_attributes=False,
        )
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        apply_brushless(self, self.drone)
        print("[cf_rl]", self.drone.summary())

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        # The PID finite-differences yaw. Seed it with the pose we just loaded
        # so the first step does not see a huge yaw-rate error.
        self.ctrl.last_rpy = np.array(self.rpy[0], dtype=float)
        return obs, info

    def _housekeeping(self) -> None:
        self._sample_episode()
        super()._housekeeping()
        self._build_arena()
        apply_brushless(self, self.drone)
        self.last_command = self._slew.reset(
            np.array([0.0, 0.0, 0.0, self.INIT_XYZS[0, 2]], dtype=np.float32)
        )
        self._failed = False
        self._holding = False
        self._on_episode_start()

    def _on_episode_start(self) -> None:
        return None

    def _rng(self) -> np.random.Generator:
        rng = getattr(self, "np_random", None)
        if rng is None:
            return self._rng_fallback
        return rng

    def _sample_episode(self) -> None:
        task = self.task
        if not self.randomize:
            self.target_pos = np.array([1.20, 0.0, 0.55])
            start = np.array([0.10, 0.0, 0.55])
            yaw = 0.0
        else:
            rng = self._rng()
            self.target_pos = np.array(
                [
                    rng.uniform(*task.target_x),
                    rng.uniform(*task.target_y),
                    rng.uniform(*task.target_z),
                ]
            )
            start = np.array(
                [
                    rng.uniform(*task.start_x),
                    rng.uniform(*task.start_y),
                    rng.uniform(*task.start_z),
                ]
            )
            yaw = float(rng.uniform(*task.start_yaw))
        self.standoff = self.target_pos + np.array([-task.standoff_m, 0.0, 0.0])
        self.INIT_XYZS = start.reshape(1, 3)
        self.INIT_RPYS = np.array([[0.0, 0.0, yaw]])

    def _build_arena(self) -> None:
        self._tint(int(self.PLANE_ID), (0.18, 0.18, 0.18, 1.0))
        self._wall_ids = []
        extent = self.task.arena_xy + 0.15
        wall_h = 1.2
        thickness = 0.05
        specs = (
            ([extent, 0.0, wall_h / 2], [thickness, extent, wall_h / 2]),
            ([-extent, 0.0, wall_h / 2], [thickness, extent, wall_h / 2]),
            ([0.0, extent, wall_h / 2], [extent, thickness, wall_h / 2]),
            ([0.0, -extent, wall_h / 2], [extent, thickness, wall_h / 2]),
        )
        for origin, half in specs:
            self._wall_ids.append(self._box(origin, half, (0.12, 0.12, 0.14, 1.0)))
        self._target_id = self._sphere(
            self.target_pos, self.task.target_radius_m, (1.0, 0.20, 0.12, 1.0)
        )

    def _box(self, origin, half_extents, rgba) -> int:
        collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.CLIENT
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=rgba,
            physicsClientId=self.CLIENT,
        )
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=origin,
            physicsClientId=self.CLIENT,
        )
        return int(body)

    def _sphere(self, origin, radius, rgba) -> int:
        collision = p.createCollisionShape(
            p.GEOM_SPHERE, radius=radius, physicsClientId=self.CLIENT
        )
        visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.CLIENT
        )
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=np.asarray(origin, dtype=float).tolist(),
            physicsClientId=self.CLIENT,
        )
        p.changeVisualShape(
            body, -1, specularColor=[0.6, 0.6, 0.6], physicsClientId=self.CLIENT
        )
        return int(body)

    def _tint(self, body: int, rgba) -> None:
        # textureUniqueId=-1 drops the plane's checkered texture, which
        # otherwise stays near white and washes out a grayscale target.
        p.changeVisualShape(
            body,
            -1,
            rgbaColor=list(rgba),
            textureUniqueId=-1,
            physicsClientId=self.CLIENT,
        )

    def _actionSpace(self):
        limits = self.task.limits
        low = np.array([-limits.vx, -limits.vy, -limits.yaw_rate, limits.z_min], dtype=np.float32)
        high = np.array([limits.vx, limits.vy, limits.yaw_rate, limits.z_max], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _preprocessAction(self, action):
        commanded = self._slew.apply(np.asarray(action, dtype=np.float32), self.CTRL_TIMESTEP)
        self.last_command = commanded
        setpoint = HoverSetpoint.from_array(commanded)
        state = self._getDroneStateVector(0)
        yaw = float(state[9])
        vx_w, vy_w = heading_to_world(setpoint.vx, setpoint.vy, yaw)
        target_pos = np.array([state[0], state[1], setpoint.z], dtype=float)
        target_vel = np.array([vx_w, vy_w, 0.0], dtype=float)
        target_yaw = wrap_angle(yaw + setpoint.yaw_rate * self.CTRL_TIMESTEP)
        rpm, _, _ = self.ctrl.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, target_yaw]),
            target_vel=target_vel,
            target_rpy_rates=np.array([0.0, 0.0, setpoint.yaw_rate]),
        )
        rpm = np.asarray(rpm, dtype=float).reshape(4)
        if not np.all(np.isfinite(rpm)):
            raise RuntimeError(f"PID returned non-finite RPM {rpm}")
        return rpm.reshape(1, 4)

    def _distance(self) -> float:
        return float(np.linalg.norm(self.pos[0] - self.standoff))

    def _is_failure(self) -> bool:
        pos = self.pos[0]
        roll, pitch = float(self.rpy[0, 0]), float(self.rpy[0, 1])
        task = self.task
        if pos[2] < task.floor_m or pos[2] > task.ceiling_m:
            return True
        if abs(pos[0]) > task.arena_xy or abs(pos[1]) > task.arena_xy:
            return True
        if abs(roll) > task.max_tilt_rad or abs(pitch) > task.max_tilt_rad:
            return True
        if np.linalg.norm(pos - self.target_pos) < task.target_radius_m + _DRONE_RADIUS_M:
            return True
        return False

    def _is_holding(self) -> bool:
        task = self.task
        roll, pitch = float(self.rpy[0, 0]), float(self.rpy[0, 1])
        return (
            self._distance() < task.hold_radius_m
            and float(np.linalg.norm(self.vel[0])) < task.hold_speed_m_s
            and abs(roll) < 0.35
            and abs(pitch) < 0.35
        )

    def _computeReward(self) -> float:
        self._failed = self._is_failure()
        self._holding = self._is_holding()
        task = self.task
        roll, pitch = float(self.rpy[0, 0]), float(self.rpy[0, 1])
        cmd = self.last_command
        # z is an absolute height, not an effort. Penalize the rate commands.
        effort = float(cmd[0] ** 2 + cmd[1] ** 2 + cmd[2] ** 2)
        reward = -self._distance()
        reward -= task.action_penalty * effort
        reward -= task.tilt_penalty * (roll**2 + pitch**2)
        if self._holding:
            reward += task.hold_bonus
        if self._failed:
            reward -= task.crash_penalty
        return float(reward)

    def _computeTerminated(self) -> bool:
        return bool(self._failed)

    def _computeTruncated(self) -> bool:
        if self._failed:
            return False
        return (self.step_counter / self.PYB_FREQ) >= self.task.episode_seconds

    def _computeInfo(self) -> dict:
        cmd = HoverSetpoint.from_array(self.last_command)
        vx, vy, yaw_deg, z = cmd.as_hover()
        return {
            "distance": self._distance(),
            "holding": bool(self._holding),
            "crashed": bool(self._failed),
            "hover_vx": vx,
            "hover_vy": vy,
            "hover_yaw_deg_s": yaw_deg,
            "hover_z": z,
        }

    def grab_forward_rgba(self) -> np.ndarray:
        """Upright RGBA from a camera sitting just above the airframe, looking along +X body."""
        rot = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape(3, 3)
        eye = np.asarray(self.pos[0], dtype=float) + rot @ np.array([0.02, 0.0, 0.05])
        forward = rot @ np.array([1.0, 0.0, 0.0])
        up = rot @ np.array([0.0, 0.0, 1.0])
        view = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=(eye + forward).tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self.CLIENT,
        )
        size = int(self.task.image_size)
        projection = p.computeProjectionMatrixFOV(
            fov=self.task.camera_fov_deg,
            aspect=1.0,
            nearVal=0.02,
            farVal=8.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=size,
            height=size,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
            shadow=0,
            physicsClientId=self.CLIENT,
        )
        # This PyBullet build already returns row 0 at the top of the image.
        # Flipping it puts the empty background under the target.
        return np.reshape(np.asarray(rgba, dtype=np.uint8), (size, size, 4))

    def capture_overview(self, width: int = 320, height: int = 240) -> np.ndarray:
        """Upright RGB of the arena, for GIFs. Not an observation."""
        focus = 0.5 * (np.asarray(self.pos[0]) + self.target_pos)
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=focus.tolist(),
            distance=2.6,
            yaw=35,
            pitch=-28,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.CLIENT,
        )
        projection = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=width / height, nearVal=0.05, farVal=20.0
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
            shadow=0,
            physicsClientId=self.CLIENT,
        )
        image = np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))
        return image[:, :, :3]

    def render(self):
        return self.capture_overview()
