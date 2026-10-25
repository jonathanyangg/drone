"""Push a ``DroneConfig`` into a gym-pybullet-drones aviary and its PID.

The URDF is reloaded on every reset, which restores the Crazyflie 2.x mass.
Call ``apply_brushless`` from ``_housekeeping`` so each episode matches
``DroneConfig``. Motor coefficients are scaled once; scaling them again
would compound.
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from cf_rl.config import DroneConfig


def apply_brushless(env, drone: DroneConfig) -> None:
    """Set mass, inertia scale, and thrust coefficients on ``env``.

    ``env`` is a ``BaseAviary`` that has already loaded its URDF. The DSL
    PID, if ``env.ctrl`` exists, is pointed at the same mass and ``kf``
    the physics uses, then its integrators are cleared.
    """
    if not getattr(env, "_brushless_scaled", False):
        scale = drone.kf_scale
        env.KF = float(env.KF) * scale
        env.KM = float(env.KM) * scale
        env._brushless_scaled = True
        env._kf_scale = scale

    env.M = float(drone.mass_kg)
    env.GRAVITY = float(env.G) * env.M
    env.THRUST2WEIGHT_RATIO = drone.max_thrust_n / env.GRAVITY
    env.HOVER_RPM = float(np.sqrt(env.GRAVITY / (4.0 * env.KF)))
    env.MAX_RPM = float(np.sqrt((env.THRUST2WEIGHT_RATIO * env.GRAVITY) / (4.0 * env.KF)))
    env.MAX_THRUST = float(4.0 * env.KF * env.MAX_RPM**2)

    for body in np.atleast_1d(env.DRONE_IDS):
        _scale_body(int(body), drone.mass_kg, env.CLIENT)

    ctrl = getattr(env, "ctrl", None)
    if ctrl is not None:
        ctrl.KF = env.KF
        ctrl.KM = env.KM
        # BaseControl.GRAVITY is the weight in newtons (mass * g), despite the name.
        ctrl.GRAVITY = env.GRAVITY
        ctrl.reset()


def _scale_body(body: int, mass_kg: float, client: int) -> None:
    info = p.getDynamicsInfo(body, -1, physicsClientId=client)
    loaded_mass = float(info[0])
    if loaded_mass <= 0:
        raise RuntimeError(f"drone body {body} has mass {loaded_mass}")
    inertia = info[2]
    ratio = mass_kg / loaded_mass
    p.changeDynamics(
        body,
        -1,
        mass=mass_kg,
        localInertiaDiagonal=[float(v) * ratio for v in inertia],
        physicsClientId=client,
    )
