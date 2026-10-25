"""Setpoint conversion, thrust override, and image noise. No simulator."""

import math

import numpy as np
import pytest

from cf_rl.config import LIBRARY_CF2X, DroneConfig, HoverLimits
from cf_rl.setpoints import (
    HoverSetpoint,
    SlewLimiter,
    clip_hover_action,
    heading_to_world,
    world_to_heading,
)
from cf_rl.vision import FrameNoise, apply_noise, blur3, rgb_to_gray


def test_brushless_override_against_library_urdf():
    drone = DroneConfig()
    assert drone.mass_kg == pytest.approx(0.035)
    assert drone.max_thrust_g == pytest.approx(120.0)
    # cf2x.urdf: thrust2weight 2.25, mass 0.027 kg -> about 61 g, not 120 g.
    assert LIBRARY_CF2X.max_thrust_g == pytest.approx(2.25 * 27.0)
    assert drone.kf_scale == pytest.approx(120.0 / LIBRARY_CF2X.max_thrust_g)
    assert drone.kf_scale > 1.5
    assert drone.thrust_to_weight == pytest.approx(0.120 / 0.035)
    assert drone.with_legs().mass_kg == pytest.approx(0.034)
    assert drone.with_guards().mass_kg == pytest.approx(0.037)


def test_thrust_must_exceed_weight():
    with pytest.raises(ValueError):
        DroneConfig(mass_kg=0.035, max_thrust_g=30.0)


def test_hover_setpoint_matches_cflib_units():
    setpoint = HoverSetpoint(vx=0.2, vy=-0.1, yaw_rate=0.5, z=0.4)
    vx, vy, yaw_deg, z = setpoint.as_hover()
    assert (vx, vy, z) == (0.2, -0.1, 0.4)
    assert yaw_deg == pytest.approx(math.degrees(0.5))

    yaw = math.pi / 2
    vx_w, vy_w, vz, yaw_deg_w = setpoint.as_velocity_world(yaw)
    # Heading +Y when yaw is 90 deg: body +x is world +y, body +y is world -x.
    assert vx_w == pytest.approx(0.1)
    assert vy_w == pytest.approx(0.2)
    assert vz == 0.0
    assert yaw_deg_w == pytest.approx(yaw_deg)


def test_heading_rotation_roundtrip():
    yaw = 0.4
    back_x, back_y = world_to_heading(*heading_to_world(0.3, -0.2, yaw), yaw)
    assert back_x == pytest.approx(0.3)
    assert back_y == pytest.approx(-0.2)


def test_clip_and_slew():
    limits = HoverLimits(vx=0.4, vy=0.4, yaw_rate=1.0, z_min=0.15, z_max=1.2, max_dv=2.0, max_dz=0.5)
    clipped = clip_hover_action(np.array([5.0, -5.0, 3.0, 0.0]), limits)
    assert clipped.tolist() == pytest.approx([0.4, -0.4, 1.0, 0.15])
    limiter = SlewLimiter(limits)
    # z=0 is clipped up to z_min before the slew state is stored.
    assert limiter.reset(np.zeros(4))[3] == pytest.approx(limits.z_min)
    stepped = limiter.apply(np.array([0.4, 0.0, 0.0, 0.5]), dt=0.1)
    # 2 m/s^2 * 0.1 s = 0.2 m/s, so vx cannot jump straight to 0.4.
    assert stepped[0] == pytest.approx(0.2)
    # Height command may rise by 0.5 m/s * 0.1 s from z_min.
    assert stepped[3] == pytest.approx(limits.z_min + 0.05)


def test_grayscale_and_noise_stay_in_range():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    gray = rgb_to_gray(rgb)
    assert gray[0, 0] == pytest.approx(0.299 * 255)
    noisy = apply_noise(gray, FrameNoise(gain=2.0, bias=10.0, sigma=30.0, blur=True), np.random.default_rng(0))
    assert noisy.dtype == np.uint8
    assert noisy.min() >= 0 and noisy.max() <= 255
    flat = np.full((5, 5), 40.0, dtype=np.float32)
    assert np.allclose(blur3(flat), 40.0)
