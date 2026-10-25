"""A few simulated steps and a tiny PPO update. Needs PyBullet."""

import numpy as np
import pybullet as p
import pytest

from cf_rl.state_env import STATE_OBS_NAMES, StateApproachEnv
from cf_rl.vecenv import make_env, raw_env
from cf_rl.vision_env import VisionApproachEnv
from train import run


@pytest.fixture
def state_env():
    env = StateApproachEnv(randomize=False, ctrl_freq=30)
    try:
        yield env
    finally:
        env.close()


def test_brushless_mass_is_applied(state_env):
    state_env.reset(seed=0)
    assert state_env.M == pytest.approx(0.035)
    assert state_env.MAX_THRUST == pytest.approx(0.120 * 9.8, rel=1e-6)
    body = int(state_env.DRONE_IDS[0])
    mass = p.getDynamicsInfo(body, -1, physicsClientId=state_env.CLIENT)[0]
    assert mass == pytest.approx(0.035)
    # Controller weight matches the body, not the 27 g URDF.
    assert state_env.ctrl.GRAVITY == pytest.approx(0.035 * 9.8)
    assert state_env.ctrl.KF == pytest.approx(state_env.KF)


def test_state_hover_setpoint_holds_height(state_env):
    obs, info = state_env.reset(seed=1)
    assert obs.shape == (len(STATE_OBS_NAMES),)
    assert obs.dtype == np.float32
    z0 = float(state_env.pos[0, 2])
    action = np.array([0.0, 0.0, 0.0, z0], dtype=np.float32)
    for _ in range(45):
        obs, reward, terminated, truncated, info = state_env.step(action)
        assert np.isfinite(reward)
        assert not terminated, f"hover command crashed at z={state_env.pos[0, 2]:.3f}"
    assert abs(float(state_env.pos[0, 2]) - z0) < 0.25
    assert abs(float(state_env.pos[0, 0]) - 0.10) < 0.45
    vx, vy, yaw_deg, z = (
        info["hover_vx"],
        info["hover_vy"],
        info["hover_yaw_deg_s"],
        info["hover_z"],
    )
    assert yaw_deg == pytest.approx(0.0)
    assert z == pytest.approx(z0)
    assert abs(vx) < 1e-5 and abs(vy) < 1e-5


def test_forward_velocity_moves_the_drone(state_env):
    state_env.reset(seed=2)
    z0 = float(state_env.pos[0, 2])
    x0 = float(state_env.pos[0, 0])
    action = np.array([0.35, 0.0, 0.0, z0], dtype=np.float32)
    terminated = False
    for _ in range(60):
        _, _, terminated, _, _ = state_env.step(action)
        if terminated:
            break
    assert not terminated
    assert float(state_env.pos[0, 0]) > x0 + 0.08


def test_vision_frame_is_a_real_image():
    env = VisionApproachEnv(randomize=False, ctrl_freq=30)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (1, 64, 64)
        assert obs.dtype == np.uint8
        assert int(obs.max()) - int(obs.min()) > 10
        action = np.array([0.0, 0.0, 0.0, float(env.pos[0, 2])], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (1, 64, 64)
        assert np.isfinite(reward)
        assert "distance" in info
        proprio_env = VisionApproachEnv(proprio=True, randomize=False, ctrl_freq=30)
        try:
            proprio_obs, _ = proprio_env.reset(seed=0)
            assert set(proprio_obs) == {"image", "proprio"}
            assert proprio_obs["image"].shape == (1, 64, 64)
            assert proprio_obs["proprio"].shape == (4,)
        finally:
            proprio_env.close()
    finally:
        env.close()


def test_vec_env_unwraps_to_the_simulator():
    vec = make_env("vision", seed=0, n_stack=2, randomize=False)
    try:
        env = raw_env(vec)
        vec.reset()
        frame = env.capture_overview()
        onboard = env.onboard_frame()
        assert frame.shape == (240, 320, 3)
        assert onboard.shape == (64, 64)
    finally:
        vec.close()


def test_tiny_state_ppo(tmp_path):
    summary = run(
        kind="state",
        timesteps=64,
        seed=0,
        device="cpu",
        output=tmp_path / "state",
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        verbose=0,
        randomize=False,
    )
    assert summary["model_path"].endswith("model.zip")
    assert (tmp_path / "state" / "model.zip").is_file()
    assert (tmp_path / "state" / "config.json").is_file()


def test_tiny_vision_ppo(tmp_path):
    summary = run(
        kind="vision",
        timesteps=64,
        seed=0,
        n_stack=4,
        device="cpu",
        output=tmp_path / "vision",
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        verbose=0,
        randomize=False,
    )
    assert (tmp_path / "vision" / "model.zip").is_file()
    assert summary["episodes"] >= 0
