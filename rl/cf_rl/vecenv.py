"""Vector-env construction shared by training, evaluation, and the smoke test."""

from __future__ import annotations

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from cf_rl.config import DroneConfig, TaskConfig
from cf_rl.policy import SmallCNN
from cf_rl.state_env import StateApproachEnv
from cf_rl.vision_env import VisionApproachEnv


def make_env(
    kind: str,
    *,
    seed: int = 0,
    n_stack: int = 4,
    n_envs: int = 1,
    drone: DroneConfig | None = None,
    task: TaskConfig | None = None,
    randomize: bool = True,
    ctrl_freq: int = 30,
    proprio: bool = False,
) -> VecEnv:
    if kind not in ("state", "vision"):
        raise ValueError(f"env kind must be 'state' or 'vision', got {kind!r}")
    if kind == "state" and proprio:
        raise ValueError("proprio is only for the vision env; the state env already observes height and velocity")
    cls = StateApproachEnv if kind == "state" else VisionApproachEnv
    kwargs = dict(
        drone=drone or DroneConfig(),
        task=task or TaskConfig(),
        randomize=randomize,
        ctrl_freq=ctrl_freq,
    )
    if kind == "vision":
        kwargs["proprio"] = proprio
    vec = make_vec_env(cls, n_envs=n_envs, seed=seed, env_kwargs=kwargs)
    if kind == "vision":
        channels = {"image": "first", "proprio": "last"} if proprio else "first"
        vec = VecFrameStack(vec, n_stack=n_stack, channels_order=channels)
    return vec


def policy_name_and_kwargs(kind: str, proprio: bool = False) -> tuple[str, dict]:
    if kind == "state":
        return "MlpPolicy", {"net_arch": dict(pi=[64, 64], vf=[64, 64])}
    if proprio:
        # Combined extractor: NatureCNN on the image, a small MLP on proprio.
        return "MultiInputPolicy", {"net_arch": dict(pi=[64], vf=[64])}
    return "CnnPolicy", {
        "features_extractor_class": SmallCNN,
        "features_extractor_kwargs": {"features_dim": 64},
        "net_arch": dict(pi=[64], vf=[64]),
    }


def raw_env(vec: VecEnv):
    """The simulator under a DummyVecEnv, past frame-stack and Monitor."""
    current = vec
    while hasattr(current, "venv"):
        current = current.venv
    env = current.envs[0]
    return env.unwrapped if hasattr(env, "unwrapped") else env
