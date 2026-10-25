"""Train a PPO policy on the state baseline or the vision approach task.

Run from this directory so ``cf_rl`` imports::

    python train.py --env state --timesteps 100000
    python train.py --env vision --timesteps 200000

The vision run stacks grayscale frames and uses ``CnnPolicy`` with
``cf_rl.policy.SmallCNN``. ``--proprio`` adds height and velocity and switches
to ``MultiInputPolicy``. Device defaults to CPU: stable-baselines3 2.9 does
not ship an MPS backend.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from cf_rl.config import DroneConfig
from cf_rl.vecenv import make_env, policy_name_and_kwargs


ROOT = Path(__file__).resolve().parent


class EpisodeLog(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            episode = info.get("episode")
            if episode is not None:
                self.episode_rewards.append(float(episode["r"]))
                self.episode_lengths.append(int(episode["l"]))
        return True


def run(
    kind: str = "state",
    timesteps: int = 100_000,
    seed: int = 0,
    n_stack: int = 4,
    device: str = "cpu",
    output: Path | None = None,
    n_steps: int = 512,
    batch_size: int = 128,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    proprio: bool = False,
    randomize: bool = True,
    mass_kg: float = 0.035,
    max_thrust_g: float = 120.0,
    ctrl_freq: int = 30,
    verbose: int = 1,
) -> dict:
    if device == "mps":
        print(
            "[cf_rl] MPS was requested. stable-baselines3 2.9 has no released MPS "
            "backend (draft PR #2203). This will likely fail; CPU is the supported device."
        )
    drone = DroneConfig(mass_kg=mass_kg, max_thrust_g=max_thrust_g)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output) if output is not None else ROOT / "outputs" / f"{kind}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    vec = make_env(
        kind,
        seed=seed,
        n_stack=n_stack,
        drone=drone,
        randomize=randomize,
        ctrl_freq=ctrl_freq,
        proprio=proprio,
    )
    policy, kwargs = policy_name_and_kwargs(kind, proprio=proprio)
    model = PPO(
        policy,
        vec,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        seed=seed,
        device=device,
        verbose=verbose,
        policy_kwargs=kwargs,
    )
    callback = EpisodeLog()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    model_path = run_dir / "model"
    model.save(str(model_path))
    config = {
        "env": kind,
        "timesteps": timesteps,
        "seed": seed,
        "n_stack": n_stack,
        "device": device,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "proprio": proprio,
        "randomize": randomize,
        "mass_kg": mass_kg,
        "max_thrust_g": max_thrust_g,
        "ctrl_freq": ctrl_freq,
        "policy": policy,
        "episode_rewards": callback.episode_rewards,
        "episode_lengths": callback.episode_lengths,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    vec.close()
    saved = run_dir / "model.zip"
    rewards = np.asarray(callback.episode_rewards, dtype=float)
    summary = {
        "model_path": str(saved),
        "config_path": str(run_dir / "config.json"),
        "episodes": int(rewards.size),
        "mean_episode_reward": None if rewards.size == 0 else float(rewards.mean()),
        "last_episode_reward": None if rewards.size == 0 else float(rewards[-1]),
    }
    print(
        f"[cf_rl] saved {saved}  episodes {summary['episodes']}  "
        f"mean reward {summary['mean_episode_reward']}"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Crazyflie approach-and-hover policy")
    parser.add_argument("--env", choices=("state", "vision"), default="state")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu, or mps if you want to try it")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--proprio", action="store_true", help="vision only: also observe height and velocity")
    parser.add_argument("--mass-kg", type=float, default=0.035)
    parser.add_argument("--max-thrust-g", type=float, default=120.0)
    parser.add_argument("--ctrl-freq", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        kind=args.env,
        timesteps=args.timesteps,
        seed=args.seed,
        n_stack=args.n_stack,
        device=args.device,
        output=args.output,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        proprio=args.proprio,
        mass_kg=args.mass_kg,
        max_thrust_g=args.max_thrust_g,
        ctrl_freq=args.ctrl_freq,
    )


if __name__ == "__main__":
    main()
