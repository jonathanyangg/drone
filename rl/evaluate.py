"""Roll out a saved policy and write a GIF of the flight.

The GIF is a third-person view. Vision runs also write the grayscale frames
the policy saw (one camera frame per control step, not the stacked tensor).

    python evaluate.py --model outputs/state-.../model.zip
    python evaluate.py --model outputs/vision-.../model.zip --episodes 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from cf_rl.config import DroneConfig
from cf_rl.gif import save_gif
from cf_rl.vecenv import make_env, raw_env


def rollout(model, vec, frame_skip: int = 2) -> dict:
    env = raw_env(vec)
    obs = vec.reset()
    overview: list[np.ndarray] = []
    onboard: list[np.ndarray] = []
    rewards: list[float] = []
    distances: list[float] = []
    final_info: dict = {}
    # A long episode is a few hundred steps. Cap so a bug cannot spin forever.
    for step in range(5000):
        if step % frame_skip == 0:
            overview.append(env.capture_overview())
            if hasattr(env, "onboard_frame"):
                onboard.append(np.array(env.onboard_frame(), copy=True))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = vec.step(action)
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        final_info = infos[0]
        if "distance" in final_info:
            distances.append(float(final_info["distance"]))
        if bool(np.asarray(done).reshape(-1)[0]):
            break
    return {
        "return": float(np.sum(rewards)),
        "length": len(rewards),
        "final_distance": None if not distances else distances[-1],
        "min_distance": None if not distances else float(np.min(distances)),
        "holding": bool(final_info.get("holding", False)),
        "crashed": bool(final_info.get("crashed", False)),
        "overview": overview,
        "onboard": onboard,
    }


def evaluate(
    model_path: Path,
    episodes: int = 1,
    seed: int = 0,
    device: str = "cpu",
    output: Path | None = None,
    frame_skip: int = 2,
) -> dict:
    model_path = Path(model_path)
    config_path = model_path.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    kind = config.get("env", "state")
    n_stack = int(config.get("n_stack", 4))
    proprio = bool(config.get("proprio", False))
    ctrl_freq = int(config.get("ctrl_freq", 30))
    drone = DroneConfig(
        mass_kg=float(config.get("mass_kg", 0.035)),
        max_thrust_g=float(config.get("max_thrust_g", 120.0)),
    )
    out_dir = Path(output) if output is not None else model_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(model_path), device=device)
    results = []
    fps = max(1, ctrl_freq // frame_skip)
    for episode in range(episodes):
        vec = make_env(
            kind,
            seed=seed + episode,
            n_stack=n_stack,
            drone=drone,
            proprio=proprio,
            ctrl_freq=ctrl_freq,
            randomize=True,
        )
        try:
            stats = rollout(model, vec, frame_skip=frame_skip)
        finally:
            vec.close()
        overview_path = out_dir / f"overview_{episode}.gif"
        save_gif(stats["overview"], overview_path, fps=fps)
        onboard_path = None
        if stats["onboard"]:
            onboard_path = out_dir / f"onboard_{episode}.gif"
            save_gif(stats["onboard"], onboard_path, fps=fps)
        record = {
            "return": stats["return"],
            "length": stats["length"],
            "final_distance": stats["final_distance"],
            "min_distance": stats["min_distance"],
            "holding": stats["holding"],
            "crashed": stats["crashed"],
            "overview_gif": str(overview_path),
            "onboard_gif": None if onboard_path is None else str(onboard_path),
        }
        results.append(record)
        print(
            f"[cf_rl] episode {episode}  return {record['return']:.2f}  "
            f"len {record['length']}  min distance {record['min_distance']}  "
            f"crashed {record['crashed']}  gif {overview_path}"
        )
    summary = {"episodes": results}
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved approach-and-hover policy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frame-skip", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        output=args.output,
        frame_skip=args.frame_skip,
    )


if __name__ == "__main__":
    main()
