# Crazyflie 2.1 Brushless — approach and hover

A vision policy that flies a simulated Crazyflie up to a bright red target and
holds a point in front of it. Observations are small grayscale images, in the
spirit of the AI-deck camera, plus an optional height and velocity reading.
There is a state-based baseline that sees the target position directly, so the
training loop can be checked before the camera is involved.

The simulator is [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones)
2.2.0 (Gymnasium, PyBullet, stable-baselines3 PPO). Actions are hover
setpoints, the same shape `cflib` sends over the Crazyradio. Nothing in this
directory imports `cflib`, and the flight scripts at the repo root are not
used here.

## Why this simulator

gym-pybullet-drones is the default. Its README lists Apple Silicon / macOS 26
as a tested platform, it already trains PPO, and `BaseAviary` renders a
forward camera. [crazyflow](https://github.com/learnsyslab/crazyflow) has an
identified `cf21B_500` brushless model, which this URDF does not. It was not
used because:

* the GPU extra is Linux x86-64 CUDA only (Apple Silicon would be CPU JAX)
* the Gymnasium envs take attitude or force/torque actions, not a hover setpoint
* there is no camera observation to hang a CNN on

The brushless mass and thrust are applied on top of the library's Crazyflie
2.x model. Swapping in crazyflow's dynamics later is listed in `ROADMAP.md`.

## Vehicle parameters

| | This project (default) | gym-pybullet-drones `cf2x.urdf` |
| --- | --- | --- |
| Vehicle | Crazyflie 2.1 Brushless | Crazyflie 2.x, X configuration |
| Mass | 35 g (34 g with legs, 37 g with guards) | 27 g |
| Max thrust | 120 g (4 × 30 g) | thrust-to-weight 2.25 → about 61 g |
| Thrust / weight | 3.43 | 2.25 |
| Arm | **still the URDF's 39.7 mm** | 39.7 mm |
| Brushless arm | 50 mm (100 mm motor diagonal) | — |
| Props | **still the URDF radius, 23 mm** | 23 mm |

Bitcraze's datasheet is the source for 34 g / 37 g and 120 g. The inner loop
is the library's DSL PID, not the Crazyflie firmware. Motor thrust is matched
by scaling the URDF `kf` and `km` so the same maximum RPM produces 120 g
instead of ~61 g. Inertia is the Crazyflie 2.x tensor scaled by the mass
ratio. On startup the library prints the URDF numbers (`[INFO] BaseAviary`);
the next line (`[cf_rl]`) is the override the episode actually uses.

Pass `--mass-kg 0.034` or `--mass-kg 0.037` to train the datasheet endpoints.
`DroneConfig` refuses a max thrust that cannot hover the given mass.

## Apple Silicon

Target machine: Apple Silicon Mac, CPU. No CUDA.

* Python 3.12. The simulator's package metadata requires it.
* PyBullet has no prebuilt wheel after Python 3.10. On macOS, build it with
  `CFLAGS="-Dfdopen=fdopen"` (see the install block below). That flag is from
  the gym-pybullet-drones install instructions.
* `pip install torch==2.13.0` on macOS is the CPU/MPS wheel. Do not use a CUDA
  index. The Linux wheel is large because PyPI serves a CUDA build; that is
  the default Linux artifact, not a choice this project makes.
* Train with `--device cpu`. stable-baselines3 2.9 does not support MPS. The
  MPS backend is an unmerged draft
  ([PR 2203](https://github.com/DLR-RM/stable-baselines3/pull/2203)), and MPS
  rejects float64. Observations here are float32 or uint8 so a future SB3
  build has a chance. `--device mps` is accepted and will probably error.

These instructions were checked against gym-pybullet-drones `main` at
`f83fbdd` (2026-10-25) and against the packages pinned in `requirements.txt`.
The smoke test and the short training run in the pull request were executed
on Linux x86_64 CPU, not on a Mac. A Mac build of PyBullet was not run.

## Setup

```bash
cd rl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# macOS only, before the requirements install:
# CFLAGS="-Dfdopen=fdopen" pip install pybullet==3.2.7 --no-cache-dir

pip install -r requirements.txt
pytest
```

## Train

State baseline (target position is in the observation). A reasonable first run:

```bash
python train.py --env state --timesteps 100000
```

Vision, 64×64 grayscale, 4 stacked frames, `CnnPolicy` + `SmallCNN`:

```bash
python train.py --env vision --timesteps 200000 --n-stack 4
```

Height and heading-frame velocity as well (no target coordinates). This uses
`MultiInputPolicy` instead of the small CNN:

```bash
python train.py --env vision --timesteps 200000 --proprio
```

Checkpoints land in `rl/outputs/<env>-<timestamp>/` (`model.zip` and
`config.json`). That directory is gitignored.

Both tasks randomize the target and the start pose. The reward is the
negative distance to a point 0.45 m in front of the target (toward −X), a
small penalty on lateral velocity, yaw rate, and tilt, and a bonus while the
drone holds inside 0.15 m at low speed. Leaving the arena, hitting the floor
or the ceiling, tipping past about 57°, or touching the target ends the
episode. Episodes last about 8 s at 30 Hz.

These timestep counts are starting points, not a claim that the policy has
learned the task. Look at `config.json`'s `episode_rewards` and at the
evaluation GIF.

## Evaluate

```bash
python evaluate.py --model outputs/state-YYYYMMDD-HHMMSS/model.zip --episodes 3
```

Writes `overview_*.gif` (third person) next to the model, and for a vision
model also `onboard_*.gif` (the grayscale frame, before stacking). The JSON
beside the model supplies mass, stack length, and whether proprioception was
on, so the rollout matches training.

## What the policy commands

One action, four floats, already clipped to the training limits:

| | Unit | Limit (default) | cflib |
| --- | --- | --- | --- |
| `vx` | m/s, heading frame | ±0.40 | `send_hover_setpoint` vx |
| `vy` | m/s, heading frame | ±0.40 | vy |
| `yaw_rate` | rad/s in the policy, **deg/s on the radio** | ±1.0 rad/s | yawrate |
| `z` | m, absolute height | 0.15 – 1.20 | zdistance |

Heading frame means world XY rotated by yaw only. It is the horizontal
body frame a level hover uses, and it matches `send_hover_setpoint`'s body
velocities closely enough to deploy. Roll and pitch are not part of the
rotation, so a pitched command does not point the velocity into the floor.

```python
from cf_rl.setpoints import HoverSetpoint

vx, vy, yaw_deg, z = HoverSetpoint.from_array(action).as_hover()
cf.commander.send_hover_setpoint(vx, vy, yaw_deg, z)
```

Send that every control step (this project steps at 30 Hz; the flight scripts
use 50 Hz). The firmware cuts the motors if setpoints stop arriving.

World-frame velocity, if you would rather call `send_velocity_world_setpoint`,
is `as_velocity_world(yaw_rad)`. The vertical rate in that packet is 0,
because height is the absolute `z` setpoint, not a climb rate.

The environment also slews the command (`HoverLimits`: 2 m/s² horizontal,
4 rad/s² yaw, 0.5 m/s of height-command change). Use `SlewLimiter` on the
radio side with the same numbers. A battery cutoff, an estimator height
ceiling, and a hand-coded fallback are not implemented; they are in
`ROADMAP.md`.

The DSL PID inside the sim turns the setpoint into motor RPM. The policy
never sees RPM, and RPM is not the action to replay on the vehicle. The
firmware's own position/velocity controller is what `send_hover_setpoint`
drives.

## Files

| File | Role |
| --- | --- |
| `train.py` | PPO training entry point |
| `evaluate.py` | Rollout, GIF, `eval.json` |
| `requirements.txt` | Pinned install, including the gym-pybullet-drones git commit |
| `pytest.ini` | `pytest` from this directory |
| `cf_rl/config.py` | `DroneConfig`, task and action limits, URDF comparison |
| `cf_rl/setpoints.py` | Hover setpoint, clip, slew, cflib argument order |
| `cf_rl/params.py` | Writes mass and scaled `kf`/`km` into the simulator and the PID |
| `cf_rl/base_env.py` | PyBullet arena, target, reward, PID wiring |
| `cf_rl/state_env.py` | Kinematic observation, target known |
| `cf_rl/vision_env.py` | 64×64 forward camera, optional height/velocity |
| `cf_rl/vision.py` | Luminance, gain, bias, noise, 3×3 blur |
| `cf_rl/policy.py` | `SmallCNN` used by `CnnPolicy` |
| `cf_rl/vecenv.py` | Vector env, frame stack, policy name |
| `cf_rl/gif.py` | GIF writer (Pillow) |
| `tests/test_units.py` | Setpoint math and thrust scale, no simulator |
| `tests/test_smoke.py` | A few env steps and a tiny PPO update |
| `ROADMAP.md` | What to do before this flies on the real aircraft |

## Tests

```bash
cd rl && pytest
```

`test_smoke.py` resets both envs, holds a height with a zero-velocity
setpoint, checks that a forward command moves the drone, checks that the
camera image is not blank, and runs 64 PPO steps on the state env and on the
vision env (4-frame stack, `CnnPolicy`). It does not check that the policy
improved.
