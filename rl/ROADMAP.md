# Next steps

The sim loop is in place. None of this has been on the aircraft.

## AI-deck frames

Stream the AI-deck camera to the laptop and put a real frame next to
`evaluate.py`'s `onboard_*.gif`. The sim image is 64×64, 90° field of view,
Rec. 601 grayscale, with gain, bias, Gaussian noise, and an optional 3×3
blur. The deck is a different sensor (Himax, its own resolution and lens).
Expect the blob size, the vignetting, and the black level to disagree.
Adjust `TaskConfig.image_size`, `camera_fov_deg`, and `FrameNoise` only
after that comparison, not before.

## Flow deck

A Flow deck gives body-frame velocity and a height, which is the `--proprio`
observation and the `vx, vy, z` part of the hover setpoint. Deployment sketch,
once the deck is estimating:

1. Load the policy on the laptop.
2. Read `stateEstimate` velocity and height (and yaw, to build the heading frame).
3. For a vision policy, replace the sim frame with the AI-deck frame, stacked the same way (`n_stack` in `config.json`).
4. Clip and slew with `HoverLimits` / `SlewLimiter` at the same 30 Hz the policy trained at, or retrain at the rate you will actually send.
5. `cf.commander.send_hover_setpoint(*setpoint.as_hover())` every tick.
6. Keep the existing `hover.py` / `fly_keyboard.py` ceiling check. Those scripts already stop if the height estimate runs away.

Do not replay motor RPM from the sim. The firmware controller is the inner loop.

## Safety

Already in the sim, and should be duplicated on the radio link:

* clip `vx`, `vy` to ±0.4 m/s, yaw rate to ±1 rad/s, height to 0.15–1.20 m
* slew limits in `HoverLimits` (2 m/s², 4 rad/s², 0.5 m/s of height command)

Still to write in the flight script, not in this package:

* height ceiling on the estimator, same idea as `CEILING` in `hover.py`
* battery cutoff from `pm.vbat`, and refuse to arm below a full-cell threshold
* a dead-man timeout: if the policy or the camera stalls, send a zero-velocity hover and then `send_stop_setpoint`
* a hand-coded fallback (the current keyboard / hover scripts) that can take the radio back without restarting the Crazyflie
* geofence in the horizontal plane, tighter than the sim arena, with a stop rather than a learned recovery

## Dynamics

Mass and total thrust match the brushless datasheet. The mesh, arm (39.7 mm
vs 50 mm), propeller, inertia shape, and PID are still the Crazyflie 2.x
model in gym-pybullet-drones. crazyflow's `cf21B_500` is the identified
brushless model and the right source when the setpoint interface here needs
a better plant. Drag, motor delay, and the firmware's actual velocity
controller are not modeled.
