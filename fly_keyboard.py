"""Fly the Crazyflie with your keyboard over the Crazyradio. Height is held by the barometer.
No Flow deck: it drifts sideways and bobs ~10 cm, so keep correcting with the arrows.
Front is the antenna bump; the blue LEDs are the back. Start with the blue LEDs facing you.
Prop guards on, 2 m clear around it, USB unplugged. Q, Esc, closing the window or Ctrl+C cuts the motors.
Close cfclient (or click Disconnect) first: only one program can use the radio at a time.
A small window opens: click it so it gets your key presses."""
import time
import tkinter as tk
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = 'radio://0/80/2M/E7E7E7E7E7'
TILT, YAW_RATE = 8, 60          # degrees of tilt and degrees/s of spin while a key is held
TAKEOFF_H, STEP_H = 0.3, 0.05   # metres
MIN_H, MAX_H = 0.1, 1.0         # W/S stay within these heights
CEILING = MAX_H + 0.5           # cut the motors if the height estimate goes above this
REPEAT_S = 0.03                 # macOS key auto-repeat sends a release+press pair; ignore releases shorter than this
HELP = "T take off   W/S up/down   arrows move   A/D spin   SPACE land   Q/Esc quit (cuts motors)"

zs = []      # barometer height estimates, newest last
latest = {}  # newest value of every logged variable

def on_log(timestamp, data, logconf):
    zs.append(data['stateEstimate.z'])
    latest.update(data)

def land(cf, ground, start):  # descend at ~0.2 m/s to slightly below the floor, then cut the motors
    n = int((start + 0.1) / 0.2 * 50)
    for i in range(n):
        cf.commander.send_zdistance_setpoint(0, 0, 0, ground + start - (start + 0.1) * i / n)
        time.sleep(0.02)
    cf.commander.send_stop_setpoint()
    cf.supervisor.send_arming_request(False)

def fly(cf, ground):
    root = tk.Tk()
    root.title('Crazyflie keyboard')
    label = tk.Label(root, font=('Menlo', 14), justify='left', padx=20, pady=20)
    label.pack()
    state = {'target': None}  # height to hold, None = landed and disarmed
    down = {}                 # keysym -> time of its latest press, while held

    def release(key, pressed_at):
        if down.get(key) == pressed_at:  # no new press since, so it was a real release
            down.pop(key, None)

    def on_press(e):
        key = e.keysym.lower()
        down[key] = time.time()
        if key in ('q', 'escape'):
            root.destroy()
        elif key == 't' and state['target'] is None:
            cf.supervisor.send_arming_request(True)
            time.sleep(1)
            cf.commander.send_setpoint(0, 0, 0, 0)  # zero thrust first unlocks the motors
            state['target'] = TAKEOFF_H
        elif key == 'space' and state['target'] is not None:
            land(cf, ground, state['target'])
            state['target'] = None
        elif key in ('w', 's') and state['target'] is not None:
            step = STEP_H if key == 'w' else -STEP_H
            state['target'] = min(MAX_H, max(MIN_H, state['target'] + step))

    def on_release(e):
        key = e.keysym.lower()
        root.after(int(REPEAT_S * 1000), release, key, down.get(key))

    def tick():  # runs every 20 ms
        target, height = state['target'], zs[-1] - ground
        if height > CEILING:
            root.destroy()
            raise RuntimeError(f"height estimate {height:.2f} m is above the ceiling, cut the motors")
        if target is not None:
            pitch = TILT * (('up' in down) - ('down' in down))
            roll = TILT * (('right' in down) - ('left' in down))
            yaw = YAW_RATE * (('a' in down) - ('d' in down))  # positive = counter-clockwise (left) in this mode
            cf.commander.send_zdistance_setpoint(roll, pitch, yaw, ground + target)
        status = 'landed' if target is None else f'flying, holding {target:.2f} m'
        label.config(text=f"{HELP}\n\n{status}   height {height:+.2f} m   battery {latest['pm.vbat']:.2f} V")
        root.after(20, tick)

    root.bind('<KeyPress>', on_press)
    root.bind('<KeyRelease>', on_release)
    tick()
    root.mainloop()

cflib.crtp.init_drivers()
log = LogConfig(name='z', period_in_ms=20)
log.add_variable('stateEstimate.z', 'float')
log.add_variable('pm.vbat', 'float')
log.data_received_cb.add_callback(on_log)

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./.cache')) as scf:
    cf = scf.cf
    cf.log.add_config(log)
    log.start()
    time.sleep(1.5)
    ground = sum(zs) / len(zs)  # barometer reads height above sea level, so average the resting value
    try:
        fly(cf, ground)
    finally:  # always cut motors and disarm, even on Ctrl+C, closing the window or an error
        cf.commander.send_stop_setpoint()
        cf.supervisor.send_arming_request(False)
        time.sleep(0.1)
