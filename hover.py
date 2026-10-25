"""Hover over the Crazyradio: take off to ~30 cm, hold 2 s, land.
Height comes from the barometer only (no Flow deck): expect ~10 cm bobbing and sideways drift.
Prop guards on, 2 m clear around it, USB unplugged. Ctrl+C cuts the motors.
Close cfclient (or click Disconnect) first: only one program can use the radio at a time."""
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = 'radio://0/80/2M/E7E7E7E7E7'
HEIGHT, HOVER_S = 0.3, 2.0  # metres above the take-off spot, seconds
CEILING = HEIGHT + 0.5      # cut the motors if the height estimate goes above this

zs = []  # barometer height estimates, newest last

def run(cf, ground, start, end, seconds):  # move target height linearly from start to end, at 50 Hz
    n = int(seconds * 50)
    for i in range(n):
        if zs[-1] - ground > CEILING:
            raise RuntimeError(f"height estimate {zs[-1] - ground:.2f} m is above the ceiling, cutting motors")
        cf.commander.send_zdistance_setpoint(0, 0, 0, ground + start + (end - start) * i / n)  # level, firmware holds height
        time.sleep(0.02)

input(f"Hover at {HEIGHT} m for {HOVER_S} s. Guards on, area clear, USB unplugged? Enter to fly, Ctrl+C to quit. ")
cflib.crtp.init_drivers()
log = LogConfig(name='z', period_in_ms=20)
log.add_variable('stateEstimate.z', 'float')
log.data_received_cb.add_callback(lambda t, data, conf: zs.append(data['stateEstimate.z']))

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./.cache')) as scf:
    cf = scf.cf
    cf.log.add_config(log)
    log.start()
    time.sleep(1.5)
    ground = sum(zs) / len(zs)  # barometer reads height above sea level, so average the resting value
    try:
        cf.supervisor.send_arming_request(True)
        time.sleep(1)
        cf.commander.send_setpoint(0, 0, 0, 0)  # zero thrust first unlocks the motors
        run(cf, ground, 0, HEIGHT, 1.0)         # take off
        run(cf, ground, HEIGHT, HEIGHT, HOVER_S)
        run(cf, ground, HEIGHT, -0.1, 1.5)      # aim slightly below the floor so it settles
    finally:  # always cut motors and disarm, even on Ctrl+C or an error
        cf.commander.send_stop_setpoint()
        cf.supervisor.send_arming_request(False)
        time.sleep(0.1)
