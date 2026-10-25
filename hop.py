"""Short hop over the Crazyradio: arm, ramp thrust up, hold briefly, ramp down, disarm.
No Flow deck means no height or position hold: it stays level but WILL drift.
Prop guards on, 2 m clear around it, USB cable unplugged. Ctrl+C cuts the motors.
Close cfclient (or click Disconnect) first: only one program can use the radio at a time."""
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = 'radio://0/80/2M/E7E7E7E7E7'
MAX_THRUST = 40000  # out of 65535. Start low; raise ~2000 per try until it just lifts off.
RAMP_S, HOLD_S = 1.5, 0.5

def run(cf, start, end, seconds):  # move thrust linearly from start to end, at 50 Hz
    n = int(seconds * 50)
    for i in range(n):
        cf.commander.send_setpoint(0, 0, 0, int(start + (end - start) * i / n))  # roll, pitch, yaw rate, thrust
        time.sleep(0.02)

input(f"Hop to thrust {MAX_THRUST}. Guards on, area clear, USB unplugged? Enter to fly, Ctrl+C to quit. ")
cflib.crtp.init_drivers()
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./.cache')) as scf:
    cf = scf.cf
    try:
        cf.supervisor.send_arming_request(True)
        time.sleep(1)
        cf.commander.send_setpoint(0, 0, 0, 0)  # zero thrust first unlocks the motors
        run(cf, 0, MAX_THRUST, RAMP_S)
        run(cf, MAX_THRUST, MAX_THRUST, HOLD_S)
        run(cf, MAX_THRUST, 0, RAMP_S)
    finally:  # always cut motors and disarm, even on Ctrl+C or an error
        cf.commander.send_stop_setpoint()
        cf.supervisor.send_arming_request(False)
        time.sleep(0.1)
