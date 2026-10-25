"""Watch the Crazyflie read its sensors live over the Crazyradio. Tilt it! Ctrl+C to stop.
Close cfclient (or click Disconnect) first: only one program can use the radio at a time."""
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = 'radio://0/80/2M/E7E7E7E7E7'

def bar(deg):  # draw a -90..+90 degree value as a slider
    pos = round((max(-90, min(90, deg)) + 90) / 180 * 30)
    return '[' + '-' * pos + '|' + '-' * (30 - pos) + ']'

def show(timestamp, data, logconf):
    r, p, y, v = data['stabilizer.roll'], data['stabilizer.pitch'], data['stabilizer.yaw'], data['pm.vbat']
    print(f"\rroll {r:6.1f}° {bar(r)}  pitch {p:6.1f}° {bar(p)}  yaw {y:6.1f}°  battery {v:.2f} V   ", end='', flush=True)

cflib.crtp.init_drivers()
log = LogConfig(name='live', period_in_ms=100)
for name in ('stabilizer.roll', 'stabilizer.pitch', 'stabilizer.yaw', 'pm.vbat'):
    log.add_variable(name, 'float')
log.data_received_cb.add_callback(show)

try:
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./.cache')) as scf:
        scf.cf.log.add_config(log)
        log.start()
        while True:
            time.sleep(1)
except KeyboardInterrupt:
    print()
except Exception as e:
    raise SystemExit(f"Couldn't talk to the Crazyflie on {URI}: {e}\nIs it powered on, plugged in, and is cfclient disconnected?")
