import base64
import glob
import html
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import serial

import rover                     # your motor code (rover.py)

# ================== SETTINGS (change these after testing) ==================
PORT = os.environ.get("ROVER_PORT", "/dev/ttyACM0")
BAUD = 115200

SPEED = 40          # motor power for moving (0-100)
TURN_SPEED = 50     # motor power for turning
CM_PER_SEC = 55.0   # distance the rover moves in 1 second at CALIB_SPEED  (CALIBRATE)
CALIB_SPEED = 40    # the motor power you used when you measured CM_PER_SEC
TURN_90_SEC = 0.35  # seconds needed to spin 90 degrees at TURN_SPEED (CALIBRATE)
IMU_LEAD_SEC = 0.04 # lead time to cut power before target heading (compensates for inertia)
TURN_TOLERANCE_DEG = 3.0 # acceptable heading error for closed-loop turns
STALL_VIB_RISE_MG = 3.5  # minimum vibration rise over standing baseline to confirm wheels rolling
DOOR_MIN_WIDTH_CM = 55.0 # min doorway gap width
DOOR_MAX_WIDTH_CM = 180.0 # max doorway gap width

STOP_DIST = 25      # stop when something is closer than this (cm)
SIDE_MIN = 25       # a side needs at least this much free space to turn there (cm)
BACK_STOP_DIST = 12 # return only: stop backing up when something is closer than this (cm).
                    # Must be SMALLER than STOP_DIST: the rover backs up to spots where it once
                    # stopped ~STOP_DIST from a wall, and that wall is now behind it.

LEFT_ANGLE = 150    # if left and right come out swapped, swap these two numbers
RIGHT_ANGLE = 30

WEB_PORT = int(os.environ.get("ROVER_WEB_PORT", 8000))
FACE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_recog.py")
GCS_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcs", "static", "index.html")
if not os.path.isfile(GCS_PAGE):
    GCS_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
FACE_WIDTH, FACE_HEIGHT = 640, 480   # camera size: smaller = faster on the Pi and less power
FACE_MIRROR = False                  # False = true view (a mirrored view swaps left/right on the dashboard)
FACE_DEBUG = False                   # True prints match scores into this terminal (clutters the prompt)
FACE_FRESH_START = True              # every start of brain.py forgets all previously detected faces + photos
FACE_URL = "http://127.0.0.1:8765"   # face_recog.py's local interface (live people count)
MAX_PEOPLE = 20                      # largest number accepted for "people to find"
CAMERA_LOST_SEC = 5                  # a counting mission halts if the camera is gone this long
MAX_MINUTES = 120                    # largest search time limit accepted (minutes)
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# --- battery: the Uno measures it on A0 (see uno_two_sensors.ino) ---
BATT_FULL_V = 12.6                   # 3 x 18650 fully charged (dashboard bar = 100%)
BATT_EMPTY_V = 9.0                   # 3 x 18650 empty (dashboard bar = 0%)
BATT_WARN_V = 10.5                   # log a warning below this
BATT_RETURN_V = 10.0                 # an exploring rover comes home when it stays below this ...
BATT_LOW_SEC = 5                     # ... for this many seconds (motor load makes the voltage dip briefly)

# --- gyro + accelerometer: the MPU-6500 on the Uno sends  M,yaw,rate,vib  50 times a second ---
IMU_YAW_SIGN = 1                     # +1 = heading grows when the rover turns LEFT (same as turn(deg > 0)). Use -1 if your chip is mounted upside down
IMU_STALE_SEC = 1.0                  # imu_data() returns None if the Uno has not sent a good M line for this long

# --- second dashboard window: the route graph (corners the rover has been to, and the runs between them) ---
ROUTE_MERGE_CM = 30                  # two points closer than this are the SAME node (coming back to a place closes a loop)

# --- walking up to a new face (camera sees a person who is not counted yet) ---
APPROACH_FACES = True                # True: go to a new face first, then carry on with the maze
CAM_HFOV_DEG = 60.0                  # camera horizontal field of view in degrees (CALIBRATE: 60-78 for most USB webcams)
APPROACH_MIN_FRAMES = 4              # a face must be seen in this many camera frames in a row before the rover reacts
APPROACH_AIM_DEG = 8                 # aiming errors smaller than this are ignored (degrees)
APPROACH_MIN_TURN = 12               # smallest aiming turn (very short motor pulses do not move the rover)
APPROACH_MAX_TURN = 35               # largest aiming turn in one step
APPROACH_REAIM_DEG = 16              # while driving up to the person, stop and re-aim if the face drifts this far off-centre
APPROACH_STOP_CM = 30                # stop this far from the person (kept above STOP_DIST)
APPROACH_HOLD_SEC = 3                # wait this long in front of the person for the face to be counted
APPROACH_LOST_SEC = 2.5              # give up if the face is out of sight this long
APPROACH_MAX_SEC = 30                # give up after this long in total
APPROACH_COOLDOWN_SEC = 15           # after giving up, carry on with the maze this long before trying again
command_queue = queue.Queue()
event_log = []
event_id = 0
event_lock = threading.Lock()
control_state = {"mode": "MANUAL", "state": "IDLE", "estop": False, "stalled": False, "space": "room"}
detected_doors = []                  # list of dicts: {"id": "D1", "x": ..., "y": ..., "visited": False}
stationary_vib_baseline = 2.0        # dynamically learned standing-still vibration baseline (mg)
stall_recovery_count = 0             # current tier of stall recovery (0 to 3)
sweep_results = {}                   # angle -> dist during wide sweep
sweep_event = threading.Event()      # set when E,0,0 terminator arrives
is_sweeping = False                  # whether wide sweep is currently active
dashboard_drive_time = None
dashboard_drive_active = False
job = None


def add_event(message):
    global event_id
    with event_lock:
        event_id += 1
        event_log.append({"id": event_id, "text": str(message)})
        del event_log[:-60]
        if mission.get("t0") is not None:
            mission["events"].append((round(time.time() - mission["t0"], 1), str(message)))


def safe_motor(fn, *args):
    """Run a motor command; an I2C hiccup must not kill the whole program."""
    try:
        fn(*args)
        return True
    except OSError as exc:
        print("MOTOR COMMAND FAILED (I2C):", exc)
        add_event(f"Motor command failed: {exc}")
        return False


def update_vib_baseline():
    global stationary_vib_baseline
    data = imu_data()
    if data is not None:
        v = data.get("vib", 2.0)
        if v < 4.5:
            stationary_vib_baseline = 0.95 * stationary_vib_baseline + 0.05 * v


def safe_stop():
    res = safe_motor(rover.stop)
    update_vib_baseline()
    return res
# ============================================================================


def cm_per_sec():
    """Real speed at the current motor power (SPEED can be changed from the terminal)."""
    return CM_PER_SEC * SPEED / CALIB_SPEED


class Halt(Exception):
    """Raised to stop the rover safely."""


class ReturnNow(Halt):
    """The search is over: come home by itself. `reason` says why."""
    reason = ""


class TargetReached(ReturnNow):
    reason = "target"


class TimeUp(ReturnNow):
    reason = "time"


class LowBattery(ReturnNow):
    reason = "battery"


class FaceSpotted(Halt):
    """A person who is not counted yet is in view: walk up to them first."""


# state of the current mission (written by face_monitor and the start command)
mission = {"target": 0, "found": 0, "people": [], "exploring": False,
           "camera": False, "cam_bad_since": None, "epoch": None, "marks": [],
           "approaching": False, "approach_after": 0.0, "face": None,
           "t0": None, "t_end": None, "deadline": None, "limit_min": 0.0,
           "return_reason": None, "outcome": "", "events": [], "batt_start": None,
           "report_dir": None, "report_path": None, "closed": False, "approach_hit": None}
battery = {"v": None, "t": 0.0, "low_since": None, "warned": False}


def battery_volts():
    """Smoothed battery voltage, or None if the Uno is not reporting it."""
    if battery["v"] is None or time.time() - battery["t"] > 3.0:
        return None
    return battery["v"]


def battery_low():
    low = battery["low_since"]
    return low is not None and time.time() - low >= BATT_LOW_SEC


def handle_battery(v):
    now = time.time()
    if v < 3.0:                          # Uno sends -1 when nothing is connected to A0
        battery.update(v=None, low_since=None)
        return
    battery["v"] = v if battery["v"] is None else 0.7 * battery["v"] + 0.3 * v   # smooth the motor-load ripple
    battery["t"] = now
    if battery["v"] < BATT_RETURN_V:
        if battery["low_since"] is None:
            battery["low_since"] = now
    else:
        battery["low_since"] = None
    if battery["v"] < BATT_WARN_V and not battery["warned"]:
        battery["warned"] = True
        print(f"Battery low: {battery['v']:.1f} V")
        add_event(f"Battery low: {battery['v']:.1f} V")
    elif battery["v"] > BATT_WARN_V + 0.3:
        battery["warned"] = False


# latest gyro / accelerometer reading from the Uno
imu = {"yaw": 0.0, "rate": 0.0, "vib": 0.0, "t": 0.0, "ok": False}
imu_watch = {"on": False, "err_t": 0.0}    # terminal command 'imu' switches the live printout on and off


def handle_imu(p):
    """p = the split line: ['M', yaw, rate, vib]  or  ['M', 'ERR']. Bad lines are ignored."""
    if len(p) == 2 and p[1] == "ERR":
        imu["ok"] = False                  # the MPU stopped answering; the Uno keeps trying to restart it
        if imu_watch["on"] and time.time() - imu_watch["err_t"] > 2.0:
            imu_watch["err_t"] = time.time()
            print("MPU not answering (check SDA/SCL on A4/A5, 3.3 V and GND)")
        return
    if len(p) != 4:
        return
    try:
        yaw, rate, vib = float(p[1]), float(p[2]), float(p[3])
    except ValueError:
        return
    imu.update(yaw=yaw * IMU_YAW_SIGN, rate=rate * IMU_YAW_SIGN, vib=vib, t=time.time(), ok=True)
    if imu_watch["on"]:
        print(f"yaw {imu['yaw']:8.2f} deg   rate {imu['rate']:7.2f} deg/s   vib {vib:6.1f} mg")


def imu_data():
    """{'yaw': deg, 'rate': deg/s, 'vib': mg} or None if the MPU is not reporting."""
    if not imu["ok"] or time.time() - imu["t"] > IMU_STALE_SEC:
        return None
    return {"yaw": imu["yaw"], "rate": imu["rate"], "vib": imu["vib"]}


IMU_GUIDE = """
MPU check.  Lines print 10 times a second; type  imu  again to stop.
  * Rover standing still:  rate close to 0.0 and yaw not moving.  Note the vib number.
  * Turn the rover by hand 90 degrees to the LEFT: yaw should go to about +90.
    If it goes to -90, set IMU_YAW_SIGN = -1 at the top of brain.py.
  * Drive on the floor: vib should rise.  Hold a wheel so it is stuck while driving: note the rate and vib values.
  * zero       sets the heading to 0
  * calibrate  re-measures the gyro zero point (keep the rover completely still, takes 1-3 s)
"""


def imu_command(letter):
    """Send Z (heading = 0) or C (re-measure gyro zero point) to the Uno. Returns True if it was sent."""
    try:
        send(letter.encode())
    except Halt as exc:
        print(f"Cannot reach the Uno: {exc}")
        return False
    return True


# furniture / doors the camera program sees right now (filled by face_monitor from face_recog.py's /status)
scene = {"objects": [], "on": False}


def dijkstra_path(nodes_list, edges_list, start_id, end_id):
    """Compute Dijkstra shortest path between start_id and end_id.
    Returns (path_of_node_ids, total_distance_cm)."""
    adj = {n["id"]: [] for n in nodes_list}
    for e in edges_list:
        if e.get("link"):
            continue
        u, v, w = e["a"], e["b"], e["cm"]
        if u in adj and v in adj:
            adj[u].append((v, w))
            adj[v].append((u, w))

    dist = {n["id"]: float("inf") for n in nodes_list}
    prev = {n["id"]: None for n in nodes_list}
    if start_id not in dist or end_id not in dist:
        return [], 0.0
    dist[start_id] = 0.0
    unvisited = set(dist.keys())

    while unvisited:
        curr = min(unvisited, key=lambda node: dist[node])
        if dist[curr] == float("inf") or curr == end_id:
            break
        unvisited.remove(curr)
        for neighbor, weight in adj.get(curr, []):
            if neighbor in unvisited:
                alt = dist[curr] + weight
                if alt < dist[neighbor]:
                    dist[neighbor] = alt
                    prev[neighbor] = curr

    if dist[end_id] == float("inf"):
        return [], 0.0
    path_nodes = []
    step = end_id
    while step is not None:
        path_nodes.append(step)
        step = prev[step]
    path_nodes.reverse()
    return path_nodes, dist[end_id]


def route_graph():
    """Graph for the second dashboard window.
    nodes: the entry, every corner the rover reached, every door found, and every person found.
    edges: straight runs, length in cm.
    return_path: node ids of the Dijkstra shortest way home."""
    nodes, seq = [], []
    for (x, y) in list(trail_out):
        k = next((i for i, n in enumerate(nodes) if math.hypot(n["x"] * 100 - x, n["y"] * 100 - y) <= ROUTE_MERGE_CM), None)
        if k is None:
            nodes.append({"id": f"N{len(nodes)}", "kind": "start" if not nodes else "corner",
                          "x": round(x / 100.0, 3), "y": round(y / 100.0, 3)})
            k = len(nodes) - 1
        if not seq or seq[-1] != k:
            seq.append(k)
    edges, seen_pairs = [], set()
    for a, b in zip(seq, seq[1:]):
        pair = (min(a, b), max(a, b))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        cm = math.hypot(nodes[a]["x"] - nodes[b]["x"], nodes[a]["y"] - nodes[b]["y"]) * 100
        edges.append({"a": nodes[a]["id"], "b": nodes[b]["id"], "cm": round(cm)})
    route_count = len(nodes)
    for d in list(detected_doors):
        dx, dy = d["x"] / 100.0, d["y"] / 100.0
        did = str(d["id"])
        nodes.append({"id": did, "kind": "door", "x": round(dx, 3), "y": round(dy, 3), "label": did})
        if route_count:
            j = min(range(route_count), key=lambda i: math.hypot(nodes[i]["x"] - dx, nodes[i]["y"] - dy))
            edges.append({"a": nodes[j]["id"], "b": did, "cm": round(math.hypot(nodes[j]["x"] - dx, nodes[j]["y"] - dy) * 100), "link": True})
    for m in list(mission["marks"]):
        px, py = m.get("px", m["x"]), m.get("py", m["y"])
        pid = "P:" + str(m["name"])
        nodes.append({"id": pid, "kind": "person", "x": round(px, 3), "y": round(py, 3), "label": str(m["name"])})
        if route_count:
            j = min(range(route_count), key=lambda i: math.hypot(nodes[i]["x"] - px, nodes[i]["y"] - py))
            edges.append({"a": nodes[j]["id"], "b": pid, "cm": round(math.hypot(nodes[j]["x"] - px, nodes[j]["y"] - py) * 100), "link": True})
    
    start_node_id = nodes[seq[-1]]["id"] if seq and seq[-1] < len(nodes) else ("N0" if nodes else "")
    return_path, _ = dijkstra_path(nodes, edges, start_node_id, "N0") if start_node_id else ([], 0.0)
    return {"nodes": nodes, "edges": edges, "return_path": return_path}


ser = None                      # opened by open_uno()

latest = {"F": {}, "B": {}}      # latest[side][angle] = distance in cm
rx = {"t": time.time()}          # time the last line came from the Uno
seen = {"F": 0.0, "B": 0.0}      # time each sensor last reported (to hide stale values)
stop_flag = threading.Event()

pose = {"x": 0.0, "y": 0.0, "h": 90.0}   # position (cm) and heading (degrees)
live = {"x": 0.0, "y": 0.0, "h": 90.0}   # same, but updated while the rover is moving
path = []                                  # recorded moves: ("F", cm) or ("T", degrees)
trail_out = [(0.0, 0.0)]                   # points while going
trail_ret = []                             # points while returning
phase = {"name": "explore"}
done = {"cm": 0.0, "deg": 0.0}            # what the last move/turn really did


# ---------------- reading the Uno ----------------
def reader():
    global is_sweeping
    while True:
        port = ser
        if port is None:
            time.sleep(0.1)
            continue
        try:
            line = port.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            p = line.split(",")
            if p[0] == "M":              # MPU line: M,yaw,rate,vib  (NOT sensor data: rx["t"] must only follow F/B lines)
                handle_imu(p)
                continue
            if p[0] == "E":              # Wide sweep terminator line: E,0,0
                is_sweeping = False
                sweep_event.set()
                continue
            if len(p) != 3 or p[0] not in ("F", "B", "V"):
                continue
            angle = int(p[1])
            dist = float(p[2])
        except ValueError:
            continue
        except (serial.SerialException, OSError, TypeError):
            time.sleep(0.1)      # port is dead, do not spin
            continue
        if p[0] == "V":
            handle_battery(dist)     # NOT counted as sensor data: rx["t"] must only follow F/B lines
            continue
        if dist < 0:
            dist = 999.0         # no echo = nothing close
        if is_sweeping and p[0] == "F":
            sweep_results[angle] = dist
        latest[p[0]][angle] = dist
        rx["t"] = seen[p[0]] = time.time()


threading.Thread(target=reader, daemon=True).start()


def open_uno():
    """Open the Uno: PORT first, then ROVER_PORT or any other ttyACM/ttyUSB/COM. Returns True on success."""
    global ser
    env_port = os.environ.get("ROVER_PORT")
    ports = []
    if env_port:
        ports.append(env_port)
    ports.append(PORT)
    ports.extend(sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")))
    if sys.platform.startswith("win"):
        ports.extend([f"COM{i}" for i in range(1, 10)])
    unique_ports = []
    for p in ports:
        if p and p not in unique_ports:
            unique_ports.append(p)

    last_error = None
    for port in unique_ports:
        try:
            if "://" in str(port):
                new = serial.serial_for_url(port, baudrate=BAUD, timeout=1)
            else:
                new = serial.Serial(port, BAUD, timeout=1)
        except (serial.SerialException, OSError, ValueError) as exc:
            last_error = exc
            continue
        time.sleep(1)                    # allow device / mock to settle
        try:
            new.reset_input_buffer()
        except Exception:
            pass
        old, ser = ser, new
        if old is not None:
            try:
                old.close()
            except (serial.SerialException, OSError):
                pass
        latest["F"].clear()
        latest["B"].clear()
        print("Connected to the Uno on", port)
        return True
    if last_error is not None:
        print("Could not open the Uno:", last_error)
        if "ermission" in str(last_error):
            print("  -> run once, then log out and in:  sudo usermod -aG dialout $USER")
    return False


def send(data):
    """Send a command to the Uno. Stops the rover safely if the USB link is lost."""
    if ser is None:
        raise Halt("Uno is not connected")
    try:
        ser.write(data)
    except (serial.SerialException, OSError):
        raise Halt("USB link to the Uno is lost")


def ensure_link():
    """If the USB link to the Uno is missing or was lost, try to open it again."""
    if ser is not None:
        try:
            ser.write(b"X\n")
            return True
        except (serial.SerialException, OSError):
            pass
    if open_uno():
        return True
    print("Cannot reach the Uno. Check the USB cable and power, then try again.")
    return False


def blocked(side):
    d = latest[side].get(90)
    limit = STOP_DIST if side == "F" else BACK_STOP_DIST
    return d is not None and d < limit


def wait_fresh(side):
    """Wait for a new straight-ahead reading."""
    latest[side].pop(90, None)
    t0 = time.time()
    while 90 not in latest[side]:
        check()
        if time.time() - t0 > 2:
            raise Halt("no data from the Uno")
        time.sleep(0.02)


def check(max_age=None):
    if stop_flag.is_set():
        raise Halt("stopped")
    if mission["exploring"]:
        if mission["target"] > 0 and mission["found"] >= mission["target"]:
            raise TargetReached()
        if mission["deadline"] is not None:
            # Return time budgeting: factor estimated Dijkstra return travel time into mission deadline
            spd = max(10.0, cm_per_sec())
            rg = route_graph()
            nodes_by_id = {n["id"]: n for n in rg["nodes"]}
            rp = rg.get("return_path", [])
            ret_cm = 0.0
            for i in range(len(rp) - 1):
                n1, n2 = nodes_by_id.get(rp[i]), nodes_by_id.get(rp[i+1])
                if n1 and n2:
                    ret_cm += math.hypot((n1["x"] - n2["x"]) * 100, (n1["y"] - n2["y"]) * 100)
            if ret_cm <= 1.0:
                ret_cm = math.hypot(live["x"], live["y"])
            est_return_sec = (ret_cm / spd) + max(1, len(rp)) * 1.5 + 8.0
            if time.time() + est_return_sec >= mission["deadline"]:
                add_event(f"Return time budgeting: leaving for home early ({est_return_sec:.1f}s return buffer)")
                raise TimeUp()
            if time.time() >= mission["deadline"]:
                raise TimeUp()
        if battery_low():
            raise LowBattery()
        bad = mission["cam_bad_since"]
        if mission["target"] > 0 and bad is not None and time.time() - bad > CAMERA_LOST_SEC:
            raise Halt("camera / face detection lost, cannot count people")
        if (APPROACH_FACES and not mission["approaching"] and mission["face"] is not None
                and time.time() >= mission["approach_after"]):
            raise FaceSpotted()
    if max_age and time.time() - rx["t"] > max_age:
        raise Halt("lost data from the Uno")


def sleep_check(sec):
    end = time.time() + sec
    while time.time() < end:
        check()
        time.sleep(0.03)


# ---------------- position and graph ----------------
def trail():
    return trail_out if phase["name"] == "explore" else trail_ret


def sync_live():
    live.update(x=pose["x"], y=pose["y"], h=pose["h"])


def set_live(kind, cm):
    """Estimate the position while the rover is still moving."""
    sign = 1 if kind == "F" else -1
    rad = math.radians(pose["h"])
    live["x"] = pose["x"] + sign * cm * math.cos(rad)
    live["y"] = pose["y"] + sign * cm * math.sin(rad)


def add_move(kind, cm):
    """kind: F = moved forward, B = moved backward"""
    if cm < 1:
        sync_live()
        return
    sign = 1 if kind == "F" else -1
    rad = math.radians(pose["h"])
    pose["x"] += sign * cm * math.cos(rad)
    pose["y"] += sign * cm * math.sin(rad)
    if phase["name"] == "explore":
        path.append(("F", cm))
    trail().append((pose["x"], pose["y"]))
    sync_live()
    print(f"Position: x={pose['x']:.0f} cm, y={pose['y']:.0f} cm")


def add_turn(deg):
    if abs(deg) < 1:
        sync_live()
        return
    pose["h"] += deg
    sync_live()
    if phase["name"] == "explore":
        path.append(("T", deg))


# ---------------- movements & sweeps ----------------
def perform_wide_sweep(side='F', timeout=8.0):
    """Trigger the Uno W wide servo sweep and collect readings until E,0,0 terminator."""
    global is_sweeping
    sweep_results.clear()
    sweep_event.clear()
    is_sweeping = True
    try:
        send(b"W\n")
        t0 = time.time()
        while not sweep_event.is_set() and time.time() - t0 < timeout:
            check()
            time.sleep(0.05)
    finally:
        is_sweeping = False
    return dict(sweep_results)


def front_cm():
    """Latest straight-ahead distance, or None if the front sensor is not reporting."""
    return latest["F"].get(90) if time.time() - seen["F"] < 1.5 else None


def handle_stall_recovery():
    """Escalating 3-tier stall recovery sequence."""
    global stall_recovery_count
    stall_recovery_count += 1
    control_state["stalled"] = True
    add_event(f"Stall detected: initiating recovery attempt {stall_recovery_count} of 3")
    print(f"Stall recovery attempt {stall_recovery_count}...")
    safe_stop()

    if stall_recovery_count == 1:
        # Tier 1: Back off and wiggle
        try:
            move_back(18.0)
            turn(25)
            turn(-50)
            turn(25)
        except Halt:
            pass
        control_state["stalled"] = False
        add_event("Stall recovery tier 1 complete (backoff & wiggle)")
    elif stall_recovery_count == 2:
        # Tier 2: Lateral sidestep maneuver
        try:
            move_back(20.0)
            left_d, right_d = scan_sides()
            side_ang = 90 if left_d >= right_d else -90
            turn(side_ang)
            forward_until_dist(25.0)
            turn(-side_ang)
        except Halt:
            pass
        control_state["stalled"] = False
        add_event("Stall recovery tier 2 complete (lateral sidestep)")
    else:
        # Tier 3: Give up after 3 failed attempts
        control_state["state"] = "STUCK"
        control_state["stalled"] = True
        add_event("Stall recovery failed after 3 attempts: rover STUCK")
        raise Halt("stuck: rover cannot be freed")


def forward_until_dist(dist_cm):
    """Drive forward up to dist_cm, with stall detection, map freeze, and tape-measure calibration."""
    if dist_cm <= 1.0:
        return
    need_time = dist_cm / cm_per_sec()
    t0 = time.time()
    last_t = t0
    accumulated_time = 0.0
    start_yaw = imu["yaw"] if imu_data() else live["h"]

    last_d = front_cm()
    last_d_t = t0
    calib_d0 = last_d
    calib_t0 = t0

    try:
        send(b"F\n")
    except Exception:
        pass
    stall_detected = False
    rover.forward(SPEED)
    try:
        while accumulated_time < need_time:
            now = time.time()
            dt = now - last_t
            last_t = now

            check(0.5)
            fc = front_cm()
            data = imu_data()

            if blocked("F"):
                safe_stop()
                break

            # Signal 1: Front distance unchanged near obstacle (< 40 cm)
            if fc is not None and fc < 40.0:
                if last_d is not None and abs(fc - last_d) < 0.8:
                    if now - last_d_t > 0.6:
                        stall_detected = True
                        break
                else:
                    last_d = fc
                    last_d_t = now
            else:
                last_d = fc
                last_d_t = now

            # Signal 2: Unexpected swing in gyro heading during straight drive
            if data is not None:
                yaw_drift = abs(data["yaw"] - start_yaw)
                rate = abs(data.get("rate", 0.0))
                if yaw_drift > 18.0 or rate > 35.0:
                    stall_detected = True
                    break

            # Signal 3: IMU vibration stays at standing baseline
            if data is not None and now - t0 > 0.4:
                vib = data.get("vib", 0.0)
                thresh = min(5.5, stationary_vib_baseline + STALL_VIB_RISE_MG)
                if vib <= thresh:
                    stall_detected = True
                    break

            # Dynamic distance calibration using front ultrasonic sensor as tape measure
            if calib_d0 is not None and fc is not None and calib_d0 > 30.0 and fc > 15.0:
                delta_d = calib_d0 - fc
                delta_t = now - calib_t0
                if delta_d >= 10.0 and delta_t >= 0.25:
                    v_meas = delta_d / delta_t
                    if 20.0 <= v_meas <= 100.0:
                        global CM_PER_SEC
                        calib_v = v_meas * CALIB_SPEED / SPEED
                        CM_PER_SEC = 0.85 * CM_PER_SEC + 0.15 * calib_v
                        calib_d0 = fc
                        calib_t0 = now

            accumulated_time += dt
            set_live("F", accumulated_time * cm_per_sec())
            time.sleep(0.02)
    finally:
        safe_stop()
        actual_cm = accumulated_time * cm_per_sec()
        done["cm"] = actual_cm
        add_move("F", actual_cm)

    if stall_detected:
        handle_stall_recovery()
    else:
        global stall_recovery_count
        stall_recovery_count = 0
        control_state["stalled"] = False


def forward_until(stop_fn):
    """Drive forward until stop_fn() is True, with stall detection, map freeze, and calibration."""
    t0 = time.time()
    last_t = t0
    accumulated_time = 0.0
    start_yaw = imu["yaw"] if imu_data() else live["h"]

    last_d = front_cm()
    last_d_t = t0
    calib_d0 = last_d
    calib_t0 = t0

    stall_detected = False
    rover.forward(SPEED)
    try:
        while not (stop_fn() or blocked("F")):
            now = time.time()
            dt = now - last_t
            last_t = now

            check(0.5)
            fc = front_cm()
            data = imu_data()

            # Signal 1: Front distance not changing near obstacle
            if fc is not None and fc < 40.0:
                if last_d is not None and abs(fc - last_d) < 0.8:
                    if now - last_d_t > 0.6:
                        stall_detected = True
                        break
                else:
                    last_d = fc
                    last_d_t = now
            else:
                last_d = fc
                last_d_t = now

            # Signal 2: Gyro heading swing during straight drive
            if data is not None:
                yaw_drift = abs(data["yaw"] - start_yaw)
                rate = abs(data.get("rate", 0.0))
                if yaw_drift > 18.0 or rate > 35.0:
                    stall_detected = True
                    break

            # Signal 3: IMU vibration stays at standing baseline
            if data is not None and now - t0 > 0.4:
                vib = data.get("vib", 0.0)
                thresh = min(5.5, stationary_vib_baseline + STALL_VIB_RISE_MG)
                if vib <= thresh:
                    stall_detected = True
                    break

            # Dynamic calibration
            if calib_d0 is not None and fc is not None and calib_d0 > 30.0 and fc > 15.0:
                delta_d = calib_d0 - fc
                delta_t = now - calib_t0
                if delta_d >= 10.0 and delta_t >= 0.25:
                    v_meas = delta_d / delta_t
                    if 20.0 <= v_meas <= 100.0:
                        global CM_PER_SEC
                        calib_v = v_meas * CALIB_SPEED / SPEED
                        CM_PER_SEC = 0.85 * CM_PER_SEC + 0.15 * calib_v
                        calib_d0 = fc
                        calib_t0 = now

            accumulated_time += dt
            set_live("F", accumulated_time * cm_per_sec())
            time.sleep(0.02)
    finally:
        safe_stop()
        actual_cm = accumulated_time * cm_per_sec()
        done["cm"] = actual_cm
        add_move("F", actual_cm)

    if stall_detected:
        handle_stall_recovery()
    else:
        global stall_recovery_count
        stall_recovery_count = 0
        control_state["stalled"] = False


def forward_until_blocked():
    forward_until(lambda: False)


def move_back(cm):
    need = cm / cm_per_sec()           # seconds of moving needed
    moved = 0.0
    last = time.time()
    blocked_since = None
    try:
        while moved < need:
            check(0.5)
            now = time.time()
            if blocked("B"):
                safe_stop()
                if blocked_since is None:
                    blocked_since = now
                    print("Something is behind the rover, waiting...")
                if now - blocked_since > 10:
                    raise Halt("path behind is blocked")
            else:
                blocked_since = None
                rover.backward(SPEED)
                moved += now - last
            last = now
            set_live("B", moved * cm_per_sec())
            time.sleep(0.03)
    finally:
        safe_stop()
        done["cm"] = moved * cm_per_sec()
        add_move("B", moved * cm_per_sec())


def timed_turn(deg):
    """Purely time-based turn using TURN_90_SEC."""
    t0 = time.time()
    if deg > 0:
        rover.turn_left(TURN_SPEED)
    else:
        rover.turn_right(TURN_SPEED)
    total = abs(deg) / 90.0 * TURN_90_SEC
    try:
        while True:
            el = time.time() - t0
            if el >= total:
                break
            check()
            live["h"] = pose["h"] + (1 if deg > 0 else -1) * el / TURN_90_SEC * 90.0
            time.sleep(max(0.001, min(0.02, total - el)))
    finally:
        safe_stop()
        turned = min(abs(deg), (time.time() - t0) / TURN_90_SEC * 90.0)
        done["deg"] = turned if deg > 0 else -turned
        add_turn(done["deg"])
    time.sleep(0.1)


def turn(deg):
    """Closed-loop gyro heading control with IMU_LEAD_SEC; automatic fallback to timed turns."""
    if abs(deg) < 0.5:
        sync_live()
        return

    data0 = imu_data()
    if data0 is not None:
        start_yaw = data0["yaw"]
        target_yaw = start_yaw + deg
        sign = 1 if deg > 0 else -1
        t0 = time.time()
        timeout = max(1.5, abs(deg) / 90.0 * TURN_90_SEC * 4.0 + 2.0)

        if deg > 0:
            rover.turn_left(TURN_SPEED)
        else:
            rover.turn_right(TURN_SPEED)

        last_imu_t = time.time()
        fallback = False
        try:
            while time.time() - t0 < timeout:
                check()
                curr_data = imu_data()
                if curr_data is None:
                    if time.time() - last_imu_t > 0.35:
                        fallback = True
                        break
                else:
                    last_imu_t = time.time()
                    curr_yaw = curr_data["yaw"]
                    rate = curr_data.get("rate", 0.0)
                    projected_yaw = curr_yaw + rate * IMU_LEAD_SEC
                    remaining = (target_yaw - projected_yaw) * sign
                    if remaining <= 0:
                        break
                    live["h"] = pose["h"] + (curr_yaw - start_yaw)
                time.sleep(0.01)
        finally:
            safe_stop()

        if fallback:
            curr_y = imu["yaw"] if imu.get("ok") else start_yaw
            rem = target_yaw - curr_y
            if abs(rem) > 2.0:
                timed_turn(rem)
                return

        time.sleep(0.06)
        final_data = imu_data()
        final_yaw = final_data["yaw"] if final_data else (start_yaw + deg)
        actual_turned = final_yaw - start_yaw
        done["deg"] = actual_turned
        add_turn(actual_turned)
    else:
        timed_turn(deg)


def turn_to(target_heading, max_corrections=3):
    """Turn to target heading, verify final angle and correct missed headings up to 3 times."""
    for attempt in range(max_corrections):
        check()
        err = (target_heading - live["h"] + 180) % 360 - 180
        if abs(err) <= TURN_TOLERANCE_DEG:
            break
        turn(err)


def scan_sides():
    """Look left and right. Returns (left_cm, right_cm)."""
    latest["F"].pop(LEFT_ANGLE, None)
    latest["F"].pop(RIGHT_ANGLE, None)
    send(b"S\n")
    t0 = time.time()
    while time.time() - t0 < 4:
        check()
        if LEFT_ANGLE in latest["F"] and RIGHT_ANGLE in latest["F"]:
            break
        time.sleep(0.05)
    left = latest["F"].get(LEFT_ANGLE)
    right = latest["F"].get(RIGHT_ANGLE)
    if left is None or right is None:
        raise Halt("no scan data from the Uno")
    return left, right


# ---------------- people counting (face_recog.py) ----------------
def face_call(path, post=False, timeout=0.6):
    """Talk to face_recog.py on this Pi. Returns the JSON answer, or None if it is not reachable."""
    try:
        req = Request(FACE_URL + path, data=b"" if post else None, method="POST" if post else "GET")
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, ValueError):
        return None


def face_monitor():
    """Poll the people count twice a second so check() can compare plain numbers."""
    while True:
        time.sleep(0.5)
        st = face_call("/status")
        ok = bool(st) and st.get("frame_age") is not None and st["frame_age"] < 3.0
        mission["camera"] = ok
        scene["on"] = bool(st) and bool(st.get("objects_enabled"))
        fresh = bool(st) and ok and st.get("objects_age") is not None and st["objects_age"] < 3.0
        scene["objects"] = (st.get("objects") or []) if fresh else []
        if ok:
            mission["cam_bad_since"] = None
        elif mission["cam_bad_since"] is None:
            mission["cam_bad_since"] = time.time()
        if st and ok and st.get("streak", 0) >= APPROACH_MIN_FRAMES and st.get("targets"):
            mission["face"] = max(st["targets"], key=lambda g: g["w"])    # the biggest = closest uncounted face
        else:
            mission["face"] = None
        if not st:
            continue
        if mission["epoch"] is not None and st.get("epoch") != mission["epoch"]:
            continue                          # an answer from before this mission started
        people = st.get("people", [])
        for name in people:
            if name not in mission["people"]:
                target = mission["target"]
                of = f" of {target}" if 0 < target and len(people) <= target else " so far"
                msg = f"Person found: {name} ({len(people)}{of})"
                hit = mission["approach_hit"]
                recent = hit is not None and time.time() - hit["t"] < 4.0     # the approach just ended with this person
                approached = mission["approaching"] or recent
                where = hit if recent else {"x": live["x"], "y": live["y"], "h": live["h"], "front": front_cm()}
                mark = {"name": name, "x": where["x"] / 100.0, "y": where["y"] / 100.0,
                        "t": round(time.time() - mission["t0"], 1) if mission["t0"] else None,
                        "how": "walked up to" if approached else "seen while exploring"}
                fc = where["front"]
                if approached and fc is not None and fc < 150:    # facing the person: estimate where they stand
                    rad = math.radians(where["h"])
                    mark["px"] = (where["x"] + fc * math.cos(rad)) / 100.0
                    mark["py"] = (where["y"] + fc * math.sin(rad)) / 100.0
                mission["marks"].append(mark)
                print(msg)
                add_event(msg)
        mission["people"] = people
        mission["found"] = len(people)


threading.Thread(target=face_monitor, daemon=True).start()


# ---------------- the two jobs ----------------
def approach_person():
    """A new face is in view: turn toward it and drive up to it until it is counted. The maze goes on afterwards.
    Every move is recorded like normal exploring, so the return path still works."""
    stop_cm = max(APPROACH_STOP_CM, STOP_DIST + 3)
    t_start = time.time()
    count0 = mission["found"]
    st = {"t": 0.0, "tg": None, "seen": time.time(), "count": count0}
    result = "gave up"
    mission["approaching"] = True
    add_event("New face in view: walking up to it")
    print("New face in view: walking up to it")

    def poll(force=False):
        if force or time.time() - st["t"] > 0.15:
            reply = face_call("/status", timeout=0.4)
            st["t"] = time.time()
            tg = None
            if reply and reply.get("frame_age") is not None and reply["frame_age"] < 1.5:
                if mission["epoch"] is None or reply.get("epoch") == mission["epoch"]:
                    st["count"] = max(st["count"], reply.get("count", 0))
                tgs = reply.get("targets") or []
                if tgs:
                    tg = max(tgs, key=lambda g: g["w"])
            st["tg"] = tg
            if tg:
                st["seen"] = time.time()
        return st["tg"]

    def degrees(tg):
        return -tg["x"] * CAM_HFOV_DEG / 2.0          # + = the face is to the left

    def stop_fn():
        tg = poll()
        if st["count"] > count0:
            return True
        if tg is None:
            return time.time() - st["seen"] > 0.8
        if abs(degrees(tg)) > APPROACH_REAIM_DEG:
            return True                                # drifted off-centre: stop and re-aim
        fc = front_cm()
        return fc is not None and fc < stop_cm

    try:
        wait_fresh("F")                                # a scan may have been interrupted: wait for the sensor stream
        while True:
            check()
            if time.time() - t_start > APPROACH_MAX_SEC:
                result = "took too long"
                break
            tg = poll(True)
            if st["count"] > count0:
                result = "counted"
                break
            if tg is None:
                if time.time() - st["seen"] > APPROACH_LOST_SEC:
                    result = "lost sight of it"
                    break
                time.sleep(0.1)
                continue
            ang = degrees(tg)
            if abs(ang) >= APPROACH_AIM_DEG:
                turn(math.copysign(min(max(abs(ang), APPROACH_MIN_TURN), APPROACH_MAX_TURN), ang))
                continue
            fc = front_cm()
            if fc is not None and fc < stop_cm:
                t_hold = time.time()                   # arrived: stand still so the face can be counted
                while time.time() - t_hold < APPROACH_HOLD_SEC and st["count"] <= count0:
                    check()
                    time.sleep(0.15)
                    poll(True)
                result = "counted" if st["count"] > count0 else "was not counted"
                break
            forward_until(stop_fn)
    finally:
        if result == "counted":                    # remember where we stood, so the report can place the person
            mission["approach_hit"] = {"t": time.time(), "x": live["x"], "y": live["y"], "h": live["h"], "front": front_cm()}
        mission["approaching"] = False
        mission["approach_after"] = time.time() + (1.5 if result == "counted" else APPROACH_COOLDOWN_SEC)
    if result != "counted":
        add_event(f"Approach ended ({result}); back to the maze")
        print(f"Approach ended ({result}); back to the maze")


def get_360_profile():
    """Build a 360-degree surround profile using two wide sweeps (front and turned 180)."""
    send(b"F\n")
    h0 = live["h"]
    sweep1 = perform_wide_sweep('F')
    turn_to(h0 + 180)
    sweep2 = perform_wide_sweep('F')
    turn_to(h0)

    profile = {}
    for ang, dist in sweep1.items():
        bearing = (ang - 90) % 360
        profile[bearing] = dist
    for ang, dist in sweep2.items():
        bearing = (ang - 90 + 180) % 360
        profile[bearing] = dist
    return profile


def classify_space(profile):
    """Classify environment as 'room' or 'corridor' from surround profile."""
    left_samples = [d for b, d in profile.items() if 45 <= b <= 135 and d < 900]
    right_samples = [d for b, d in profile.items() if 225 <= b <= 315 and d < 900]
    left_dist = min(left_samples) if left_samples else 200.0
    right_dist = min(right_samples) if right_samples else 200.0
    total_width = left_dist + right_dist

    fwd_samples = [d for b, d in profile.items() if (b <= 30 or b >= 330) and d < 900]
    back_samples = [d for b, d in profile.items() if 150 <= b <= 210 and d < 900]
    ahead_dist = max(fwd_samples) if fwd_samples else 100.0
    behind_dist = max(back_samples) if back_samples else 100.0
    length = ahead_dist + behind_dist

    if total_width <= 220.0 and length > total_width * 1.3:
        space = "corridor"
    else:
        space = "room"
    control_state["space"] = space
    add_event(f"Space classified as: {space.upper()} (width {total_width:.0f}cm, length {length:.0f}cm)")
    return space


def detect_doors_in_profile(profile):
    """Identify doors as gaps 55 cm to 180 cm wide in ultrasonic sweep and camera."""
    sorted_bearings = sorted(profile.keys())
    if not sorted_bearings:
        return

    for i, b in enumerate(sorted_bearings):
        d = profile[b]
        prev_b = sorted_bearings[(i - 1) % len(sorted_bearings)]
        next_b = sorted_bearings[(i + 1) % len(sorted_bearings)]
        d_prev = profile[prev_b]
        d_next = profile[next_b]
        wall_dist = min(d_prev, d_next)
        if wall_dist < 400.0 and (d > wall_dist + 35.0 or (d >= 350.0 and wall_dist < 250.0)):
            rad = math.radians(live["h"] + b)
            door_dist = wall_dist + 10.0
            door_x = live["x"] + door_dist * math.cos(rad)
            door_y = live["y"] + door_dist * math.sin(rad)
            if not any(math.hypot(d_exist["x"] - door_x, d_exist["y"] - door_y) < 70.0 for d_exist in detected_doors):
                d_id = f"D{len(detected_doors) + 1}"
                new_door = {"id": d_id, "x": round(door_x, 1), "y": round(door_y, 1), "visited": False}
                detected_doors.append(new_door)
                add_event(f"Door {d_id} detected at ({door_x/100:.2f}m, {door_y/100:.2f}m)")

    for obj in scene.get("objects", []):
        if obj.get("label") == "door":
            cam_x = obj.get("x", 0.0)
            cam_bearing = cam_x * (CAM_HFOV_DEG / 2.0)
            fc = front_cm() or 120.0
            rad = math.radians(live["h"] + cam_bearing)
            door_x = live["x"] + fc * math.cos(rad)
            door_y = live["y"] + fc * math.sin(rad)
            if not any(math.hypot(d_exist["x"] - door_x, d_exist["y"] - door_y) < 70.0 for d_exist in detected_doors):
                d_id = f"D{len(detected_doors) + 1}"
                new_door = {"id": d_id, "x": round(door_x, 1), "y": round(door_y, 1), "visited": False}
                detected_doors.append(new_door)
                add_event(f"Door {d_id} detected via camera at ({door_x/100:.2f}m, {door_y/100:.2f}m)")


def approach_and_pass_door(door):
    """Navigate to unvisited door, stop at ~1m distance to verify opening, then pass through."""
    add_event(f"Navigating toward door {door['id']} at ({door['x']/100:.2f}m, {door['y']/100:.2f}m)")
    dx = door["x"] - live["x"]
    dy = door["y"] - live["y"]
    dist = math.hypot(dx, dy)
    target_heading = math.degrees(math.atan2(dy, dx))

    turn_to(target_heading)

    stop_dist_from_door = 100.0
    drive_dist = max(0.0, dist - stop_dist_from_door)
    if drive_dist > 10.0:
        forward_until_dist(drive_dist)

    add_event(f"Verifying doorway opening for door {door['id']}")
    time.sleep(0.3)
    wait_fresh("F")
    fc = front_cm()
    if fc is not None and fc < 35.0:
        add_event(f"Door {door['id']} blocked ahead ({fc:.0f} cm)")
        return False

    add_event(f"Passing through door {door['id']}")
    forward_until_dist(130.0)

    door["visited"] = True
    add_event(f"Door {door['id']} passed through and marked visited")
    return True


def explore():
    phase["name"] = "explore"
    send(b"F\n")
    time.sleep(0.3)
    print("Exploring... type 'return' to come back, 'stop' to halt")
    add_event("Exploration mission started")

    steps_without_door = 0
    while True:
        try:
            check()
            time.sleep(0.1)

            profile = get_360_profile()
            space_type = classify_space(profile)

            detect_doors_in_profile(profile)

            unvisited_doors = [d for d in detected_doors if not d.get("visited")]

            if space_type == "room":
                if unvisited_doors:
                    closest_door = min(unvisited_doors, key=lambda d: math.hypot(d["x"] - live["x"], d["y"] - live["y"]))
                    approach_and_pass_door(closest_door)
                    steps_without_door = 0
                else:
                    all_unvisited = [d for d in detected_doors if not d.get("visited")]
                    if all_unvisited:
                        closest_door = min(all_unvisited, key=lambda d: math.hypot(d["x"] - live["x"], d["y"] - live["y"]))
                        approach_and_pass_door(closest_door)
                        steps_without_door = 0
                    else:
                        print("everything reachable explored")
                        add_event("everything reachable explored")
                        raise TargetReached()
            else:
                # In corridor: if an unvisited side door is nearby (< 180 cm), explore it
                nearby_doors = [d for d in unvisited_doors if math.hypot(d["x"] - live["x"], d["y"] - live["y"]) < 180.0]
                if nearby_doors:
                    closest_door = min(nearby_doors, key=lambda d: math.hypot(d["x"] - live["x"], d["y"] - live["y"]))
                    approach_and_pass_door(closest_door)
                    steps_without_door = 0
                else:
                    wait_fresh("F")
                    fc = front_cm()
                    if fc is not None and fc > STOP_DIST + 40:
                        step_cm = min(100.0, fc - STOP_DIST - 10)
                        if step_cm > 25.0:
                            add_event(f"Corridor step: advancing {step_cm:.0f} cm")
                            forward_until_dist(step_cm)
                            steps_without_door += 1
                        elif unvisited_doors:
                            closest_door = min(unvisited_doors, key=lambda d: math.hypot(d["x"] - live["x"], d["y"] - live["y"]))
                            approach_and_pass_door(closest_door)
                            steps_without_door = 0
                        else:
                            left, right = scan_sides()
                            if max(left, right) < SIDE_MIN:
                                turn(180)
                            elif left >= right:
                                turn(90)
                            else:
                                turn(-90)
                    elif unvisited_doors:
                        closest_door = min(unvisited_doors, key=lambda d: math.hypot(d["x"] - live["x"], d["y"] - live["y"]))
                        approach_and_pass_door(closest_door)
                        steps_without_door = 0
                    else:
                        left, right = scan_sides()
                        if max(left, right) < SIDE_MIN:
                            all_unvisited = [d for d in detected_doors if not d.get("visited")]
                            if all_unvisited:
                                closest_door = min(all_unvisited, key=lambda d: math.hypot(d["x"] - live["x"], d["y"] - live["y"]))
                                approach_and_pass_door(closest_door)
                            else:
                                print("everything reachable explored")
                                add_event("everything reachable explored")
                                raise TargetReached()
                        elif left >= right:
                            turn(90)
                        else:
                            turn(-90)

        except FaceSpotted:
            approach_person()


def go_home():
    phase["name"] = "return"
    trail_ret.append((pose["x"], pose["y"]))
    send(b"F\n")
    time.sleep(0.3)
    print("Returning home along Dijkstra shortest route...")
    add_event("Returning home along Dijkstra shortest route")

    rg = route_graph()
    nodes_by_id = {n["id"]: n for n in rg["nodes"]}
    current_node = min([n for n in rg["nodes"] if n["kind"] != "person"],
                       key=lambda n: math.hypot(n["x"]*100 - live["x"], n["y"]*100 - live["y"]),
                       default=None)
    start_id = current_node["id"] if current_node else "N0"
    edges = list(rg["edges"])

    while True:
        check()
        dist_to_start = math.hypot(live["x"], live["y"])
        if dist_to_start < 15.0:
            break

        ret_path, total_cm = dijkstra_path(rg["nodes"], edges, start_id, "N0")
        if not ret_path or len(ret_path) < 2:
            dx = 0.0 - live["x"]
            dy = 0.0 - live["y"]
            heading = math.degrees(math.atan2(dy, dx))
            turn_to(heading)
            dist_leg = math.hypot(dx, dy)
            forward_until_dist(dist_leg)
            break

        next_id = ret_path[1]
        next_node = nodes_by_id.get(next_id)
        if not next_node:
            break

        target_x, target_y = next_node["x"] * 100.0, next_node["y"] * 100.0
        dx = target_x - live["x"]
        dy = target_y - live["y"]
        target_heading = math.degrees(math.atan2(dy, dx))
        leg_dist = math.hypot(dx, dy)

        turn_to(target_heading)

        wait_fresh("F")
        fc = front_cm()
        if fc is not None and fc < min(leg_dist, STOP_DIST + 10):
            add_event(f"Return route leg {start_id}->{next_id} blocked! Re-planning via Dijkstra...")
            edges = [e for e in edges if not ((e["a"] == start_id and e["b"] == next_id) or (e["a"] == next_id and e["b"] == start_id))]
            continue

        try:
            forward_until_dist(leg_dist)
            start_id = next_id
        except Halt as h:
            if "stuck" in str(h):
                raise
            pass

    turn_to(90.0)
    final_err = math.hypot(pose["x"], pose["y"])
    print(f"Back at the start. Distance from exact start point: {final_err:.1f} cm")
    add_event(f"Returned to start: distance from exact start point is {final_err:.1f} cm")
    control_state["state"] = "DONE"


# ---------------- mission report ----------------
REPORT_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#eef2f1;color:#17303a}
main{max-width:980px;margin:0 auto;padding:24px 18px 60px}
h1{margin:0 0 4px;font-size:28px} h2{margin:32px 0 12px;font-size:20px;border-bottom:2px solid #c5d3d0;padding-bottom:6px}
.sub{color:#5b7580;margin:0 0 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.stat{background:#fff;border:1px solid #c5d3d0;border-radius:8px;padding:12px}
.stat b{display:block;font-size:22px;margin-top:2px} .stat span{font-size:13px;color:#5b7580}
.card{display:flex;gap:16px;background:#fff;border:1px solid #c5d3d0;border-radius:8px;padding:14px;margin-bottom:12px;flex-wrap:wrap}
.card img{height:150px;border-radius:6px;border:1px solid #c5d3d0;background:#dde6e4}
.card .pics{display:flex;gap:10px;flex-wrap:wrap} .card h3{margin:0 0 6px}
dl{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;margin:0;font-size:14px} dt{color:#5b7580}
dd{margin:0}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #c5d3d0;font-size:14px}
td,th{padding:6px 10px;border-bottom:1px solid #e1e9e7;text-align:left} td:first-child{font-family:ui-monospace,monospace;width:70px;color:#5b7580}
.map{background:#d9e3e1;border:1px solid #c5d3d0;border-radius:8px;max-width:600px;width:100%}
.note{font-size:13px;color:#5b7580;margin-top:18px}
@media print{body{background:#fff}.card,.stat{break-inside:avoid}}
"""


def _fmt_time(sec):
    sec = int(max(0, sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _image_uri(path):
    try:
        if path and os.path.isfile(path) and os.path.getsize(path) <= 600000:
            with open(path, "rb") as f:
                return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except OSError:
        pass
    return None


def _svg_map(out_m, ret_m, marks, size=560):
    """The path as an inline SVG (same look as the dashboard map). Coordinates in metres."""
    pts = [(0.0, 0.0)] + out_m + ret_m + [(m.get("px", m["x"]), m.get("py", m["y"])) for m in marks]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    pad = 46
    sc = min((size - 2 * pad) / max(maxx - minx, 2.0), (size - 2 * pad) / max(maxy - miny, 2.0))
    mx, my = (minx + maxx) / 2, (miny + maxy) / 2

    def X(x):
        return size / 2 + (x - mx) * sc

    def Y(y):
        return size / 2 - (y - my) * sc

    step = next((v for v in (0.25, 0.5, 1, 2, 5, 10, 20) if v * sc >= 48), 20)
    grid = []
    g = math.floor((mx - size / 2 / sc) / step) * step
    while g <= mx + size / 2 / sc:
        grid.append(f'<line x1="{X(g):.1f}" y1="0" x2="{X(g):.1f}" y2="{size}"/>')
        g += step
    g = math.floor((my - size / 2 / sc) / step) * step
    while g <= my + size / 2 / sc:
        grid.append(f'<line x1="0" y1="{Y(g):.1f}" x2="{size}" y2="{Y(g):.1f}"/>')
        g += step

    def poly(points, color, dash=""):
        if len(points) < 2:
            return ""
        d = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in points)
        return (f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="3" '
                f'stroke-linejoin="round" {dash}/>')

    parts = [f'<svg class="map" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="Path of the rover">',
             f'<g stroke="rgba(23,48,58,.15)">{"".join(grid)}</g>',
             poly(out_m, "#1f6fd0"), poly(ret_m, "#e5432d", 'stroke-dasharray="9 6"'),
             f'<rect x="{X(0) - 7:.1f}" y="{Y(0) - 7:.1f}" width="14" height="14" fill="#d9e3e1" stroke="#17303a" stroke-width="2"/>',
             f'<text x="{X(0) + 12:.1f}" y="{Y(0) + 4:.1f}" font-size="12" font-weight="600" fill="#17303a">Entry / Exit</text>']
    for m in marks:
        px, py = m.get("px", m["x"]), m.get("py", m["y"])
        parts.append(f'<circle cx="{X(px):.1f}" cy="{Y(py):.1f}" r="8" fill="#f2b63d" stroke="#17303a" stroke-width="2"/>')
        parts.append(f'<text x="{X(px) + 12:.1f}" y="{Y(py) - 8:.1f}" font-size="12" font-weight="600" fill="#17303a">'
                     f'{html.escape(m["name"])}</text>')
    parts.append(f'<text x="10" y="{size - 10}" font-size="12" fill="rgba(23,48,58,.75)">grid {step} m, up = direction at start</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_report():
    """Write reports/mission_<time>/report.html (+ report.json and photos/). Safe to call again: it overwrites."""
    if mission["t0"] is None or mission["report_dir"] is None:
        return
    t0 = mission["t0"]
    end = mission["t_end"] or time.time()
    folder = mission["report_dir"]
    os.makedirs(os.path.join(folder, "photos"), exist_ok=True)
    files = face_call("/people", timeout=2.0) or {}

    people = []
    for m in mission["marks"]:
        entry = dict(m)
        entry["files"] = {}
        for key in ("photo", "snapshot", "crop"):
            src = files.get(m["name"], {}).get(key)
            if src and os.path.isfile(src):
                dst = os.path.join(folder, "photos", f"{m['name'].replace(' ', '_')}_{key}.jpg")
                shutil.copyfile(src, dst)
                entry["files"][key] = os.path.join("photos", os.path.basename(dst))
        people.append(entry)

    reasons = {"target": "people target reached", "time": "time limit reached",
               "battery": "battery low", "operator": "operator pressed Return"}
    if control_state["state"] == "DONE":
        how = "Returned to the entry point"
        if mission["return_reason"]:
            how += f" ({reasons.get(mission['return_reason'], mission['return_reason'])})"
    else:
        how = mission["outcome"] or "Running"

    out_m = [(x / 100.0, y / 100.0) for x, y in trail_out]
    ret_m = [(x / 100.0, y / 100.0) for x, y in trail_ret]

    def length(p):
        return sum(math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]) for i in range(len(p) - 1))

    travelled = length(out_m) + length(ret_m)
    from_entry = math.hypot(live["x"], live["y"]) / 100.0
    v_end = battery_volts()
    target = mission["target"]
    limit = mission["limit_min"]
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
    data = {
        "started": started, "duration_s": round(end - t0, 1), "outcome": how,
        "people_target": target, "people_found": len(people), "time_limit_min": limit,
        "battery_start_v": None if mission["batt_start"] is None else round(mission["batt_start"], 2),
        "battery_end_v": None if v_end is None else round(v_end, 2),
        "estimated_distance_travelled_m": round(travelled, 1),
        "estimated_distance_from_entry_at_end_m": round(from_entry, 2),
        "people": people,
        "path_out_m": [[round(a, 3), round(b, 3)] for a, b in out_m],
        "path_return_m": [[round(a, 3), round(b, 3)] for a, b in ret_m],
        "events": [{"t": t, "text": txt} for t, txt in mission["events"]],
        "note": "Positions are estimates from motor run time (no encoders / IMU) and drift as the rover moves.",
    }
    with open(os.path.join(folder, "report.json"), "w") as f:
        json.dump(data, f, indent=2)

    esc = html.escape

    def stat(label, value):
        return f'<div class="stat"><span>{esc(label)}</span><b>{esc(str(value))}</b></div>'

    batt = "not measured" if mission["batt_start"] is None else (
        f"{mission['batt_start']:.1f} V to {v_end:.1f} V" if v_end is not None else f"{mission['batt_start']:.1f} V at start")
    stats = "".join([
        stat("People found", f"{len(people)} of {target}" if target > 0 else f"{len(people)} (no target)"),
        stat("Mission time", _fmt_time(end - t0) + (f" of {limit:g} min limit" if limit > 0 else "")),
        stat("How it ended", how), stat("Battery", batt),
        stat("Distance driven (est.)", f"{travelled:.1f} m"),
        stat("Distance from entry at end (est.)", f"{from_entry:.2f} m")])

    cards = []
    for i, p in enumerate(people, 1):
        pics = []
        for key in (("photo", "crop"), ("snapshot",)):
            rel = next((p["files"][k] for k in key if k in p["files"]), None)
            uri = _image_uri(os.path.join(folder, rel)) if rel else None
            if uri:
                pics.append(f'<img src="{uri}" alt="{esc(p["name"])} photo">')
        est = f'({p["px"]:.2f}, {p["py"]:.2f}) m' if "px" in p else "not measured"
        cards.append(
            f'<article class="card"><div class="pics">{"".join(pics) or "<em>no photo saved</em>"}</div><div>'
            f'<h3>{i}. {esc(p["name"])}</h3><dl>'
            f'<dt>Found at</dt><dd>{_fmt_time(p["t"]) if p.get("t") is not None else "-"} after start</dd>'
            f'<dt>How</dt><dd>{esc(p.get("how", ""))}</dd>'
            f'<dt>Rover position then</dt><dd>({p["x"]:.2f}, {p["y"]:.2f}) m from the entry point</dd>'
            f'<dt>Person (front sensor)</dt><dd>{est}</dd></dl></div></article>')
    events = "".join(f"<tr><td>{_fmt_time(t)}</td><td>{esc(txt)}</td></tr>" for t, txt in mission["events"])
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Mission report {esc(started)}</title><style>{REPORT_CSS}</style></head><body><main>'
            f'<h1>Mission report</h1><p class="sub">Team Phoenix rover, started {esc(started)}</p>'
            f'<div class="grid">{stats}</div>'
            f'<h2>People found</h2>{"".join(cards) or "<p>Nobody was found.</p>"}'
            f'<h2>Path</h2>{_svg_map(out_m, ret_m, people)}'
            f'<p class="note"><span style="color:#1f6fd0">&#9644;</span> going out &nbsp; <span style="color:#e5432d">- -</span> returning &nbsp; '
            f'yellow dot = where the person was (front-sensor estimate when the rover walked up to them, otherwise where the rover was)</p>'
            f'<h2>Timeline</h2><table>{events or "<tr><td></td><td>No events</td></tr>"}</table>'
            f'<p class="note">{esc(data["note"])}</p></main></body></html>')
    path_html = os.path.join(folder, "report.html")
    with open(path_html, "w", encoding="utf-8") as f:
        f.write(page)
    first = mission["report_path"] is None
    mission["report_path"] = path_html
    if first:
        add_event("Mission report saved")
    print("Mission report:", path_html)


def write_report_safe():
    try:
        write_report()
    except Exception as exc:        # a report problem must never stop the rover
        print("Could not write the mission report:", repr(exc))


def mission_run():
    """Explore; if a people target, a time limit or a low battery says so, come home by itself."""
    mission["exploring"] = True
    try:
        explore()
    except ReturnNow as e:
        mission["exploring"] = False
        mission["return_reason"] = e.reason
        found, target = mission["found"], mission["target"]
        of = f" of {target}" if target > 0 else ""
        if e.reason == "target":
            msg = f"Target reached: {found}{of} people found. Returning to the entry point"
        elif e.reason == "time":
            msg = (f"Time limit ({mission['limit_min']:g} min) reached with {found}{of} people found. "
                   f"Returning to the entry point")
        else:
            v = battery["v"]
            volts = f" ({v:.1f} V)" if v is not None else ""
            msg = f"Battery low{volts} with {found}{of} people found. Returning to the entry point"
        print(msg)
        add_event(msg)
        control_state.update(mode="AUTO", state="RETURNING")
        go_home()                             # the move that was interrupted is already recorded
    finally:
        mission["exploring"] = False


def worker(fn):
    global dashboard_drive_active, dashboard_drive_time
    try:
        fn()
    except Halt as e:
        print("Halted:", e)
        add_event(f"Mission halted: {e}")
        mission["outcome"] = "Stopped by the operator" if str(e) == "stopped" else f"Halted: {e}"
    except Exception as e:           # I2C error or a bug: report it, never die silently
        print("Mission crashed:", repr(e))
        add_event(f"Mission crashed: {e!r}")
        mission["outcome"] = f"Crashed: {e!r}"
    finally:
        mission["exploring"] = False
        safe_stop()
        try:
            if ser is not None:
                ser.write(b"X\n")
        except (serial.SerialException, OSError):
            pass
        if not control_state["estop"]:
            if control_state["state"] != "DONE":
                control_state["state"] = "IDLE"
            control_state["mode"] = "MANUAL"
        dashboard_drive_active = False
        if mission["t0"] is not None and not mission["closed"]:
            mission["t_end"] = time.time()
            if control_state["state"] == "DONE":
                mission["closed"] = True      # finished: later jobs must not rewrite this report
            write_report_safe()
        print("Ready. Type help to see the commands")


# ---------------- live web page ----------------
PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rover live graph</title>
<style>
body{font-family:sans-serif;margin:12px;background:#fafafa;color:#222}
canvas{width:100%;max-width:700px;background:#fff;border:1px solid #ccc}
#info{margin:8px 0;font-size:15px}
</style></head><body>
<h3>Rover live graph</h3>
<div id="info">Waiting for data...</div>
<canvas id="c" width="700" height="700"></canvas>
<script>
const c = document.getElementById("c");
const g = c.getContext("2d");
function line(pts, X, Y, color, dash){
  if (pts.length < 2) return;
  g.beginPath(); g.setLineDash(dash); g.strokeStyle = color; g.lineWidth = 3;
  pts.forEach((p, i) => { if (i == 0) g.moveTo(X(p[0]), Y(p[1])); else g.lineTo(X(p[0]), Y(p[1])); });
  g.stroke(); g.setLineDash([]);
  g.fillStyle = color;
  pts.forEach(p => { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 4, 0, 7); g.fill(); });
}
function draw(d){
  const out = d.out.slice(), ret = d.ret.slice();
  if (d.phase == "explore") out.push([d.x, d.y]); else if (ret.length) ret.push([d.x, d.y]);
  const all = out.concat(ret, [[0, 0]]);
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  const minx = Math.min(...xs), maxx = Math.max(...xs);
  const miny = Math.min(...ys), maxy = Math.max(...ys);
  const span = Math.max(maxx - minx, maxy - miny, 60);
  const scale = (c.width - 120) / span;
  const cx = (minx + maxx) / 2, cy = (miny + maxy) / 2;
  const X = x => c.width / 2 + (x - cx) * scale;
  const Y = y => c.height / 2 - (y - cy) * scale;
  g.clearRect(0, 0, c.width, c.height);
  line(out, X, Y, "#1f6fd0", []);
  line(ret, X, Y, "#d03a1f", [8, 6]);
  g.fillStyle = "#1a9c4b"; g.font = "bold 14px sans-serif";
  g.beginPath(); g.arc(X(0), Y(0), 9, 0, 7); g.fill();
  g.fillText("START / EXIT", X(0) + 14, Y(0) + 5);
  g.save(); g.translate(X(d.x), Y(d.y)); g.rotate(-d.h * Math.PI / 180);
  g.fillStyle = "#222"; g.beginPath();
  g.moveTo(14, 0); g.lineTo(-9, 8); g.lineTo(-9, -8); g.closePath(); g.fill(); g.restore();
  const nice = [10, 20, 50, 100, 200, 500, 1000].filter(v => v <= span / 3).pop() || 10;
  g.strokeStyle = "#444"; g.lineWidth = 3; g.beginPath();
  g.moveTo(30, c.height - 30); g.lineTo(30 + nice * scale, c.height - 30); g.stroke();
  g.fillStyle = "#444"; g.font = "13px sans-serif"; g.fillText(nice + " cm", 30, c.height - 38);
  const f = d.front == null ? "-" : d.front.toFixed(0);
  const b = d.back == null ? "-" : d.back.toFixed(0);
  document.getElementById("info").textContent =
    "Mode: " + d.phase + "  |  x=" + d.x.toFixed(0) + " cm, y=" + d.y.toFixed(0) +
    " cm  |  front " + f + " cm, back " + b + " cm";
}
async function tick(){
  try { const r = await fetch("/data"); draw(await r.json()); } catch (e) {}
}
setInterval(tick, 500); tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] == "/report":
            rp = mission["report_path"]
            if rp and os.path.isfile(rp):
                with open(rp, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404, "No mission report yet")
            return
        if self.path.startswith("/api/data"):
            try:
                from urllib.parse import urlparse, parse_qs
                qs = parse_qs(urlparse(self.path).query)
                since = int(qs.get("since", [0])[0])
            except (ValueError, TypeError):
                since = 0
            running = job is not None and job.is_alive()
            state = control_state["state"]
            now = time.time()
            volts = battery_volts()
            pct = None if volts is None else max(0, min(100, round((volts - BATT_EMPTY_V) / (BATT_FULL_V - BATT_EMPTY_V) * 100)))
            bstate = ("none" if volts is None else "low" if volts < BATT_RETURN_V
                      else "warn" if volts < BATT_WARN_V else "ok")
            elapsed = None if mission["t0"] is None else (mission["t_end"] or now) - mission["t0"]
            data = {
                "mode": control_state["mode"],
                "state": "ESTOP" if control_state["estop"] else state,
                "estop": control_state["estop"],
                "x": live["x"] / 100.0,
                "y": live["y"] / 100.0,
                "heading": live["h"],
                "speed": SPEED,
                "battery": volts,
                "battery_pct": pct,
                "battery_state": bstate,
                "imu_ok": imu_data() is not None,
                "imu_yaw": imu["yaw"],
                "imu_vib": imu["vib"],
                "stalled": control_state.get("stalled", False),
                "space": control_state.get("space", "room"),
                "doors_count": len(detected_doors),
                "doors": list(detected_doors),
                "objects_on": scene["on"],
                "objects": list(scene["objects"]),
                "route_graph": route_graph(),
                "elapsed_s": elapsed,
                "time_limit_min": mission["limit_min"],
                "approaching": mission["approaching"],
                "report_ready": mission["report_path"] is not None,
                "survivors": mission["found"],
                "survivor_target": mission["target"],
                "people": list(mission["people"]),
                "camera_ok": mission["camera"],
                "crumbs": len(trail_out) + len(trail_ret),
                "running": running,
                "front_cm": latest["F"].get(90) if now - seen["F"] < 1.5 else None,
                "back_cm": latest["B"].get(90) if now - seen["B"] < 1.5 else None,
                "path_out": [[x / 100.0, y / 100.0] for x, y in trail_out],
                "path_return": [[x / 100.0, y / 100.0] for x, y in trail_ret],
                "map_available": True,
                "found_at": list(mission["marks"]),
            }
            with event_lock:
                data["events"] = [item for item in event_log if item["id"] > since]
                data["event_id"] = event_id
            body = json.dumps(data).encode()
            ctype = "application/json"
        elif self.path.startswith("/video"):
            upstream = None
            try:
                upstream = urlopen("http://127.0.0.1:8765/video", timeout=8)
                first = upstream.read1(8192)          # wait for the first frame
                if not first:
                    raise OSError("camera stream is empty")
            except (URLError, OSError) as exc:
                if upstream is not None:
                    upstream.close()
                body = f"Camera stream unavailable: {exc}".encode()
                self.send_response(503)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", upstream.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame"))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            try:
                chunk = first
                while chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    chunk = upstream.read1(8192)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                upstream.close()
            return
        elif self.path.startswith("/data"):
            data = {
                "out": list(trail_out),
                "ret": list(trail_ret),
                "x": live["x"], "y": live["y"], "h": live["h"],
                "phase": phase["name"],
                "front": latest["F"].get(90) if time.time() - seen["F"] < 1.5 else None,
                "back": latest["B"].get(90) if time.time() - seen["B"] < 1.5 else None,
            }
            body = json.dumps(data).encode()
            ctype = "application/json"
        elif self.path == "/graph":
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        elif self.path == "/" or self.path.startswith("/index.html"):
            try:
                gcs_file = GCS_PAGE
                if not os.path.isfile(gcs_file):
                    gcs_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
                body = Path(gcs_file).read_bytes()
            except OSError as exc:
                body = f"GCS page unavailable: {exc}".encode()
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/control":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            action = payload.get("action")
            if action not in {"start", "return", "stop", "mode", "estop", "reset", "drive", "speed"}:
                raise ValueError("unknown action")
            if action == "mode" and payload.get("value") not in {"AUTO", "MANUAL"}:
                raise ValueError("mode must be AUTO or MANUAL")
            if action == "drive":
                lin = float(payload.get("lin", 0))
                ang = float(payload.get("ang", 0))
                if not math.isfinite(lin) or not math.isfinite(ang):
                    raise ValueError("drive values must be finite")
                payload = {"action": "drive", "lin": max(-1.0, min(1.0, lin)),
                           "ang": max(-1.0, min(1.0, ang))}
            if action == "start":
                people = int(payload.get("people", 0))
                if not 0 <= people <= MAX_PEOPLE:
                    raise ValueError(f"people must be between 0 and {MAX_PEOPLE}")
                minutes = float(payload.get("minutes", 0))
                if not math.isfinite(minutes) or not 0 <= minutes <= MAX_MINUTES:
                    raise ValueError(f"minutes must be between 0 and {MAX_MINUTES}")
                payload = {"action": "start", "people": people, "minutes": minutes}
            if action == "speed":
                speed = int(payload.get("value"))
                if not 10 <= speed <= 100:
                    raise ValueError("speed must be between 10 and 100")
                payload = {"action": "speed", "value": speed}
            command_queue.put({"source": "web", **payload})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            body = json.dumps({"ok": False, "error": str(exc)}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b'{"ok":true}'
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass                      # keep the terminal clean


def my_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.255.255.255", 1))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "PI-IP"


def start_web():
    server = ThreadingHTTPServer(("0.0.0.0", WEB_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Ground control dashboard: open http://{my_ip()}:{WEB_PORT} in your laptop browser")


def start_face_recognition():
    """Run face recognition alongside the rover, owned by this brain process."""
    if os.environ.get("ROVER_SIM"):
        return None
    try:
        req = Request(FACE_URL + "/status")
        with urlopen(req, timeout=0.3):
            return None
    except Exception:
        pass
    if not os.path.isfile(FACE_SCRIPT):
        print(f"Face recognition unavailable: {FACE_SCRIPT} was not found")
        return None

    # The face-recognition process owns the camera and publishes its annotated
    # frames for the GCS, so don't open a second preview window on the Pi.
    command = [sys.executable, "-u", FACE_SCRIPT, "--headless",
               "--width", str(FACE_WIDTH), "--height", str(FACE_HEIGHT)]
    if FACE_FRESH_START:
        command.append("--fresh")         # forget faces/photos from earlier runs
    if not FACE_MIRROR:
        command.append("--no-mirror")
    if FACE_DEBUG:
        command.append("--debug")

    try:
        process = subprocess.Popen(
            command,
            cwd=os.path.dirname(FACE_SCRIPT),
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"Could not start face recognition: {exc}")
        return None

    print("Face recognition/camera stream started headless; camera/model errors will appear below.")
    return process


def stop_face_recognition(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


# ---------------- terminal ----------------
MANUAL = {
    "w": rover.forward,
    "s": rover.backward,
    "a": rover.turn_left,
    "d": rover.turn_right,
    "q": rover.left,
    "e": rover.right,
}


def show_menu():
    print(f"""
AUTONOMOUS:  start 3      explore until 3 different people are found, then return by itself
             start 3 10   same, but come home after 10 minutes at the latest
             start        explore with no target (type 'return' when you want it back)
             return | stop | quit
             (a new face in view is walked up to first; battery low also brings it home)
MANUAL (press Enter after each key; Enter alone = stop):
  w forward      s backward
  a spin left    d spin right
  q curve left   e curve right
  speed 60       set motor power 10-100 (now {SPEED})
MPU (gyro):  imu          live yaw / rate / vib printout on or off (shows what to look for)
             zero         set heading to 0
             calibrate    re-measure the gyro zero point (rover must stand still)
  help           show this menu
""")


def terminal_reader():
    while True:
        try:
            cmd = input("> ")
        except (EOFError, KeyboardInterrupt):
            cmd = "quit"
        command_queue.put({"source": "terminal", "cmd": cmd.strip().lower()})
        if cmd.strip().lower() == "quit":
            return


def post_command(item):
    """Put a command item into command_queue (e.g. from tests or scripts)."""
    command_queue.put(item)


def start_mission(target_people=0, limit_min=0.0):
    """Start an autonomous mission (callable from main event loop, web API, or test suites)."""
    global job, stall_recovery_count, dashboard_drive_active
    busy = job is not None and job.is_alive()
    if control_state["estop"]:
        add_event("Start refused: reset the E-stop first")
        return False
    if busy:
        print("Already running")
        return False
    safe_stop()
    if not ensure_link():
        add_event("Start refused: Uno link is unavailable")
        return False
    v0 = battery_volts()
    if v0 is not None and v0 < BATT_RETURN_V:
        msg = f"Start refused: battery too low ({v0:.1f} V, limit {BATT_RETURN_V:.1f} V)"
        print(msg)
        add_event(msg)
        return False
    reset = None
    if target_people > 0:
        st = face_call("/status", timeout=1.0)
        if not st or st.get("frame_age") is None or st["frame_age"] > 3.0:
            msg = "Start refused: camera / face detection is not running, so people cannot be counted"
            print(msg)
            add_event(msg)
            return False
        reset = face_call("/mission/reset", post=True, timeout=1.0)
        if reset is None:
            msg = "Start refused: could not reset the people counter"
            print(msg)
            add_event(msg)
            return False
    now_t = time.time()
    control_state.update(mode="AUTO", state="EXPLORING", stalled=False, space="room")
    stall_recovery_count = 0
    detected_doors.clear()
    mission.update(target=target_people, found=0, people=[], marks=[], exploring=False,
                   cam_bad_since=None, epoch=reset["epoch"] if reset else None,
                   approaching=False, approach_after=0.0, face=None, approach_hit=None,
                   t0=now_t, t_end=None, closed=False, limit_min=limit_min,
                   deadline=(now_t + limit_min * 60.0) if limit_min > 0 else None,
                   return_reason=None, outcome="Running", events=[], batt_start=v0,
                   report_dir=os.path.join(REPORT_DIR, time.strftime("mission_%Y%m%d_%H%M%S", time.localtime(now_t))),
                   report_path=None)
    pose.update(x=0.0, y=0.0, h=90.0)
    live.update(x=0.0, y=0.0, h=90.0)
    path.clear()
    trail_out[:] = [(0.0, 0.0)]
    trail_ret.clear()
    sync_live()
    stop_flag.clear()
    dashboard_drive_active = False
    lim = f", time limit {limit_min:g} min" if limit_min > 0 else ""
    msg = (f"Mission started: find {target_people} people{lim}, then return to the entry point"
           if target_people > 0 else f"Explore mission started (no people target{lim})")
    print(msg)
    add_event(msg)
    job = threading.Thread(target=worker, args=(mission_run,), daemon=True)
    job.start()
    return True


def main():
    global SPEED, dashboard_drive_active, dashboard_drive_time, face_process, job, ser, stall_recovery_count
    rover.pca_init()                 # stops with a clear message if I2C is not usable
    if not open_uno():
        print("WARNING: Uno not found. Manual driving and the dashboard still work;")
        print("         plug the Uno in, then 'start' will retry the connection.")
    start_web()
    face_process = start_face_recognition()
    show_menu()

    if not os.environ.get("ROVER_NO_INPUT"):
        threading.Thread(target=terminal_reader, daemon=True).start()

    try:
        while True:
            if dashboard_drive_active and dashboard_drive_time is not None and time.monotonic() - dashboard_drive_time > 0.4:
                safe_stop()
                dashboard_drive_active = False
                add_event("Manual drive stopped: dashboard heartbeat timed out")

            try:
                item = command_queue.get(timeout=0.08)
            except queue.Empty:
                item = None
            if item is None:
                if face_process is not None and face_process.poll() is not None:
                    add_event(f"Face recognition exited with status {face_process.returncode}")
                    print(f"Face recognition exited with status {face_process.returncode}.")
                    face_process = None
                continue

            target_people = 0
            limit_min = 0.0
            source = item.get("source", "terminal")
            if source == "web":
                action = item.get("action")
                if action == "drive":
                    busy = job is not None and job.is_alive()
                    if control_state["estop"] or control_state["mode"] != "MANUAL" or busy:
                        continue
                    lin, ang = item["lin"], item["ang"]
                    if lin == 0 and ang == 0:
                        safe_stop()
                        dashboard_drive_active = False
                        dashboard_drive_time = None
                    else:
                        left = int(max(-1.0, min(1.0, lin - ang)) * SPEED)
                        right = int(max(-1.0, min(1.0, lin + ang)) * SPEED)
                        safe_motor(rover.drive, left, right)
                        dashboard_drive_active = True
                        dashboard_drive_time = time.monotonic()
                    continue
                if action == "mode":
                    if control_state["estop"]:
                        continue
                    if item["value"] == "MANUAL":
                        if job is not None and job.is_alive():
                            stop_flag.set()
                            job.join()
                        safe_stop()
                        dashboard_drive_active = False
                        control_state.update(mode="MANUAL", state="IDLE")
                        add_event("Manual mode selected")
                    else:
                        if job is None or not job.is_alive():
                            safe_stop()
                            dashboard_drive_active = False
                        control_state["mode"] = "AUTO"
                        add_event("Auto mode selected; press Start mission to move")
                    continue
                if action == "estop":
                    control_state.update(estop=True, mode="MANUAL", state="ESTOP")
                    dashboard_drive_active = False
                    stop_flag.set()
                    safe_stop()
                    if job is not None and job.is_alive():
                        job.join()
                    add_event("Emergency stop latched; motor outputs stopped")
                    continue
                if action == "reset":
                    safe_stop()
                    control_state.update(estop=False, mode="MANUAL", state="IDLE")
                    stop_flag.clear()
                    add_event("Software E-stop reset; manual mode")
                    continue
                if action == "speed":
                    cmd = f"speed {item['value']}"
                elif action == "stop":
                    cmd = "stop"
                else:
                    cmd = action
                    if action == "start":
                        target_people = item.get("people", 0)
                        limit_min = float(item.get("minutes", 0))
            else:
                cmd = item.get("cmd", "")
                parts = cmd.split()
                if len(parts) > 1 and parts[0] == "start":
                    ok = len(parts) <= 3 and parts[1].isdigit() and 0 <= int(parts[1]) <= MAX_PEOPLE
                    minutes = 0.0
                    if ok and len(parts) == 3:
                        try:
                            minutes = float(parts[2])
                            ok = 0 <= minutes <= MAX_MINUTES
                        except ValueError:
                            ok = False
                    if ok:
                        cmd, target_people, limit_min = "start", int(parts[1]), minutes
                    else:
                        print(f"Use:  start   |   start N   |   start N M   (N = people 0-{MAX_PEOPLE}, M = minutes 0-{MAX_MINUTES}; 0 = none)")
                        continue

            if face_process is not None and face_process.poll() is not None:
                print(f"Face recognition exited with status {face_process.returncode}.")
                add_event(f"Face recognition exited with status {face_process.returncode}")
                face_process = None
            busy = job is not None and job.is_alive()

            if cmd == "start":
                start_mission(target_people, limit_min)

            elif cmd == "return":
                if control_state["estop"]:
                    add_event("Return refused: reset the E-stop first")
                    continue
                if busy and phase["name"] == "return":
                    print("Already returning")
                    continue
                if busy:
                    stop_flag.set()
                    job.join()
                safe_stop()
                if not ensure_link():
                    add_event("Return refused: Uno link is unavailable")
                    continue
                stop_flag.clear()
                dashboard_drive_active = False
                if mission["t0"] is not None and not mission["closed"]:
                    mission["t_end"] = None                   # the mission goes on until it is home
                    if mission["return_reason"] is None:
                        mission["return_reason"] = "operator"
                control_state.update(mode="AUTO", state="RETURNING")
                add_event("Returning along recorded path")
                job = threading.Thread(target=worker, args=(go_home,))
                job.start()

            elif cmd in ("stop", ""):
                if busy:
                    stop_flag.set()
                    job.join()
                    control_state.update(mode="MANUAL", state="IDLE")
                    add_event("Autonomous motion stopped")
                else:
                    safe_stop()
                    dashboard_drive_active = False
                    if cmd == "stop":
                        print("Stopped")
                        control_state.update(mode="MANUAL", state="IDLE")
                        add_event("Motors stopped")

            elif cmd in MANUAL:
                if control_state["estop"]:
                    print("E-stop is latched. Reset it from the dashboard first.")
                    continue
                if busy:
                    print("Autonomous is running. Type stop first.")
                    continue
                control_state.update(mode="MANUAL", state="IDLE")
                dashboard_drive_active = False
                if safe_motor(MANUAL[cmd], SPEED):
                    print(f"Manual '{cmd}' at {SPEED}% (Enter = stop)")
                add_event(f"Terminal manual command: {cmd}")

            elif cmd.startswith("speed"):
                if busy:
                    print("Type stop first, then change the speed.")
                    continue
                parts = cmd.split()
                if len(parts) == 2 and parts[1].isdigit() and 10 <= int(parts[1]) <= 100:
                    SPEED = int(parts[1])
                    print(f"Motor power set to {SPEED}%")
                    add_event(f"Motor command speed set to {SPEED}%")
                else:
                    print(f"Speed is {SPEED}%. Use for example: speed 60  (10-100)")

            elif cmd == "imu":
                imu_watch["on"] = not imu_watch["on"]
                if imu_watch["on"]:
                    print(IMU_GUIDE)
                    if ser is None:
                        print("The Uno is not connected yet. Plug it in, then type  start  once or restart brain.py.")
                else:
                    print("MPU printout off")

            elif cmd == "zero":
                if imu_command("Z"):
                    print("Heading set to 0")

            elif cmd == "calibrate":
                if busy:
                    print("Type stop first. The rover must stand still while the gyro is measured.")
                elif imu_command("C"):
                    print("Measuring the gyro zero point: keep the rover still for about 3 seconds")
                    add_event("Gyro zero point re-measured")

            elif cmd in ("help", "h", "?"):
                show_menu()

            elif cmd == "quit":
                break

            else:
                print("Unknown command. Type help")
    finally:
        imu_watch["on"] = False
        stop_flag.set()
        if job is not None:
            job.join()
        safe_stop()
        stop_face_recognition(face_process)
        try:
            if ser is not None:
                ser.write(b"X\n")
                ser.close()
        except (serial.SerialException, OSError):
            pass
        print("Bye")


if __name__ == "__main__":
    main()
