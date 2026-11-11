#!/usr/bin/env python3
"""
Face detection + recognition with automatic registration for Raspberry Pi 5 Ubuntu.

  - YuNet : detects faces
  - SFace : turns each face into a 128-number embedding so people can be told apart

New faces are registered as "Person 1", "Person 2", ... and remembered between runs
in faces_db.npz. Model files are downloaded automatically on first run.

Furniture / door detection (optional, runs in its own thread so faces stay fast):
  - a small YOLO model (yolov8n) finds chairs, tables, sofas, benches, beds, plants, TVs ...
  - the standard model has NO "door" class. If you have a YOLO model trained on doors, pass it with
    --obj-model my_doors.pt and it is used automatically (the class name must be "door").
  - what it sees is sent to brain.py in /status ("objects") and drawn in blue on the camera stream.

Install (once):
  pip install opencv-python numpy        # OpenCV 4.8 or newer
  pip install ultralytics --break-system-packages    # only for furniture detection (large: it brings PyTorch)
  Run python3 face_recog.py once with internet so yolov8n.pt is downloaded next to this script.
  Without ultralytics the program still works: faces only, object detection stays off.

Usage:
  python3 face_recog.py                         # live, auto-registers people
  python3 face_recog.py --debug                 # show match scores and registration progress
  python3 face_recog.py --camera 1              # select another camera
  python3 face_recog.py --no-auto               # recognize only, never register
  python3 face_recog.py --rename "Person 1" Rahul
  python3 face_recog.py --reset
  python3 face_recog.py --enroll Alic

Press 'q' in the window (or Ctrl+C) to quit.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
import os
import re
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DET_MODEL = os.path.join(HERE, "face_detection_yunet_2023mar.onnx")
REC_MODEL = os.path.join(HERE, "face_recognition_sface_2021dec.onnx")
DB_FILE = os.path.join(HERE, "faces_db.npz")
FACES_DIR = os.path.join(HERE, "known_faces")
PHOTOS_DIR = os.path.join(HERE, "mission_photos")   # one close-up per person, taken when they are counted
OBJ_MODEL = os.path.join(HERE, "yolov8n.pt")
OBJ_CLASSES = "chair,couch,bench,bed,dining table,potted plant,tv,refrigerator,door"
STREAM_HOST = "127.0.0.1"
STREAM_PORT = 8765
stream_lock = threading.Lock()
latest_stream_jpeg = None
last_frame_time = 0.0

# People counted in the CURRENT mission. brain.py asks for this over the loopback
# interface below and resets it every time a mission starts.
people_lock = threading.Lock()
mission_seen = {}          # name -> frames in which that person was clearly identified
mission_epoch = 0          # increases on every reset, so brain.py can ignore stale answers
COUNT_FRAMES = 3           # a person counts after this many clear frames (set from --count-frames)
visible_targets = []       # faces in view that are NOT counted yet (brain.py walks up to these)
targets_streak = 0         # consecutive frames that had at least one such face
person_files = {}          # name -> {"photo": close-up when counted, "snapshot": first scene photo}
snap_last = {}             # name -> time of that person's last snapshot

# Furniture / doors seen by the object detector (filled by ObjectDetector, read by status_dict)
objects_lock = threading.Lock()
visible_objects = []       # [{"label", "conf", "x" (-1 left .. +1 right), "w", "h" (fractions of the frame)}]
objects_info = {"t": 0.0, "boxes": [], "enabled": False}

MODEL_URLS = {
    DET_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    REC_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

POSES = [
    "Look straight at the camera",
    "Turn your head slightly LEFT",
    "Turn your head slightly RIGHT",
    "Tilt your head slightly UP",
    "Tilt your head slightly DOWN",
]

GREEN, RED, YELLOW, CYAN, ORANGE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (0, 165, 255)
BLUE = (255, 120, 0)      # BGR: furniture / doors


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------
def ensure_models():
    """Download the two ONNX models if they are not next to the script."""
    for path, url in MODEL_URLS.items():
        if os.path.isfile(path) and os.path.getsize(path) > 100_000:
            continue
        print(f"Downloading {os.path.basename(path)} ...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as ex:
            if os.path.isfile(path):
                os.remove(path)
            raise SystemExit(
                f"Download failed ({ex}).\nDownload it manually from:\n  {url}\n"
                f"and save it as:\n  {path}"
            )


def check_requirements():
    if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
        raise SystemExit(
            f"This OpenCV ({cv2.__version__}) is too old. Need 4.8 or newer.\n"
            "Upgrade with: pip install --upgrade opencv-python"
        )
    ensure_models()


def open_camera(args):
    """Open a laptop webcam using the right backend for the OS."""
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    cap = None
    for be in backends:
        c = cv2.VideoCapture(args.camera, be)
        if c.isOpened():
            cap = c
            break
        c.release()
    if cap is None:
        raise SystemExit(
            f"Cannot open camera {args.camera}. Close other apps using the webcam "
            "(Zoom, Teams...), allow camera access for your terminal/IDE in the OS "
            "privacy settings, or try --camera 1."
        )

    if sys.platform.startswith("linux"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # laptop webcams often return a few black/empty frames while auto-exposure settles
    frame = None
    for _ in range(30):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    if frame is None:
        raise SystemExit("Camera opened but returned no frames.")
    h, w = frame.shape[:2]
    print(f"Camera running at {w}x{h}")
    return cap


def read_frame(cap, args):
    ok, frame = cap.read()
    if ok and args.mirror:
        frame = cv2.flip(frame, 1)
    return ok, frame


def load_db():
    if not os.path.isfile(DB_FILE):
        return {}
    data = np.load(DB_FILE)
    return {k: data[k].astype(np.float32) for k in data.files}


def save_db(db):
    np.savez(DB_FILE, **db)


def face_file(name):
    return os.path.join(FACES_DIR, name.replace(" ", "_") + ".jpg")


def do_reset():
    if os.path.isfile(DB_FILE):
        os.remove(DB_FILE)
    if os.path.isdir(FACES_DIR):
        for f in os.listdir(FACES_DIR):
            if f.lower().endswith(".jpg"):
                os.remove(os.path.join(FACES_DIR, f))
    print("All registered faces deleted.")


def do_rename(old, new):
    old, new = old.strip(), new.strip()
    db = load_db()
    if old not in db:
        raise SystemExit(f"'{old}' not found. Registered: {', '.join(db) or 'nobody'}")
    if new in db:
        raise SystemExit(f"'{new}' already exists.")
    db[new] = db.pop(old)
    save_db(db)
    if os.path.isfile(face_file(old)):
        os.rename(face_file(old), face_file(new))
    print(f"Renamed '{old}' -> '{new}'")


def load_profile_cascade():
    # opencv-python ships the Haar files; cv2.data works on every OS
    p = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
    if os.path.isfile(p):
        c = cv2.CascadeClassifier(p)
        if not c.empty():
            return c
    print("Profile cascade not found; --profile disabled.")
    return None


# ----------------------------------------------------------------------------
# Detection / recognition
# ----------------------------------------------------------------------------
def detect_faces(detector, frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    return faces if faces is not None else np.empty((0, 15), dtype=np.float32)


def detect_profiles(cascade, frame, scale=0.5):
    """Haar profile detector on a half-size gray image, checking left and right profiles."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale)
    gw = gray.shape[1]
    boxes = []
    for flip in (False, True):
        g = cv2.flip(gray, 1) if flip else gray
        rects = cascade.detectMultiScale(g, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        for (x, y, w, h) in rects:
            if flip:
                x = gw - x - w
            boxes.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))
    return boxes


def center_inside(box, boxes):
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    for (x, y, w, h) in boxes:
        if x <= cx <= x + w and y <= cy <= y + h:
            return True
    return False


def embed(recognizer, frame, face):
    return recognizer.feature(recognizer.alignCrop(frame, face)).flatten()


def person_scores(feat, db):
    """Cosine similarity of one face against every registered person (best sample each)."""
    f = feat / (np.linalg.norm(feat) + 1e-9)
    out = {}
    for name, feats in db.items():
        r = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        out[name] = float(np.max(r @ f))
    return out


def best_match(feat, db, exclude=()):
    best_name, best = None, -1.0
    for name, s in person_scores(feat, db).items():
        if name not in exclude and s > best:
            best_name, best = name, s
    return best_name, best


def next_person_name(db):
    nums = [int(m.group(1)) for k in db if (m := re.match(r"^Person (\d+)$", k))]
    return f"Person {max(nums, default=0) + 1}"


def save_crop(frame, box, name):
    x, y, w, h = box
    m = int(0.2 * max(w, h))
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, frame.shape[1]), min(y + h + m, frame.shape[0])
    os.makedirs(FACES_DIR, exist_ok=True)
    cv2.imwrite(face_file(name), frame[y0:y1, x0:x1])


def prune_snapshots(outdir, keep):
    """Keep only the newest `keep` snapshots (names start with a timestamp) so the SD card never fills."""
    if keep <= 0:
        return
    files = sorted(f for f in os.listdir(outdir) if f.lower().endswith(".jpg"))
    for f in files[:-keep]:
        try:
            os.remove(os.path.join(outdir, f))
        except OSError:
            pass


def clear_snapshots(outdir):
    if os.path.isdir(outdir):
        for f in os.listdir(outdir):
            if f.lower().endswith(".jpg"):
                try:
                    os.remove(os.path.join(outdir, f))
                except OSError:
                    pass


def save_mission_photo(frame, box, name):
    x, y, w, h = box
    m = int(0.35 * max(w, h))
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, frame.shape[1]), min(y + h + m, frame.shape[0])
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    path = os.path.join(PHOTOS_DIR, name.replace(" ", "_") + ".jpg")
    cv2.imwrite(path, frame[y0:y1, x0:x1])
    return path


def mark_seen(name, frame=None, box=None):
    with people_lock:
        n = mission_seen.get(name, 0) + 1
        mission_seen[name] = n
    if n == COUNT_FRAMES and frame is not None:      # the moment this person counts: keep a close-up
        path = save_mission_photo(frame, box, name)
        with people_lock:
            person_files.setdefault(name, {})["photo"] = os.path.abspath(path)


def counted_names():
    with people_lock:
        return {n for n, c in mission_seen.items() if c >= COUNT_FRAMES}


def set_targets(cands):
    global visible_targets, targets_streak
    with people_lock:
        visible_targets = cands
        targets_streak = targets_streak + 1 if cands else 0


def people_dict():
    with people_lock:
        out = {n: dict(f) for n, f in person_files.items()}
    for n in out:
        crop = face_file(n)
        if os.path.isfile(crop):
            out[n]["crop"] = os.path.abspath(crop)
    return out


def reset_mission_count():
    global mission_epoch
    with people_lock:
        mission_seen.clear()
        person_files.clear()
        snap_last.clear()
        mission_epoch += 1
        return mission_epoch


def status_dict():
    with people_lock:
        people = sorted(n for n, c in mission_seen.items() if c >= COUNT_FRAMES)
        epoch = mission_epoch
        targets = list(visible_targets)
        streak = targets_streak
    age = None if last_frame_time == 0.0 else round(time.time() - last_frame_time, 2)
    with objects_lock:
        objects = list(visible_objects)
        obj_age = None if objects_info["t"] == 0.0 else round(time.time() - objects_info["t"], 2)
        obj_on = objects_info["enabled"]
    return {"count": len(people), "people": people, "epoch": epoch, "frame_age": age,
            "targets": targets, "streak": streak,
            "objects": objects, "objects_age": obj_age, "objects_enabled": obj_on}


def put_label(frame, text, x, y, color):
    cv2.putText(frame, text, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class StreamHandler(BaseHTTPRequestHandler):
    """Serve annotated frames to brain.py on loopback; the dashboard uses brain.py as proxy."""
    def _json(self, data):
        try:
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_POST(self):
        if self.path.split("?", 1)[0] == "/mission/reset":
            epoch = reset_mission_count()
            print(f"Mission count reset (epoch {epoch})")
            self._json({"ok": True, "epoch": epoch, "count": 0})
        else:
            self.send_error(404)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/status":
            self._json(status_dict())
            return
        if route == "/people":
            self._json(people_dict())
            return
        if route != "/video":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        try:
            while True:
                with stream_lock:
                    jpg = latest_stream_jpeg
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
                time.sleep(0.08)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, *args):
        pass


def publish_stream_frame(frame):
    global latest_stream_jpeg, last_frame_time
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if ok:
        with stream_lock:
            latest_stream_jpeg = encoded.tobytes()
        last_frame_time = time.time()


# ----------------------------------------------------------------------------
# Furniture / door detector (own thread, about once a second)
# ----------------------------------------------------------------------------
class ObjectDetector:
    """Runs a YOLO model on every Nth camera frame in a background thread, so the face
    detector never waits for it. If ultralytics or the model is missing it just stays off."""

    def __init__(self, model_path, wanted, conf, imgsz, interval, mirror):
        self.model_path = model_path
        self.wanted = {w.strip().lower() for w in wanted.split(",") if w.strip()}
        self.conf, self.imgsz, self.interval, self.mirror = conf, imgsz, interval, mirror
        self.model, self.ids = None, []
        self.ready = False
        self.pending = None
        self.last_submit = 0.0
        self.busy = threading.Event()
        self.wake = threading.Event()
        self.warned = False

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _load(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Object detection is OFF: ultralytics is not installed.\n"
                  "  pip install ultralytics --break-system-packages")
            return False
        try:
            self.model = YOLO(self.model_path)      # yolov8n.pt is downloaded on the first run
        except Exception as ex:
            print(f"Object detection is OFF: could not load {self.model_path} ({ex})")
            return False
        names = {int(i): str(n).lower() for i, n in self.model.names.items()}
        self.ids = [i for i, n in names.items() if n in self.wanted]
        if not self.ids:
            print("Object detection is OFF: this model knows none of: " + ", ".join(sorted(self.wanted)))
            return False
        missing = sorted(self.wanted - set(names.values()))
        if missing:
            print("Object detection note: this model has no class for: " + ", ".join(missing))
        try:
            import torch
            torch.set_num_threads(2)                # leave the other CPU cores to the face detector and brain.py
        except Exception:
            pass
        return True

    def submit(self, frame):
        """Called for every camera frame; only keeps a copy when it is time for a new detection."""
        now = time.time()
        if not self.ready or self.busy.is_set() or now - self.last_submit < self.interval:
            return
        self.last_submit = now
        self.pending = frame.copy()
        self.wake.set()

    def _loop(self):
        self.ready = self._load()
        if not self.ready:
            return
        with objects_lock:
            objects_info["enabled"] = True
        print("Object detection ready: " + ", ".join(sorted(self.model.names[i] for i in self.ids)))
        while True:
            self.wake.wait()
            self.wake.clear()
            frame, self.pending = self.pending, None
            if frame is None:
                continue
            self.busy.set()
            try:
                self._detect(frame)
            except Exception as ex:
                if not self.warned:
                    self.warned = True
                    print(f"Object detection error (shown once): {ex}")
            finally:
                self.busy.clear()

    def _detect(self, frame):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=self.ids, verbose=False)[0]
        fh, fw = frame.shape[:2]
        objs, boxes = [], []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            label = str(res.names[int(b.cls[0])])
            conf = float(b.conf[0])
            off = ((x1 + x2) / 2.0 - fw / 2.0) / (fw / 2.0)              # -1 = far left ... +1 = far right (as drawn)
            objs.append({"label": label, "conf": round(conf, 2),
                         "x": round(-off if self.mirror else off, 3),    # same sign rule as the faces: + = really to the right
                         "w": round((x2 - x1) / fw, 3), "h": round((y2 - y1) / fh, 3)})
            boxes.append((int(x1), int(y1), int(x2), int(y2), f"{label} {conf:.2f}"))
        objs.sort(key=lambda o: -o["w"])                                 # biggest (closest) first
        with objects_lock:
            visible_objects[:] = objs
            objects_info["t"] = time.time()
            objects_info["boxes"] = boxes


# ----------------------------------------------------------------------------
# Face tracker (keeps a face's identity between frames, works with many faces)
# ----------------------------------------------------------------------------
class Entry:
    """One detected face in the current frame."""

    def __init__(self, face):
        self.face = face
        self.x, self.y, self.w, self.h = (int(v) for v in face[:4])
        self.det = float(face[-1])
        self.track = None
        self.feat = None
        self.scores = {}
        self.match = None


class Track:
    def __init__(self, now):
        self.box = (0, 0, 0, 0)
        self.name = None
        self.score = 0.0
        self.pending = []      # embeddings of a not-yet-registered face
        self.need = 0          # samples needed before registering
        self.since_id = 0      # frames since last identification
        self.last_seen = now
        self.last_add = 0.0    # last time we added a sample to a known person


def assign_tracks(tracks, entries, now):
    """Match faces to existing tracks globally (closest pairs first), create tracks for the rest."""
    pairs = []
    for ei, e in enumerate(entries):
        cx, cy = e.x + e.w / 2, e.y + e.h / 2
        for ti, t in enumerate(tracks):
            tx, ty, tw, th = t.box
            d = math.hypot(tx + tw / 2 - cx, ty + th / 2 - cy)
            if d < 0.75 * max(e.w, tw):
                pairs.append((d, ei, ti))
    pairs.sort()
    used_e, used_t = set(), set()
    for d, ei, ti in pairs:
        if ei in used_e or ti in used_t:
            continue
        entries[ei].track = tracks[ti]
        used_e.add(ei)
        used_t.add(ti)
    for ei, e in enumerate(entries):
        if ei not in used_e:
            t = Track(now)
            tracks.append(t)
            e.track = t
        t = e.track
        t.box = (e.x, e.y, e.w, e.h)
        t.last_seen = now
        t.since_id += 1


def register(args, db, frame, box, pending, exclude):
    """Create 'Person N' from a confirmed unknown face. Returns the name, or None."""
    mean = np.mean(pending, axis=0)
    m = mean / (np.linalg.norm(mean) + 1e-9)
    # every sample must agree with the others (guards against the track jumping between people)
    for p in pending:
        if float(np.dot(p / (np.linalg.norm(p) + 1e-9), m)) < args.threshold:
            return None
    # last duplicate check using the averaged face (people visible right now are excluded)
    name, score = best_match(mean, db, exclude)
    if name is not None and score >= args.threshold:
        return name
    new_name = next_person_name(db)
    db[new_name] = np.array(pending, dtype=np.float32)
    save_crop(frame, box, new_name)
    save_db(db)
    print(f"New person registered: {new_name}")
    return new_name


# ----------------------------------------------------------------------------
# Manual enrollment (optional)
# ----------------------------------------------------------------------------
def enroll(args, detector, recognizer, cap):
    name = re.sub(r"[^A-Za-z0-9_-]", "_", args.enroll.strip())
    if not name:
        raise SystemExit("Please give a name, e.g. --enroll Alice")

    per_pose = max(2, args.samples // len(POSES))
    samples = []
    print(f"\nEnrolling '{name}'. Sit about 50-100 cm from the camera. Follow the prompts.")

    for pose in POSES:
        print(f"\n>> {pose}")
        grace_end = time.time() + 3.0
        deadline = grace_end + 30.0
        last_cap, got = 0.0, 0
        while got < per_pose:
            ok, frame = read_frame(cap, args)
            if not ok:
                raise SystemExit("Camera read failed")
            faces = detect_faces(detector, frame)
            now = time.time()
            if now > deadline:
                print("   (no clear face for 30 s, skipping this pose)")
                break
            if (now >= grace_end and now - last_cap >= 0.5 and len(faces) == 1
                    and faces[0][-1] >= 0.8 and faces[0][2] >= args.min_enroll_width):
                samples.append(embed(recognizer, frame, faces[0]))
                got += 1
                last_cap = now
                print(f"   captured {got}/{per_pose}")
            if not args.headless:
                for f in faces:
                    x, y, w, h = f[:4].astype(int)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
                cv2.putText(frame, f"{pose} ({got}/{per_pose})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
                cv2.imshow("Enroll", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise SystemExit("Enrollment cancelled")

    if not samples:
        raise SystemExit("No samples captured. Move closer / improve lighting and try again.")

    db = load_db()
    new = np.array(samples, dtype=np.float32)
    db[name] = np.vstack([db[name], new]) if name in db else new
    save_db(db)
    print(f"\nSaved {len(samples)} samples for '{name}'. Registered: {', '.join(db)}")


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run(args, detector, recognizer, cap):
    global COUNT_FRAMES
    COUNT_FRAMES = max(1, args.count_frames)
    db = load_db()
    if db:
        print("Registered people:", ", ".join(db), f"(loaded from {DB_FILE}; --reset deletes them)")
    else:
        print("Nobody registered yet." + (" New faces will be added as Person 1, 2, 3..." if args.auto else ""))

    profile_cascade = load_profile_cascade() if args.profile else None
    if args.headless:
        os.makedirs(args.outdir, exist_ok=True)

    def dbg(e, decision):
        if args.debug:
            top = sorted(e.scores.items(), key=lambda kv: -kv[1])[:3]
            txt = " ".join(f"{n}={s:.2f}" for n, s in top) or "nobody registered"
            print(f"[debug] face {e.w}px det={e.det:.2f} | {txt} | {decision}")

    tracks = []
    dirty = False
    last_save_db = time.time()
    # (snap_last is module-level so a new mission can reset it)
    prev, fps = time.time(), 0.0
    add_min = args.threshold + 0.05   # a learning sample must be confident...
    add_max = 0.75                    # ...but not a near-copy of what we already have
    stream_server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), StreamHandler)
    stream_server.daemon_threads = True
    threading.Thread(target=stream_server.serve_forever, daemon=True).start()
    print(f"Camera stream ready on local port {STREAM_PORT}")
    obj_det = None
    if args.objects:
        obj_det = ObjectDetector(args.obj_model, args.obj_classes, args.obj_conf, args.obj_size,
                                 args.obj_interval, args.mirror)
        obj_det.start()                    # loads the model in its own thread; faces start immediately

    bad_reads = 0
    try:
        while True:
            ok, frame = read_frame(cap, args)
            if not ok:
                # a USB camera on a moving rover drops frames now and then
                bad_reads += 1
                if bad_reads > 50:
                    print("Camera stopped delivering frames; exiting")
                    break
                time.sleep(0.1)
                continue
            bad_reads = 0

            now = time.time()
            if obj_det is not None:
                obj_det.submit(frame)                      # a copy of the CLEAN frame, before anything is drawn on it
            faces = detect_faces(detector, frame)
            tracks[:] = [t for t in tracks if now - t.last_seen < 1.0]

            entries = [Entry(f) for f in faces]
            big = [e for e in entries if e.w >= args.min_rec_width]
            assign_tracks(tracks, big, now)

            # IDs held by faces we are NOT re-identifying this frame are already taken
            claimed = {e.track.name for e in big
                       if e.track.name and e.track.since_id < args.recheck}
            due = [e for e in big if e.track.name is None or e.track.since_id >= args.recheck]

            for e in due:
                e.track.since_id = 0
                e.feat = embed(recognizer, frame, e.face)
                e.scores = person_scores(e.feat, db)

            # One-to-one assignment, best score first: an ID can belong to only ONE face per frame.
            cands = sorted(((s, i, n) for i, e in enumerate(due)
                            for n, s in e.scores.items() if s >= args.threshold), reverse=True)
            done = set()
            for s, i, n in cands:
                if i in done or n in claimed:
                    continue
                due[i].match = (n, s)
                done.add(i)
                claimed.add(n)

            for e in due:
                t = e.track
                if e.match:
                    name, s = e.match
                    t.name, t.score, t.pending = name, s, []
                    second = max((v for n, v in e.scores.items() if n != name), default=-1.0)
                    # learn new poses/lighting only when the match is clear-cut (avoids mixing people)
                    if (args.auto and e.det >= 0.8 and e.w >= 80 and add_min <= s <= add_max
                            and second < args.threshold and now - t.last_add >= 2.0):
                        arr = np.vstack([db[name], e.feat.reshape(1, -1)])
                        if len(arr) > args.max_samples:
                            arr = np.vstack([arr[:6], arr[-(args.max_samples - 6):]])
                        db[name] = arr
                        t.last_add = now
                        dirty = True
                    dbg(e, f"-> {name}")
                else:
                    # compare only against people NOT already seen in this frame
                    rest = max((v for n, v in e.scores.items() if n not in claimed), default=-1.0)
                    t.name, t.score = None, max(rest, 0.0)
                    clearly_new = rest < args.threshold - args.margin
                    good = e.w >= args.min_register_width and e.det >= args.min_register_det
                    if args.auto and good:
                        # grey-zone faces are registered too, but only after being seen longer
                        t.need = args.confirm if clearly_new else args.borderline_confirm
                        t.pending.append(e.feat)
                        t.pending = t.pending[-60:]
                        if len(t.pending) >= t.need:
                            new_name = register(args, db, frame, (e.x, e.y, e.w, e.h), t.pending, claimed)
                            t.pending = []
                            if new_name:
                                t.name, t.score = new_name, 1.0
                                claimed.add(new_name)
                        dbg(e, f"-> unknown ({'new' if clearly_new else 'grey zone'}, "
                               f"{len(t.pending)}/{t.need})" if not t.name else f"-> {t.name} (just registered)")
                    else:
                        t.pending = []
                        dbg(e, "-> unknown (too small / low quality, or --no-auto)")

            # mission count: an identified person counts once the face is big and clear
            for e in big:
                if (e.track.name and e.w >= args.min_register_width
                        and e.det >= args.min_register_det):
                    mark_seen(e.track.name, frame, (e.x, e.y, e.w, e.h))

            # faces the rover should walk up to: detected but not counted yet (also tiny, far-away ones)
            fw = frame.shape[1]
            counted = counted_names()
            want = []
            for e in entries:
                if e.det < args.approach_det:
                    continue
                name = e.track.name if e.track is not None else None
                if name and name in counted:
                    continue
                off = ((e.x + e.w / 2.0) - fw / 2.0) / (fw / 2.0)      # -1 = far left ... +1 = far right
                want.append({"x": round(-off if args.mirror else off, 3),
                              "w": round(e.w / float(fw), 3), "det": round(e.det, 2), "name": name})
            set_targets(want)

            # build what to draw
            results = []   # (x, y, w, h, label, color, name)
            for e in entries:
                if e.w < args.min_rec_width:
                    results.append((e.x, e.y, e.w, e.h, f"Face {e.w}px (too small)", YELLOW, "Far"))
                    continue
                t = e.track
                if t.name:
                    results.append((e.x, e.y, e.w, e.h, f"{t.name} {t.score:.2f}", GREEN, t.name))
                elif t.pending:
                    results.append((e.x, e.y, e.w, e.h, f"New person... {len(t.pending)}/{t.need}", ORANGE, "New"))
                elif args.auto and (e.w < args.min_register_width or e.det < args.min_register_det):
                    results.append((e.x, e.y, e.w, e.h, f"Unknown - come closer ({e.w}px)", RED, "Unknown"))
                else:
                    results.append((e.x, e.y, e.w, e.h, "Unknown", RED, "Unknown"))

            if profile_cascade is not None:
                face_boxes = [r[:4] for r in results]
                for box in detect_profiles(profile_cascade, frame):
                    if not center_inside(box, face_boxes):
                        results.append((*box, "Side face", CYAN, "Side"))

            for (x, y, w, h, label, color, _) in results:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                put_label(frame, label, x, y, color)

            if obj_det is not None:
                with objects_lock:
                    obj_boxes, obj_t = list(objects_info["boxes"]), objects_info["t"]
                if time.time() - obj_t < 3.0:              # do not draw old boxes if the detector stalls
                    for (x1, y1, x2, y2, label) in obj_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 2)
                        put_label(frame, label, x1, y1, BLUE)

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(frame, f"Faces: {len(results)}  People: {len(db)}  FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            publish_stream_frame(frame)

            if dirty and now - last_save_db > 60:
                save_db(db)
                dirty, last_save_db = False, now

            if args.headless:
                # One photo per PERSON, not one per second: a person is photographed when first
                # identified/registered, and again only after --cooldown seconds.
                if args.snapshots:
                    fresh = []
                    for r in results:
                        name = r[6]
                        if name in ("Far", "Unknown", "New", "Side"):     # not an identified person
                            continue
                        if name not in fresh and now - snap_last.get(name, -1e9) >= args.cooldown:
                            fresh.append(name)
                            snap_last[name] = now
                    if fresh:
                        tag = "_".join(n.replace(" ", "") for n in fresh)
                        path = os.path.join(args.outdir, time.strftime("face_%Y%m%d_%H%M%S_") + tag + ".jpg")
                        cv2.imwrite(path, frame)
                        with people_lock:
                            for n in fresh:
                                person_files.setdefault(n, {}).setdefault("snapshot", os.path.abspath(path))
                        print(f"{', '.join(fresh)} -> saved {path}")
                        prune_snapshots(args.outdir, args.max_snapshots)
            else:
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if dirty:
            save_db(db)
        stream_server.shutdown()
        stream_server.server_close()


def parse_args():
    p = argparse.ArgumentParser(description="Face detection + automatic recognition (Raspberry Pi 5)")
    p.add_argument("--enroll", metavar="NAME", help="manually register a person by name")
    p.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"), help='e.g. --rename "Person 1" Rahul')
    p.add_argument("--reset", action="store_true", help="delete all registered faces")
    p.add_argument("--fresh", action="store_true",
                   help="delete all registered faces AND old snapshots, then run (brain.py uses this "
                        "so every start of brain.py begins with no remembered people)")
    p.add_argument("--approach-det", type=float, default=0.75,
                   help="minimum detector confidence for a face to be offered to brain.py as a person to walk up to")
    p.add_argument("--count-frames", type=int, default=3,
                   help="a person counts toward the mission after this many clearly identified frames")
    p.add_argument("--no-auto", dest="auto", action="store_false",
                   help="never register new people automatically")
    p.add_argument("--no-mirror", dest="mirror", action="store_false",
                   help="do not flip the image horizontally (default is a mirror view)")
    p.add_argument("--debug", action="store_true", help="print match scores for each identified face")
    p.add_argument("--camera", type=int, default=0, help="camera index (0 = built-in webcam)")
    p.add_argument("--width", type=int, default=1280,
                   help="requested camera width (falls back to what the camera supports)")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--score", type=float, default=0.5,
                   help="detection confidence (lower = detects more/farther faces, more false positives)")
    p.add_argument("--threshold", type=float, default=0.363,
                   help="same-person cosine threshold (higher = stricter)")
    p.add_argument("--margin", type=float, default=0.08,
                   help="a face scoring below threshold minus margin is 'clearly new' and registers fast "
                        "(raise this if one person gets several IDs)")
    p.add_argument("--confirm", type=int, default=6,
                   help="frames a clearly-new face must be seen before it is registered")
    p.add_argument("--borderline-confirm", type=int, default=24,
                   help="frames a grey-zone face must be seen before it is registered")
    p.add_argument("--recheck", type=int, default=10,
                   help="re-identify a tracked face every N frames (saves CPU)")
    p.add_argument("--min-rec-width", type=int, default=40,
                   help="smallest face width (px) we try to identify")
    p.add_argument("--min-register-width", type=int, default=60,
                   help="smallest face width (px) allowed to create a new person")
    p.add_argument("--min-register-det", type=float, default=0.8,
                   help="minimum detector confidence to register a new person")
    p.add_argument("--min-enroll-width", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=40, help="max stored samples per person")
    p.add_argument("--samples", type=int, default=20, help="manual enrollment samples in total")
    p.add_argument("--no-objects", dest="objects", action="store_false",
                   help="do not look for furniture / doors (faces only)")
    p.add_argument("--obj-model", default=OBJ_MODEL,
                   help="YOLO model file. Default yolov8n.pt (no door class). Use your own .pt trained on doors to detect doors")
    p.add_argument("--obj-classes", default=OBJ_CLASSES,
                   help="comma separated class names to report")
    p.add_argument("--obj-conf", type=float, default=0.4, help="object detection confidence")
    p.add_argument("--obj-size", type=int, default=320, help="object detection image size (320 is fast on the Pi)")
    p.add_argument("--obj-interval", type=float, default=1.0, help="seconds between object detections")
    p.add_argument("--profile", action="store_true", help="also detect full side-view faces")
    p.add_argument("--headless", action="store_true", help="no window; save snapshots")
    p.add_argument("--outdir", default="snapshots")
    p.add_argument("--cooldown", type=float, default=30.0,
                   help="headless: seconds before the SAME person is photographed again")
    p.add_argument("--no-snapshots", dest="snapshots", action="store_false",
                   help="headless: do not save photos at all (the live stream still works)")
    p.add_argument("--max-snapshots", type=int, default=300,
                   help="headless: keep only the newest N snapshots")
    return p.parse_args()


def main():
    args = parse_args()

    if args.reset:
        do_reset()
        return
    if args.rename:
        do_rename(*args.rename)
        return

    if args.fresh:
        do_reset()
        clear_snapshots(args.outdir)
        clear_snapshots(PHOTOS_DIR)
        print("Fresh start: previous faces and snapshots removed.")

    check_requirements()
    detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320), args.score, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(REC_MODEL, "")
    cap = open_camera(args)

    try:
        if args.enroll:
            enroll(args, detector, recognizer, cap)
        else:
            run(args, detector, recognizer, cap)
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
