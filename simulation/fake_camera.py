"""
Fake camera and face recognition server (fake_camera.py).
Serves HTTP endpoints on 127.0.0.1:8765 (/status, /people, /mission/reset, /video)
to simulate face and object detections based on line-of-sight raycasting in SimEnvironment.
"""

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeCameraHandler(BaseHTTPRequestHandler):
    server_instance = None

    def _json(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?", 1)[0] == "/mission/reset":
            srv = self.server_instance
            srv.epoch += 1
            srv.found_people.clear()
            srv.seen_count.clear()
            srv.streak = 0
            self._json({"ok": True, "epoch": srv.epoch, "count": 0})
        else:
            self.send_error(404)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        srv = self.server_instance
        if route == "/status":
            self._json(srv.get_status())
        elif route == "/people":
            self._json({name: {"photo": None, "snapshot": None} for name in srv.found_people})
        elif route == "/video":
            # Simple dummy MJPEG frame
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                # 1x1 dummy jpeg byte
                dummy = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + dummy + b"\r\n")
            except Exception:
                pass
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


class FakeCameraServer:
    def __init__(self, env, host="127.0.0.1", port=8765):
        self.env = env
        self.host = host
        self.port = port
        self.epoch = 1
        self.found_people = []
        self.seen_count = {}
        self.streak = 0
        self.server = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        FakeCameraHandler.server_instance = self
        self.server = ThreadingHTTPServer((self.host, self.port), FakeCameraHandler)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass

    def _run(self):
        try:
            self.server.serve_forever()
        except Exception:
            pass

    def get_status(self):
        with self.lock:
            # Query visible people from simulation environment
            visible = self.env.get_visible_people(hfov=60.0, max_dist=350.0)
            targets = []
            objects = []

            for p in visible:
                name = p["name"]
                if name not in self.found_people:
                    targets.append({"w": p["w"], "x": p["x"], "h": p["h"]})
                    self.seen_count[name] = self.seen_count.get(name, 0) + 1
                    # If approached closely and observed
                    if p["dist"] < 80.0 and self.seen_count[name] >= 2:
                        self.found_people.append(name)

            if targets:
                self.streak += 1
            else:
                self.streak = 0

            # Check if any doors in view for object detection
            for d in self.env.doors:
                dx = d["x"] - self.env.x
                dy = d["y"] - self.env.y
                dist = math.hypot(dx, dy)
                if dist < 250.0:
                    ang = math.degrees(math.atan2(dy, dx))
                    rel = (ang - self.env.theta + 180) % 360 - 180
                    if abs(rel) <= 30.0:
                        blocked = False
                        for p1, p2 in self.env.walls:
                            from simulation.sim_env import line_intersection
                            res = line_intersection(p1, p2, (self.env.x, self.env.y), (d["x"], d["y"]))
                            if res is not None:
                                pt, blk_d = res
                                if blk_d < dist - 20.0:
                                    blocked = True
                                    break
                        if not blocked:
                            objects.append({
                                "label": "door",
                                "conf": 0.88,
                                "x": round(rel / 30.0, 2),
                                "w": round(80.0 / dist, 2),
                                "h": round(160.0 / dist, 2)
                            })

            return {
                "count": len(self.found_people),
                "people": list(self.found_people),
                "epoch": self.epoch,
                "frame_age": 0.05,
                "targets": targets,
                "streak": self.streak,
                "objects": objects,
                "objects_age": 0.05,
                "objects_enabled": True
            }
