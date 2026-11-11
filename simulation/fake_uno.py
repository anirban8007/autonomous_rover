"""
Fake Arduino Uno serial emulator.
Listens on a local TCP socket and emulates the Uno firmware (uno_two_sensors.ino)
at 115200 baud, 50 Hz IMU reporting, wide sweeps with E,0,0 marker, and sensor streaming.
"""

import socket
import sys
import threading
import time


class FakeUnoServer:
    def __init__(self, env, host="127.0.0.1", port=9999, motor_port=9998):
        self.env = env
        self.host = host
        self.port = port
        self.motor_port = motor_port
        self.running = False
        self.server_sock = None
        self.motor_sock = None
        self.client_sock = None
        self.client_file = None
        self.lock = threading.Lock()

        self.mode = 'F'  # 'F', 'B', 'X'
        self.last_physics_t = time.time()
        self.last_imu_t = 0.0
        self.last_batt_t = 0.0
        self.last_stream_t = 0.0

        self.thread = None
        self.motor_thread = None

    def start(self):
        self.running = True
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        self.server_sock.settimeout(0.5)

        if self.motor_port:
            try:
                self.motor_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.motor_sock.bind((self.host, self.motor_port))
                self.motor_sock.settimeout(0.5)
                self.motor_thread = threading.Thread(target=self._run_motor_listener, daemon=True)
                self.motor_thread.start()
            except Exception:
                pass

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run_motor_listener(self):
        while self.running and self.motor_sock:
            try:
                data, addr = self.motor_sock.recvfrom(1024)
                if data:
                    parts = data.decode("utf-8").split(",")
                    if len(parts) == 2:
                        left = int(parts[0])
                        right = int(parts[1])
                        self.env.set_motors(left, right)
            except (socket.timeout, BlockingIOError, OSError):
                continue
            except Exception:
                pass

    def stop(self):
        self.running = False
        if self.client_sock:
            try:
                self.client_sock.close()
            except Exception:
                pass
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
        if self.motor_sock:
            try:
                self.motor_sock.close()
            except Exception:
                pass

    def _send_line(self, line):
        with self.lock:
            if self.client_sock:
                try:
                    data = (line.strip() + "\r\n").encode("utf-8")
                    self.client_sock.sendall(data)
                except Exception:
                    self.client_sock = None

    def _handle_commands(self):
        """Read any incoming serial bytes from client."""
        if not self.client_sock:
            return
        try:
            self.client_sock.setblocking(False)
            data = self.client_sock.recv(1024)
            if not data:
                return
            for byte in data:
                char = chr(byte)
                if char in ('F', 'B', 'X'):
                    self.mode = char
                elif char == 'S':
                    self._do_scan()
                elif char == 'W':
                    self._do_wide_sweep()
                elif char == 'Z':
                    # Zero heading
                    self.env.accumulated_yaw = 0.0
                    self.env.initial_theta = self.env.theta
                elif char == 'C':
                    # Gyro calibration
                    time.sleep(0.1)
        except BlockingIOError:
            pass
        except Exception:
            self.client_sock = None

    def _do_scan(self):
        """Perform normal side scan (S)."""
        side = 'B' if self.mode == 'B' else 'F'
        # Left angle (150)
        d_left = self.env.get_front_distance(150) if side == 'F' else self.env.get_back_distance(150)
        time.sleep(0.1)
        self._send_line(f"{side},150,{d_left:.1f}")
        # Right angle (30)
        d_right = self.env.get_front_distance(30) if side == 'F' else self.env.get_back_distance(30)
        time.sleep(0.1)
        self._send_line(f"{side},30,{d_right:.1f}")

    def _do_wide_sweep(self):
        """Perform wide sweep (W) in 15 deg steps and finish with E,0,0."""
        side = 'B' if self.mode == 'B' else 'F'
        for ang in range(15, 166, 15):
            d = self.env.get_front_distance(ang) if side == 'F' else self.env.get_back_distance(ang)
            self._send_line(f"{side},{ang},{d:.1f}")
            time.sleep(0.04)  # allow physics & imu to interleave
        self._send_line("E,0,0")

    def _run(self):
        while self.running:
            if not self.client_sock:
                try:
                    conn, addr = self.server_sock.accept()
                    self.client_sock = conn
                    self.client_sock.setblocking(False)
                except (socket.timeout, OSError):
                    time.sleep(0.05)
                    continue

            now = time.time()
            dt = now - self.last_physics_t
            self.last_physics_t = now
            # 1. Update physics with real elapsed time
            self.env.update_physics(min(0.05, max(0.001, dt)))

            # 2. Check incoming commands
            self._handle_commands()

            # 3. IMU reporting at 50 Hz (every 20 ms)
            if now - self.last_imu_t >= 0.02:
                self.last_imu_t = now
                imu = self.env.get_imu()
                if imu is None:
                    self._send_line("M,ERR")
                else:
                    yaw, rate, vib = imu
                    self._send_line(f"M,{yaw:.2f},{rate:.2f},{vib:.1f}")

            # 4. Battery reporting every 0.5 s
            if now - self.last_batt_t >= 0.5:
                self.last_batt_t = now
                self._send_line("V,0,12.20")

            # 5. Continuous distance streaming
            if now - self.last_stream_t >= 0.06:
                self.last_stream_t = now
                if self.mode == 'F':
                    dist = self.env.get_front_distance(90)
                    self._send_line(f"F,90,{dist:.1f}")
                elif self.mode == 'B':
                    dist = self.env.get_back_distance(90)
                    self._send_line(f"B,90,{dist:.1f}")

            time.sleep(0.008)
