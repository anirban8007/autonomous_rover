"""
2D physics and raycasting simulation environment for Autonomous Rover.
Simulates multi-room floorplans, differential drive dynamics with 5% wheel mismatch,
ultrasonic sensors, MPU-6500 gyro/accelerometer, and collision/stall physics.
"""

import math
import random
import time


def line_intersection(p1, p2, p3, p4):
    """Find intersection point between line segments p1-p2 and ray p3 in direction p4."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-9:
        return None  # Parallel

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if 0.0 <= ua <= 1.0 and ub >= 0.0:
        ix = x1 + ua * (x2 - x1)
        iy = y1 + ua * (y2 - y1)
        dist = math.hypot(ix - x3, iy - y3)
        return (ix, iy), dist
    return None


class SimEnvironment:
    def __init__(self, walls=None, doors=None, obstacles=None, people=None):
        # Walls: list of ((x1, y1), (x2, y2)) in cm
        self.walls = walls or []
        # Doors: list of {"id": "D1", "x": ..., "y": ..., "width": 80}
        self.doors = doors or []
        # Obstacles: unmapped boxes / obstacles
        self.obstacles = obstacles or []
        # People: list of {"name": "Alice", "x": ..., "y": ...}
        self.people = people or []

        # Rover physical state
        self.x = 0.0       # cm
        self.y = 0.0       # cm
        self.theta = 90.0  # degrees (facing +y at start)
        self.initial_theta = 90.0
        self.accumulated_yaw = 0.0

        # Wheels and speed calibration
        self.wheelbase = 15.0  # cm between wheels
        self.wheel_mismatch = 0.95  # 5% slower right wheel
        self.calib_cm_per_sec = 55.0  # nominal speed at 40% power

        # Motor commands (-100 to 100)
        self.left_cmd = 0
        self.right_cmd = 0

        # Physical velocities
        self.v_actual = 0.0
        self.omega_actual = 0.0

        # Collision / Stall state
        self.is_stalled = False
        self.stall_reason = None
        self.force_stall = False  # for invisible obstacle / stall test

        # Gyro state
        self.gyro_enabled = True
        self.gyro_drift_rate = 0.02  # deg/sec drift
        self.gyro_bias = 0.0
        self.yaw_noise_std = 0.05
        self.vib_baseline = 2.0  # mg standing still
        self.vib_rolling = 10.5  # mg while wheels rolling

        # Timestamps
        self.last_update = time.time()
        self.start_time = time.time()

    def set_motors(self, left, right):
        """Called when motor drive command is received."""
        self.left_cmd = max(-100, min(100, -left))
        self.right_cmd = max(-100, min(100, -right))

    def reset_pose(self, x=0.0, y=0.0, theta=90.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.initial_theta = theta
        self.accumulated_yaw = 0.0
        self.left_cmd = 0
        self.right_cmd = 0
        self.v_actual = 0.0
        self.omega_actual = 0.0
        self.is_stalled = False
        self.force_stall = False
        self.gyro_enabled = True
        self.last_update = time.time()

    def update_physics(self, dt=0.02):
        """Step physics simulation forward by dt seconds."""
        now = time.time()

        # Calculate target wheel speeds (cm/s)
        scale = self.calib_cm_per_sec / 40.0
        v_l_target = self.left_cmd * scale
        v_r_target = self.right_cmd * scale * self.wheel_mismatch

        # Smooth acceleration / lag
        alpha = 0.35
        v_l = v_l_target * alpha
        v_r = v_r_target * alpha

        # Desired forward speed and turning rate
        if self.left_cmd * self.right_cmd < 0:
            # Turning in place (spin)
            v_desired = 0.0
            omega_desired = (self.right_cmd - self.left_cmd) / 100.0 * (90.0 / 0.35)
        else:
            # Driving forward / backward with slight differential drift
            v_desired = (v_l + v_r) / 2.0
            omega_desired = (v_r - v_l) / self.wheelbase * 25.0

        if self.force_stall and (abs(self.left_cmd) > 10 or abs(self.right_cmd) > 10):
            self.is_stalled = True
            self.v_actual = 0.0
            self.omega_actual = 0.0
            return

        # Check collision with walls / obstacles if moving
        rad = math.radians(self.theta)
        dx = v_desired * math.cos(rad) * dt
        dy = v_desired * math.sin(rad) * dt

        new_x = self.x + dx
        new_y = self.y + dy

        # Collision detection (rover radius = 12 cm)
        collided = False
        all_walls = list(self.walls) + list(self.obstacles)
        for p1, p2 in all_walls:
            dist = self._dist_point_to_segment((new_x, new_y), p1, p2)
            if dist < 12.0:
                collided = True
                break

        if collided and abs(v_desired) > 1.0:
            self.is_stalled = True
            self.v_actual = 0.0
            self.omega_actual = 15.0 if self.left_cmd > 0 else -15.0
            self.accumulated_yaw += self.omega_actual * dt
            self.theta = (self.initial_theta + self.accumulated_yaw) % 360.0
        else:
            self.is_stalled = False
            self.v_actual = v_desired
            self.omega_actual = omega_desired
            self.x = new_x
            self.y = new_y
            self.accumulated_yaw += self.omega_actual * dt
            self.theta = (self.initial_theta + self.accumulated_yaw) % 360.0

        self.last_update = now

    def _dist_point_to_segment(self, p, p1, p2):
        px, py = p
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        l2 = dx * dx + dy * dy
        if l2 == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def raycast(self, angle_offset_deg, max_range=400.0):
        """Cast ray from rover at angle (self.theta + angle_offset_deg). Returns distance in cm."""
        ray_angle = math.radians(self.theta + angle_offset_deg)
        x0 = self.x
        y0 = self.y
        x_dir = x0 + math.cos(ray_angle) * 1000.0
        y_dir = y0 + math.sin(ray_angle) * 1000.0

        closest_dist = max_range
        all_lines = list(self.walls) + list(self.obstacles)

        for p1, p2 in all_lines:
            res = line_intersection(p1, p2, (x0, y0), (x_dir, y_dir))
            if res is not None:
                pt, dist = res
                if dist < closest_dist:
                    closest_dist = dist

        noise = random.gauss(0.0, 0.4)
        measured = max(2.0, closest_dist + noise)
        return measured if measured < max_range else 999.0

    def get_front_distance(self, servo_angle=90):
        offset = servo_angle - 90
        return self.raycast(offset)

    def get_back_distance(self, servo_angle=90):
        offset = 180 + (servo_angle - 90)
        return self.raycast(offset)

    def get_imu(self):
        """Return (yaw, rate, vib) matching Arduino M line."""
        if not self.gyro_enabled:
            return None

        yaw = self.accumulated_yaw + random.gauss(0.0, self.yaw_noise_std)
        rate = self.omega_actual + random.gauss(0.0, 0.2)

        if abs(self.left_cmd) > 10 or abs(self.right_cmd) > 10:
            if self.is_stalled and self.force_stall:
                vib = self.vib_baseline + random.uniform(0.1, 0.8)
            else:
                vib = self.vib_rolling + random.gauss(0.0, 1.2)
        else:
            vib = self.vib_baseline + random.uniform(0.0, 0.4)

        return round(yaw, 2), round(rate, 2), round(vib, 1)

    def get_visible_people(self, hfov=60.0, max_dist=350.0):
        """Return visible simulated people within field of view and line of sight."""
        visible = []
        for p in self.people:
            dx = p["x"] - self.x
            dy = p["y"] - self.y
            dist = math.hypot(dx, dy)
            if dist > max_dist:
                continue

            angle_to_p = math.degrees(math.atan2(dy, dx))
            rel_angle = (angle_to_p - self.theta + 180) % 360 - 180

            if abs(rel_angle) <= hfov / 2.0:
                blocked = False
                for p1, p2 in self.walls:
                    res = line_intersection(p1, p2, (self.x, self.y), (p["x"], p["y"]))
                    if res is not None:
                        pt, d = res
                        if d < dist - 5.0:
                            blocked = True
                            break
                if not blocked:
                    w = round(max(0.1, min(0.8, 120.0 / dist)), 2)
                    off = round(rel_angle / (hfov / 2.0), 2)
                    visible.append({"name": p["name"], "dist": dist, "x": off, "w": w, "h": w * 1.2})
        return visible


def create_standard_layout():
    """Create the multi-room + corridor layout for automated test scenarios:
    Room 1 (Start): [0, 240] x [0, 240]
    Doorway D1 at (120, 240) width 80 cm
    Corridor: [80, 200] x [240, 700] (width 120 cm, length 460 cm)
    Side Door D2 at (200, 420) leading to Room 2: [200, 480] x [320, 560]
    End Door D3 at (120, 700) leading to Room 3: [0, 280] x [700, 960]
    2 people: Person 1 (Rahul) in Room 2, Person 2 (Alice) in Room 3.
    """
    walls = [
        # Room 1 (Start Room)
        ((0, 0), (240, 0)),
        ((240, 0), (240, 240)),
        ((0, 0), (0, 240)),
        ((0, 240), (80, 240)),
        ((160, 240), (240, 240)),

        # Corridor West wall
        ((80, 240), (80, 700)),

        # Corridor East wall with Door 2 opening at y=380..460
        ((160, 240), (160, 380)),
        ((160, 460), (160, 700)),

        # Room 2 (East of corridor)
        ((160, 380), (420, 380)),
        ((420, 380), (420, 560)),
        ((420, 560), (160, 560)),
        ((160, 560), (160, 460)),

        # Room 3 (North of corridor)
        ((0, 700), (80, 700)),
        ((160, 700), (280, 700)),
        ((0, 700), (0, 960)),
        ((0, 960), (280, 960)),
        ((280, 960), (280, 700)),
    ]

    doors = [
        {"id": "D1", "x": 120.0, "y": 240.0, "width": 80.0},
        {"id": "D2", "x": 160.0, "y": 420.0, "width": 80.0},
        {"id": "D3", "x": 120.0, "y": 700.0, "width": 80.0},
    ]

    people = [
        {"name": "Rahul", "x": 280.0, "y": 460.0},
        {"name": "Alice", "x": 140.0, "y": 850.0},
    ]

    env = SimEnvironment(walls=walls, doors=doors, people=people)
    env.reset_pose(x=120.0, y=100.0, theta=90.0)
    return env
