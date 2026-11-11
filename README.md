# Autonomous Search & Rescue Rover (Team Phoenix)

An autonomous indoor search-and-rescue rover designed for exploring unfamiliar environments (rooms, corridors), avoiding obstacles, identifying doorways, locating and recognizing human survivors using on-device computer vision, and navigating safely back to the entry point using Dijkstra shortest-path topological routing.

Built for **Raspberry Pi 5** paired with an **Arduino Uno** sensor/power coprocessor, **PCA9685 I2C** motor driver, **MPU-6500 IMU**, dual servo-actuated ultrasonic distance sensors, and a real-time **Ground Control Station (GCS)** web dashboard.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Hardware Overview & Pinout](#hardware-overview--pinout)
- [Software Components](#software-components)
  - [brain.py - Central Mission Coordinator](#brainpy---central-mission-coordinator)
  - [face_recog.py - AI Vision & Recognition](#face_recogpy---ai-vision--recognition)
  - [rover.py - Low-Level Actuation](#roverpy---low-level-actuation)
  - [uno_two_sensors.ino - Sensor & Power Firmware](#uno_two_sensorsino---sensor--power-firmware)
  - [index.html - Ground Control Station (GCS)](#indexhtml---ground-control-station-gcs)
- [Installation & Setup](#installation--setup)
- [Calibration & Configuration](#calibration--configuration)
- [Operating the Rover](#operating-the-rover)
  - [Autonomous Mission](#autonomous-mission)
  - [Manual Teleoperation](#manual-teleoperation)
  - [Ground Control Dashboard](#ground-control-dashboard)
- [Mission Reports](#mission-reports)
- [Repository Structure](#repository-structure)

---

## System Architecture

The rover employs a distributed computing architecture separating high-level perception and mission planning from real-time low-level sensing and motion control:

```
                          +-------------------------------------------------------+
                          |                 Raspberry Pi 5 (Linux)                |
                          |                                                       |
                          |   +-------------------+      +--------------------+   |
  Web Browser / Laptop <--+---|  index.html (GCS) |<---->|      brain.py      |   |
  (Port 8000 Dashboard)   |   |  REST / MJPEG API |      | Mission Controller |   |
                          |   +-------------------+      +---------+----------+   |
                          |                                        |              |
                          |   +-------------------+                |              |
  USB Webcam ------------>+---|   face_recog.py   |<---------------+              |
                          |   | YuNet/SFace/YOLO8 |   (HTTP / Loopback :8765)     |
                          |   +-------------------+                |              |
                          |                                        |              |
                          |   +-------------------+                |              |
                          |   |      rover.py     |<---------------+              |
                          |   | PCA9685 I2C Driver|                               |
                          |   +---------+---------+                               |
                          +-------------|--------------------------|--------------+
                                        | I2C (0x40)               | USB Serial
                                        v                          v (115200 baud)
                          +-------------------------+    +------------------------+
                          |      PCA9685 HAT        |    |      Arduino Uno       |
                          |   Dual DC Motor Driver  |    |   Sensor Coprocessor   |
                          +------------+------------+    +-----------+------------+
                                       |                             |
                      +----------------+----------------+            +-- MPU-6500 IMU (I2C)
                      |                                 |            +-- Front HC-SR04 + Servo
                [Left Motors]                     [Right Motors]     +-- Rear HC-SR04 + Servo
                                                                     +-- 3S Battery Divider (A0)
```

---

## Key Features

- **Autonomous Exploration & Space Classification**:
  - Automatically identifies whether the rover is inside a **Room** or an elongated **Corridor** using 360° surround sweep profiles.
  - Dynamically detects and prioritizes unvisited **Doorways** (55 cm to 180 cm gaps) using synchronized ultrasonic sweeps and camera object recognition.

- **AI-Powered Survivor Detection & Face Recognition**:
  - Deep-learning face detection using **YuNet** and feature embedding via **SFace** (128-d cosine similarity).
  - Automatically registers and remembers new individuals (`Person 1`, `Person 2`, ...) without requiring pre-enrollment.
  - Optional **YOLOv8** background thread for detecting furniture (chairs, couches, tables, beds) and architectural doorways.
  - Intercept behavior: When an uncounted person appears in view, the rover visually tracks, aims, drives up to the survivor, pauses to register/photograph them, and resumes exploration.

- **Fault-Tolerant Dead Reckoning & Motion Control**:
  - Closed-loop gyro turns utilizing the **MPU-6500** yaw gyro with inertial anticipation lead time (`IMU_LEAD_SEC`) and timed-turn fallbacks.
  - **Dynamic Ultrasonic Tape Measure**: Continuously calibrates linear drive velocity (`CM_PER_SEC`) in real time against stationary obstacles ahead.
  - **Multi-Modal Stall Detection**: Detects wheel stalls by correlating ultrasonic distance invariance, gyro heading swing/turn rates, and IMU accelerometer vibration baselines.
  - **3-Tier Stall Recovery**: Automatic sequence progressing through back-off wiggle, lateral sidestep maneuvers, and fail-safe latching.

- **Dijkstra Topological Routing & Return-to-Home**:
  - Automatically abstracts continuous motion into topological graph nodes and metric edges (`route_graph`).
  - Computes optimal shortest path back to the origin (`N0`) using **Dijkstra's Algorithm**.
  - Dynamic rerouting: If any path leg is found blocked during return, that edge is pruned and an alternate route is computed on the fly.
  - Return time budgeting: Continually estimates Dijkstra return travel duration and initiates return preemptively before the mission deadline expires.

- **Battery Protection & Failsafes**:
  - Continuously monitors 3S Li-ion battery voltage with analog low-pass filtering.
  - Automated emergency return triggers if voltage drops below threshold (`10.0V` for 5 seconds).
  - Web and terminal hardware software emergency stop (**E-STOP**) latches motor outputs instantly.

- **Comprehensive Ground Control Station (GCS)**:
  - Responsive web interface serving live MJPEG annotated video, battery diagnostics, navigation metrics, and live event log.
  - Dual mapping windows: Live vector breadcrumb map and topological route network graph.
  - Interactive mission modal, teleoperation D-pad, and variable speed slider.
  - Auto-generated HTML + JSON mission reports complete with SVG paths, survivor photo cards, and event timelines.

---

## Hardware Overview & Pinout

Refer to `rover_power_wiring.png` and `rover_wiring.xlsx` for complete electrical diagrams.

### 1. Arduino Uno Pin Mapping
| Pin / Port | Function | Connected To / Details |
|---|---|---|
| **Pin 9** | `F_TRIG` | Front HC-SR04 Trigger |
| **Pin 10** | `F_ECHO` | Front HC-SR04 Echo |
| **Pin 11** | `B_TRIG` | Rear HC-SR04 Trigger |
| **Pin 12** | `B_ECHO` | Rear HC-SR04 Echo |
| **Pin 5** | `F_SERVO_PIN` | Front Pan Servo Signal |
| **Pin 6** | `B_SERVO_PIN` | Rear Pan Servo Signal |
| **Pin A0** | `BATT_PIN` | Battery Voltage Divider (120kΩ top / 10kΩ bottom, 100nF filter cap) |
| **Pin A4** | `SDA` | MPU-6500 IMU I2C Data |
| **Pin A5** | `SCL` | MPU-6500 IMU I2C Clock |
| **USB** | Serial | Connected to Raspberry Pi 5 USB (`115200` baud) |

### 2. PCA9685 I2C Motor HAT (Address 0x40)
| Channel | Function | Motor Association |
|---|---|---|
| **Channel 0** | PWM Speed | Motor A (Left Track/Wheels) |
| **Channel 1** | IN1 Direction | Motor A (Left) |
| **Channel 2** | IN2 Direction | Motor A (Left) |
| **Channel 3** | IN1 Direction | Motor B (Right Track/Wheels) |
| **Channel 4** | IN2 Direction | Motor B (Right) |
| **Channel 5** | PWM Speed | Motor B (Right) |

---

## Software Components

### `brain.py` - Central Mission Coordinator
The core daemon executing on the Raspberry Pi 5. Orchestrates:
- Background threads for Arduino Uno serial processing, IMU integration, battery monitoring, and camera polling.
- High-level state machines: `MANUAL`, `EXPLORING`, `RETURNING`, `DONE`, `STUCK`, `ESTOP`.
- Multi-channel GCS Web Server on port 8000 handling REST control calls, telemetry polling, and camera proxies.
- Topological route mapping, Dijkstra return-to-base navigation, and HTML/JSON mission report generation.

### `face_recog.py` - AI Vision & Recognition
Standalone vision server running on local port 8765:
- Loads OpenCV Zoo ONNX models (`face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`), automatically downloaded on first boot.
- Performs multi-face tracking and cosine distance matching against `faces_db.npz`.
- Background asynchronous YOLOv8 thread (`yolov8n.pt`) scanning for furniture and doors.
- Exposes `/video` (annotated MJPEG), `/status` (detections, targets, bounding coordinates), `/people` (stored photos), and `/mission/reset`.

### `rover.py` - Low-Level Actuation
Direct I2C hardware interface for the PCA9685 PWM controller:
- Thread-safe motor command dispatching using reentrant locks (`threading.RLock`).
- Software motor inversion flags (`LEFT_REVERSED`, `RIGHT_REVERSED`).
- Configurable speed curves (-100 to +100).
- Fallback dummy bus (`_DummyBus`) ensuring testability when physical I2C hardware is disconnected.

### `uno_two_sensors.ino` - Sensor & Power Firmware
Embedded C++ sketch for Arduino Uno:
- 200 Hz IMU sampling and 50 Hz ASCII telemetry streaming (`M,yaw,rate,vib`).
- Trapezoidal integration with gyro deadband filtering.
- Dual median-filtered HC-SR04 sonar measurements with servo scanning sweeps (`S` quick scan, `W` wide 15°–165° sweep).
- 16x oversampled battery monitoring on internal 1.1V reference.

### `index.html` - Ground Control Station (GCS)
Single-page web dashboard providing:
- Real-time video display with detection overlays.
- Telemetry readouts: battery meter, motor command, heading, IMU vibration, detected doors, survivors, and elapsed time.
- Dual interactive canvases: Live metric map and Topological Dijkstra route graph.
- Manual teleoperation D-pad, keyboard bindings (WASD / Arrows), speed slider, and software E-Stop.

---

## Installation & Setup

### 1. Prerequisites
- **Hardware**: Raspberry Pi 5 with Raspberry Pi OS / Ubuntu, Arduino Uno, USB webcam.
- **Python**: Python 3.9+ with pip.

### 2. Install Python Dependencies
```bash
# Core vision and numerical packages
pip install opencv-python numpy pyserial

# I2C motor bus support (on Raspberry Pi)
pip install smbus2

# Optional: Object and doorway detection (brings PyTorch & YOLOv8)
pip install ultralytics
```

### 3. Permissions
Ensure your user has access to serial and I2C buses on Linux:
```bash
sudo usermod -aG dialout,i2c $USER
```
*(Log out and back in for group changes to take effect).*

### 4. Upload Arduino Firmware
1. Open `uno_two_sensors.ino` in the Arduino IDE.
2. Select **Arduino Uno** and your serial port.
3. Install standard `Servo` and `Wire` libraries if not already present.
4. Upload the sketch to the Uno.

---

## Calibration & Configuration

Tune the top constants in `brain.py` according to your specific chassis and floor friction:

```python
# Speed & Drive Calibration
SPEED = 40            # Default motor power (0 - 100)
CM_PER_SEC = 55.0     # Distance moved (cm) in 1 sec at CALIB_SPEED (Measure on floor)
CALIB_SPEED = 40      # Calibration reference motor power
TURN_90_SEC = 0.35    # Seconds required to rotate 90 degrees at TURN_SPEED
IMU_LEAD_SEC = 0.04   # Anticipation lead time (seconds) to cut power before target heading

# Obstacle & Navigation Limits
STOP_DIST = 25        # Forward obstacle stopping distance (cm)
SIDE_MIN = 25         # Minimum clearance required to turn into a path (cm)
BACK_STOP_DIST = 12   # Reverse obstacle stopping threshold (cm)

# Battery Settings (3S Li-ion)
BATT_FULL_V = 12.6
BATT_EMPTY_V = 9.0
BATT_WARN_V = 10.5
BATT_RETURN_V = 10.0  # Threshold triggering automatic return to base
```

---

## Operating the Rover

### Starting the System
Launch the main coordinator (which automatically starts `face_recog.py` and the GCS web server):
```bash
python3 brain.py
```
Upon startup, the Ground Control Station will be accessible at:
```
http://<ROVER_IP>:8000/
```

### Autonomous Mission
From the Ground Control Station:
1. Click **Start mission**.
2. Enter the target number of survivors to locate (e.g., `3`) and the maximum time limit in minutes (e.g., `10`).
3. The rover will autonomously explore, navigate through doors, approach and count people, and return to the entry point upon reaching the target, time limit, or low battery.

From the terminal:
```text
> start 3 10    # Find 3 people, return within 10 minutes
> start 2       # Find 2 people (no time limit)
> start         # Explore continuously until 'return' command
> return        # Return immediately via Dijkstra shortest path
> stop          # Emergency halt
```

### Manual Teleoperation
- **Web GCS**: Select **Manual** mode and hold the on-screen D-pad buttons or press `W`, `A`, `S`, `D` on your keyboard.
- **Terminal**: Use `w` (forward), `s` (reverse), `a` (spin left), `d` (spin right), `q` (curve left), `e` (curve right), and `Enter` to stop.
- **E-STOP**: Click the red **EMERGENCY STOP** button on the GCS or press `Space` to latch a complete motor cut.

---

## Mission Reports

Every mission automatically generates a persistent, standalone report stored under `reports/mission_YYYYMMDD_HHMMSS/`:
- **`report.html`**: Interactive briefing containing:
  - Metric summary cards (survivors found, duration, distance travelled, battery profile).
  - Survivor profile cards with captured face snapshots and spatial discovery coordinates.
  - Vector SVG diagram showing outbound path, return path, doors, and survivor locations.
  - Comprehensive chronological event log.
- **`report.json`**: Machine-readable mission metrics and path coordinates.
- **`photos/`**: High-resolution cropped face captures and context snapshots.

---

## Repository Structure

```
autonomous_rover/
├── brain.py                  # Central navigation, Dijkstra planner, mission coordinator, web server
├── face_recog.py             # YuNet face detector, SFace recognizer, YOLOv8 object detector
├── rover.py                  # PCA9685 I2C motor driver and motion primitives
├── uno_two_sensors.ino       # Arduino Uno firmware: MPU-6500 IMU, dual HC-SR04 sonar, battery
├── index.html                # Ground Control Station (GCS) web application
├── rover_power_wiring.png    # Power distribution wiring diagram
├── rover_wiring.xlsx         # Pinout & connection reference sheet
├── reports/                  # Mission reports, logs, and captured survivor images
│   └── mission_sample/
│       ├── report.html
│       ├── report.json
│       └── photos/
└── simulation/               # Mock hardware and simulated environment test suite
```

---

## License

This project is licensed under the MIT License.
