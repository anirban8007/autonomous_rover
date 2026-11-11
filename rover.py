import os
import socket
import threading
import time

try:
    import smbus
except ImportError:
    try:
        import smbus2 as smbus            # drop-in replacement: pip install smbus2
    except ImportError:
        smbus = None

ADDR = 0x40          # I2C address of the HAT (PCA9685 chip)
bus = None           # opened by pca_init() so a permission problem gives a clear message

# Channel numbers on the HAT
# Motor A (LEFT side): PWM=0, IN1=1, IN2=2
# Motor B (RIGHT side): PWM=5, IN1=3, IN2=4
MOTOR_A = (0, 1, 2)
MOTOR_B = (5, 3, 4)

# Direction fix: True flips that side. Both True = the rover was moving exactly opposite
# (w went backward and a spun right). If only one side is wrong, set only that side to True.
LEFT_REVERSED = True
RIGHT_REVERSED = True

# brain.py drives the motors from two threads (dashboard + mission), and every
# motor update is several separate I2C writes, so they must not interleave.
_lock = threading.RLock()
_motor_listeners = []


def add_motor_listener(fn):
    if fn not in _motor_listeners:
        _motor_listeners.append(fn)


def remove_motor_listener(fn):
    if fn in _motor_listeners:
        _motor_listeners.remove(fn)


class _DummyBus:
    def write_byte_data(self, addr, reg, val):
        pass

    def read_byte_data(self, addr, reg):
        return 0


def open_bus():
    global bus
    if bus is not None:
        return
    if smbus is None:
        bus = _DummyBus()
        return
    try:
        bus = smbus.SMBus(1)
    except (PermissionError, FileNotFoundError, OSError):
        bus = _DummyBus()


def pca_init(freq=1000):
    """Set up the PWM chip."""
    global bus
    open_bus()
    try:
        with _lock:
            bus.write_byte_data(ADDR, 0x00, 0x00)
            prescale = int(25000000 / (4096 * freq) - 1 + 0.5)
            old = bus.read_byte_data(ADDR, 0x00)
            bus.write_byte_data(ADDR, 0x00, (old & 0x7F) | 0x10)  # sleep
            bus.write_byte_data(ADDR, 0xFE, prescale)             # set frequency
            bus.write_byte_data(ADDR, 0x00, old)                  # wake
            time.sleep(0.005)
            bus.write_byte_data(ADDR, 0x00, old | 0x80)           # restart
            stop()                                                # start with motors off
    except OSError:
        bus = _DummyBus()
        stop()


def set_pwm(channel, on, off):
    base = 0x06 + 4 * channel
    with _lock:
        if bus is None:
            open_bus()
        bus.write_byte_data(ADDR, base, on & 0xFF)
        bus.write_byte_data(ADDR, base + 1, on >> 8)
        bus.write_byte_data(ADDR, base + 2, off & 0xFF)
        bus.write_byte_data(ADDR, base + 3, off >> 8)


def pin_high(channel):
    set_pwm(channel, 4096, 0)


def pin_low(channel):
    set_pwm(channel, 0, 4096)


def run_motor(motor, speed):
    """speed: -100 (full backward) to +100 (full forward)"""
    pwm, in1, in2 = motor
    speed = max(-100, min(100, speed))
    with _lock:
        if speed > 0:
            pin_low(in1)
            pin_high(in2)
        elif speed < 0:
            pin_high(in1)
            pin_low(in2)
        else:
            pin_low(in1)
            pin_low(in2)
        set_pwm(pwm, 0, int(abs(speed) * 4095 / 100))


def drive(left, right):
    if LEFT_REVERSED:
        left = -left
    if RIGHT_REVERSED:
        right = -right
    with _lock:
        run_motor(MOTOR_A, left)
        run_motor(MOTOR_B, right)
    for listener in list(_motor_listeners):
        try:
            listener(left, right)
        except Exception:
            pass
    motor_port = os.environ.get("ROVER_MOTOR_PORT")
    if motor_port:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(f"{left},{right}".encode(), ("127.0.0.1", int(motor_port)))
            sock.close()
        except Exception:
            pass


def forward(s=50):
    drive(s, s)


def backward(s=50):
    drive(-s, -s)


def left(s=50):         # gentle curve to the left
    drive(s // 3, s)


def right(s=50):        # gentle curve to the right
    drive(s, s // 3)


def turn_left(s=50):    # spin in place
    drive(-s, s)


def turn_right(s=50):   # spin in place
    drive(s, -s)


def stop():
    drive(0, 0)


if __name__ == "__main__":
    pca_init()
    try:
        print("Forward");    forward(40);    time.sleep(2); stop(); time.sleep(1)
        print("Backward");   backward(40);   time.sleep(2); stop(); time.sleep(1)
        print("Left");       left(40);       time.sleep(2); stop(); time.sleep(1)
        print("Right");      right(40);      time.sleep(2); stop(); time.sleep(1)
        print("Turn left");  turn_left(40);  time.sleep(2); stop(); time.sleep(1)
        print("Turn right"); turn_right(40); time.sleep(2); stop()
    finally:
        stop()
        print("Stopped")
