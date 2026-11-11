#include <Servo.h>
#include <Wire.h>

// ---------- PINS ----------
const int F_TRIG = 9,  F_ECHO = 10;   // front sensor
const int B_TRIG = 11, B_ECHO = 12;   // back sensor
const int F_SERVO_PIN = 5;            // front servo signal
const int B_SERVO_PIN = 6;            // back servo signal

// ---------- BATTERY MONITOR ----------
// Wiring:  battery +  --[ R_TOP ]--+--[ R_BOTTOM ]-- GND (battery -, same ground as the Uno)
//                                  |
//                                  +-- A0   (+ a 100 nF capacitor from A0 to GND)
// Recommended for a 3S Li-ion pack (9.0 - 12.6 V): R_TOP = 120k, R_BOTTOM = 10k, internal 1.1 V reference.
//   12.6 V -> 0.97 V on A0 (safe).   NEVER connect the battery before checking with a multimeter
//   that the junction reads about 1/13 of the battery voltage.
// If you only have 30k / 10k: set R_TOP = 30000 and USE_INTERNAL_REF = false (5 V reference,
//   less accurate because the Uno's 5 V follows the Pi's USB voltage).
const int   BATT_PIN = A0;
const float R_TOP = 120000.0;          // ohms, battery side
const float R_BOTTOM = 10000.0;        // ohms, ground side
const bool  USE_INTERNAL_REF = true;   // true: 1.1 V reference. false: 5 V reference
const float VBAT_CAL = 1.000;          // calibration: real volts (multimeter) / volts shown on the dashboard
const unsigned long BATT_PERIOD_MS = 500;

// ---------- SERVO ANGLES ----------
const int CENTER = 90;
const int LEFT_ANGLE = 150;
const int RIGHT_ANGLE = 30;

// ---------- MPU-6500 (gyro + accelerometer) ----------
// Wiring: VCC -> Uno 3.3V, GND -> GND, SDA -> A4, SCL -> A5.  Mount it flat, Z axis pointing UP.
// Every 0.1 s the Uno sends ONE line to the Pi:
//     M,<yaw>,<rate>,<vib>        example:  M,-37.52,-0.31,4.8
//   yaw  = heading change in degrees since power-up (or since the Pi sent Z). + = turning LEFT (counter-clockwise).
//          Not wrapped to 360: 725.0 means two full turns left.
//   rate = turn speed right now, degrees per second (+ = left)
//   vib  = vibration level in milli-g (about 0-3 standing still, higher when the wheels are rolling)
// If the chip stops answering the Uno sends  M,ERR  instead, and tries to restart it every 2 s.
//
// Commands from the Pi (typed in brain.py):  zero -> sends Z (heading = 0)
//                                            calibrate -> sends C (re-measure the gyro zero point, rover must stand still, takes 1-3 s)
//                                            imu -> live printout of the M lines in the brain.py terminal (this replaces imu_test.py)
// Quick check with  imu :  standing still -> rate near 0.0 and yaw not moving, vib small.
//   Turn the rover by hand 90 degrees LEFT -> yaw about +90 (if -90, set IMU_YAW_SIGN = -1 in brain.py).
//   Driving -> vib rises. Wheel held stuck while driving -> note rate and vib for the stall detector.
const uint8_t       MPU_ADDR = 0x68;           // AD0 pin to GND or left open. If AD0 is tied to 3.3V use 0x69
const unsigned long IMU_SAMPLE_MS = 5;         // read the chip 200 times a second
const unsigned long IMU_REPORT_MS = 20;        // send a line to the Pi 50 times a second (50 Hz)
const float GYRO_LSB = 65.5;                   // chip units per degree/second at the +-500 deg/s range
const float ACC_LSB = 16384.0;                 // chip units per g at the +-2 g range
const float GZ_DEADBAND = 0.2;                 // turn rates smaller than this (deg/s) count as 0 (kills slow drift)
const int   CAL_SAMPLES = 300;                 // gyro zero-point calibration: about 1 s, rover must stand still

float gyroBias = 0, yaw = 0, rate = 0, vibMg = 0, accLP = 1.0;
bool imuOk = false;
uint8_t imuFails = 0;
unsigned long lastSampleMs = 0, lastUs = 0, lastReportMs = 0, lastRetryMs = 0;

bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

int16_t rd16() {                               // two bytes -> one signed number (high byte first)
  uint8_t hi = Wire.read();
  uint8_t lo = Wire.read();
  return (int16_t)((hi << 8) | lo);
}

// Reads accelerometer X, Y, Z and gyro Z. Returns false if the chip did not answer.
bool mpuRead(int16_t &ax, int16_t &ay, int16_t &az, int16_t &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 14) != 14) return false;
  ax = rd16(); ay = rd16(); az = rd16();
  rd16();                                      // temperature, not used
  rd16(); rd16();                              // gyro X and Y, not used
  gz = rd16();
  return true;
}

bool mpuStart() {
  if (!mpuWrite(0x6B, 0x01)) return false;     // wake up, clock from the gyro
  delay(30);
  mpuWrite(0x1A, 0x03);                        // gyro low-pass 41 Hz (less noise)
  mpuWrite(0x1B, 0x08);                        // gyro range +-500 deg/s
  mpuWrite(0x1C, 0x00);                        // accelerometer range +-2 g
  mpuWrite(0x1D, 0x00);                        // accelerometer wide filter, so vibration is not smoothed away
  return true;
}

// Measure the gyro's zero point. The rover must NOT move while this runs.
void calibrateGyro() {
  for (int attempt = 0; attempt < 3; attempt++) {
    long sum = 0;
    int mn = 32767, mx = -32768, got = 0;
    for (int i = 0; i < CAL_SAMPLES; i++) {
      int16_t ax, ay, az, gz;
      if (mpuRead(ax, ay, az, gz)) {
        sum += gz; got++;
        if (gz < mn) mn = gz;
        if (gz > mx) mx = gz;
      }
      delay(3);
    }
    if (got < CAL_SAMPLES / 2) continue;
    gyroBias = (sum / (float)got) / GYRO_LSB;
    if ((mx - mn) / GYRO_LSB < 5.0) break;     // steady enough: the rover stood still
  }
  yaw = 0; rate = 0; vibMg = 0; accLP = 1.0;
  lastUs = micros();
}

// Called as often as possible. Reads the chip every 5 ms and updates yaw and vibration.
void imuUpdate() {
  if (!imuOk) {
    if (millis() - lastRetryMs >= 2000) {      // try to bring the chip back
      lastRetryMs = millis();
      if (mpuStart()) { imuOk = true; imuFails = 0; calibrateGyro(); }
    }
    return;
  }
  if (millis() - lastSampleMs < IMU_SAMPLE_MS) return;
  lastSampleMs = millis();
  int16_t ax, ay, az, gz;
  unsigned long now = micros();
  if (!mpuRead(ax, ay, az, gz)) {
    if (++imuFails >= 5) imuOk = false;
    return;
  }
  imuFails = 0;
  float dt = (now - lastUs) / 1000000.0;
  lastUs = now;
  if (dt > 0.2) dt = 0.2;
  float r = gz / GYRO_LSB - gyroBias;
  if (fabs(r) < GZ_DEADBAND) r = 0;
  yaw += 0.5 * (r + rate) * dt;                // add up the turning (trapezoid rule)
  rate = r;

  float gx = ax / ACC_LSB, gy = ay / ACC_LSB, gzz = az / ACC_LSB;
  float amag = sqrt(gx * gx + gy * gy + gzz * gzz);   // total pull, about 1 g when still, any tilt
  accLP += 0.02 * (amag - accLP);                      // slow average = gravity
  vibMg = 0.9 * vibMg + 0.1 * fabs(amag - accLP) * 1000.0;
}

void imuReport() {
  if (millis() - lastReportMs < IMU_REPORT_MS) return;
  lastReportMs = millis();
  if (!imuOk) { Serial.println("M,ERR"); return; }
  Serial.print("M,");
  Serial.print(yaw, 2);  Serial.print(',');
  Serial.print(rate, 2); Serial.print(',');
  Serial.println(vibMg, 1);
}

// Use this instead of delay(): the gyro keeps being read while we wait.
void waitMs(unsigned long ms) {
  unsigned long t0 = millis();
  while (millis() - t0 < ms) { imuUpdate(); imuReport(); }
}

Servo fServo, bServo;
char mode = 'X';   // F = front sensor streaming, B = back sensor streaming, X = off
unsigned long lastBatt = 0;

// Battery voltage in volts, or -1 if nothing is connected to A0
float readBattery() {
  analogRead(BATT_PIN);                       // dummy read: lets the ADC input settle
  long sum = 0;
  for (int i = 0; i < 16; i++) sum += analogRead(BATT_PIN);
  float refV = USE_INTERNAL_REF ? 1.1 : 5.0;
  float v = (sum / 16.0) * refV / 1023.0 * (R_TOP + R_BOTTOM) / R_BOTTOM * VBAT_CAL;
  if (v < 3.0) return -1;                     // floating pin / no battery connected
  return v;
}

float readDistance(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  long d = pulseIn(echo, HIGH, 20000);   // wait max 20 ms
  if (d == 0) return -1;                 // no echo
  return d * 0.0343 / 2;                 // cm
}

// 3 readings, return the middle one (removes wrong spikes)
float median3(int trig, int echo) {
  float a = readDistance(trig, echo); waitMs(30);
  float b = readDistance(trig, echo); waitMs(30);
  float c = readDistance(trig, echo);
  float t;
  if (a > b) { t = a; a = b; b = t; }
  if (b > c) { t = b; b = c; c = t; }
  if (a > b) { t = a; a = b; b = t; }
  return b;
}

void sendLine(char side, int angle, float dist) {
  Serial.print(side);
  Serial.print(',');
  Serial.print(angle);
  Serial.print(',');
  Serial.println(dist, 1);
}

// Look left and right with the servo, then come back to center
void scan(char side) {
  Servo &s = (side == 'F') ? fServo : bServo;
  int trig = (side == 'F') ? F_TRIG : B_TRIG;
  int echo = (side == 'F') ? F_ECHO : B_ECHO;

  int angles[2] = {LEFT_ANGLE, RIGHT_ANGLE};
  for (int i = 0; i < 2; i++) {
    s.write(angles[i]);
    waitMs(400);                         // let the servo reach the angle
    sendLine(side, angles[i], median3(trig, echo));
  }
  s.write(CENTER);
  waitMs(400);
}

// Wide sweep in 15-degree steps across the range, terminated with E,0,0
void wideSweep(char side) {
  Servo &s = (side == 'B') ? bServo : fServo;
  int trig = (side == 'B') ? B_TRIG : F_TRIG;
  int echo = (side == 'B') ? B_ECHO : F_ECHO;

  for (int ang = 15; ang <= 165; ang += 15) {
    s.write(ang);
    waitMs(120);                         // allow servo to reach angle while updating IMU
    sendLine(side, ang, median3(trig, echo));
  }
  s.write(CENTER);
  waitMs(150);
  Serial.println("E,0,0");               // explicit end of sweep marker
}

void setup() {
  Serial.begin(115200);
  if (USE_INTERNAL_REF) {
    analogReference(INTERNAL);
    delay(20);
  }
  pinMode(F_TRIG, OUTPUT); pinMode(F_ECHO, INPUT);
  pinMode(B_TRIG, OUTPUT); pinMode(B_ECHO, INPUT);
  fServo.attach(F_SERVO_PIN);
  bServo.attach(B_SERVO_PIN);
  fServo.write(CENTER);
  bServo.write(CENTER);

  Wire.begin();
  Wire.setClock(400000);
#ifdef WIRE_HAS_TIMEOUT
  Wire.setWireTimeout(3000, true);             // a stuck I2C bus must never freeze the sensor loop
#endif
  imuOk = mpuStart();
  if (imuOk) calibrateGyro();                  // rover must stand still for about 1 s after power-up
}

void loop() {
  // Commands from the Pi: F, B, X, S, W (wide sweep), Z (set heading to 0), C (re-measure gyro zero point)
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'F' || c == 'B' || c == 'X') {
      mode = c;
      fServo.write(CENTER);
      bServo.write(CENTER);
    } else if (c == 'S' && mode != 'X') {
      scan(mode);
    } else if (c == 'W') {
      wideSweep(mode == 'B' ? 'B' : 'F');
    } else if (c == 'Z') {
      yaw = 0;
    } else if (c == 'C' && imuOk) {
      calibrateGyro();
    }
  }

  imuUpdate();
  imuReport();

  // Battery voltage every 0.5 s, in every mode:  V,0,11.84
  if (millis() - lastBatt >= BATT_PERIOD_MS) {
    lastBatt = millis();
    Serial.print("V,0,");
    Serial.println(readBattery(), 2);
  }

  // Keep sending the distance straight ahead (front or back)
  if (mode == 'F') {
    sendLine('F', CENTER, readDistance(F_TRIG, F_ECHO));
    waitMs(60);
  } else if (mode == 'B') {
    sendLine('B', CENTER, readDistance(B_TRIG, B_ECHO));
    waitMs(60);
  }
}
