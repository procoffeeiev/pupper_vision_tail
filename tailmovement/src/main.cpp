#include <Arduino.h>
#include <math.h>

// ESP32-C3 SuperMini + SG90 舵机
// 接线：
//   SG90 橙（信号）→ GPIO4
//   SG90 红（VCC）  → 5V
//   SG90 棕（GND）  → G

constexpr int SERVO_PIN = 4;
constexpr int LEDC_CHANNEL = 0;
constexpr int LEDC_FREQ = 50;              // SG90 工作频率 50 Hz
constexpr int LEDC_RESOLUTION = 14;        // ESP32-C3 LEDC 最大 14 位
constexpr int PERIOD_US = 20000;           // 1/50Hz = 20ms 周期
constexpr int MAX_DUTY = (1 << 14) - 1;    // 16383

constexpr int SERVO_MIN_PULSE_US = 1000;
constexpr int SERVO_MAX_PULSE_US = 2000;
constexpr int SERVO_CENTER_PULSE_US = 1500;
constexpr float SERVO_US_PER_DEGREE = 500.0f / 90.0f;
constexpr float CENTER_ANGLE_DEG = 90.0f;
constexpr float DEFAULT_AMPLITUDE_DEG = 20.0f;
constexpr float MAX_AMPLITUDE_DEG = 85.0f;
constexpr float SINE_FREQUENCY_HZ = 1.2f;
constexpr float SINE_PERIOD_MS = 1000.0f / SINE_FREQUENCY_HZ;
constexpr unsigned long UPDATE_INTERVAL_MS = 10;

int pulseToDuty(int pulseUs) {
  return (int)((long)pulseUs * MAX_DUTY / PERIOD_US);
}

void writeServoPulse(int pulseUs) {
  int clampedPulseUs = constrain(pulseUs, SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US);
  ledcWrite(LEDC_CHANNEL, pulseToDuty(clampedPulseUs));
}

int angleToPulseUs(float angleDeg) {
  float centeredAngle = angleDeg - CENTER_ANGLE_DEG;
  float pulseUs = SERVO_CENTER_PULSE_US + centeredAngle * SERVO_US_PER_DEGREE;
  return static_cast<int>(lroundf(pulseUs));
}

void writeServoAngle(float angleDeg) {
  float clampedAngleDeg = constrain(angleDeg, 0.0f, 180.0f);
  writeServoPulse(angleToPulseUs(clampedAngleDeg));
}

float amplitudeDeg = DEFAULT_AMPLITUDE_DEG;
String serialBuffer;
unsigned long cycleStartMs = 0;
unsigned long lastUpdateMs = 0;

void restartSineCycle() {
  cycleStartMs = millis();
  writeServoAngle(CENTER_ANGLE_DEG);
}

float clampAmplitude(float amplitude) {
  if (amplitude < 0.0f) {
    return 0.0f;
  }
  if (amplitude > MAX_AMPLITUDE_DEG) {
    return MAX_AMPLITUDE_DEG;
  }
  return amplitude;
}

void printUsage() {
  Serial.println("Send sine amplitude in degrees over USB serial, e.g. 10 or 25");
  Serial.print("Center angle: ");
  Serial.print(CENTER_ANGLE_DEG, 1);
  Serial.println(" deg");
  Serial.print("Center pulse trim: ");
  Serial.print(SERVO_CENTER_PULSE_US);
  Serial.println(" us");
  Serial.print("Valid range: 0 to ");
  Serial.print(MAX_AMPLITUDE_DEG, 1);
  Serial.println(" deg");
}

void handleAmplitudeMessage(const String &message) {
  String trimmed = message;
  trimmed.trim();

  if (trimmed.isEmpty()) {
    return;
  }

  float requestedAmplitude = trimmed.toFloat();
  bool parsedZero = requestedAmplitude == 0.0f &&
                    trimmed != "0" &&
                    trimmed != "0.0" &&
                    trimmed != "0.00";
  if (parsedZero) {
    Serial.print("Invalid amplitude: ");
    Serial.println(trimmed);
    printUsage();
    return;
  }

  amplitudeDeg = clampAmplitude(requestedAmplitude);
  restartSineCycle();
  Serial.print("Amplitude set to ");
  Serial.print(amplitudeDeg, 1);
  Serial.println(" deg");
}

void readSerialAmplitude() {
  while (Serial.available() > 0) {
    char incoming = static_cast<char>(Serial.read());
    if (incoming == '\n' || incoming == '\r') {
      if (!serialBuffer.isEmpty()) {
        handleAmplitudeMessage(serialBuffer);
        serialBuffer = "";
      }
      continue;
    }
    serialBuffer += incoming;
  }
}

void updateServoSine() {
  unsigned long nowMs = millis();
  if (nowMs - lastUpdateMs < UPDATE_INTERVAL_MS) {
    return;
  }
  lastUpdateMs = nowMs;

  float elapsedMs = static_cast<float>(nowMs - cycleStartMs);
  if (elapsedMs >= SINE_PERIOD_MS) {
    unsigned long completedPeriods = static_cast<unsigned long>(elapsedMs / SINE_PERIOD_MS);
    cycleStartMs += static_cast<unsigned long>(lroundf(completedPeriods * SINE_PERIOD_MS));
    writeServoAngle(CENTER_ANGLE_DEG);
    elapsedMs = static_cast<float>(nowMs - cycleStartMs);
  }

  float phase = 2.0f * PI * (elapsedMs / SINE_PERIOD_MS);
  float angleDeg = CENTER_ANGLE_DEG + amplitudeDeg * sinf(phase);
  writeServoAngle(angleDeg);
}

void setup() {
  Serial.begin(115200);
  delay(1500);                             // 等 USB-CDC 枚举完成

  ledcSetup(LEDC_CHANNEL, LEDC_FREQ, LEDC_RESOLUTION);
  ledcAttachPin(SERVO_PIN, LEDC_CHANNEL);
  lastUpdateMs = 0;

  Serial.println("Servo sine motion ready on GPIO4 (ESP32-C3 SuperMini)");
  restartSineCycle();

  Serial.print("Default amplitude: ");
  Serial.print(amplitudeDeg, 1);
  Serial.println(" deg");
  printUsage();
}

void loop() {
  readSerialAmplitude();
  updateServoSine();
}
