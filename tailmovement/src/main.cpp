#include <Arduino.h>
#include <ESP32Servo.h>
#include <math.h>

// ESP32-S3 SuperMini + SG90 servo
// Servo signal -> GPIO4
// Feedback pin   -> GPIO1 (optional analog feedback line from the servo)

namespace {

Servo servo;

constexpr int SERVO_PIN = 4;
constexpr int FB_PIN = 1;

constexpr int SERVO_MIN_US = 500;
constexpr int SERVO_MAX_US = 2500;
constexpr int SERVO_CENTER_US = 1500;
constexpr unsigned long UPDATE_MS = 10;

float amplitudeUs = 0.0f;
float wagFrequencyHz = 3.0f;
float envelopeFrequencyHz = 0.5f;
unsigned long watchdogTimeoutMs = 500;
unsigned long lastCommandMs = 0;
unsigned long lastUpdateMs = 0;
unsigned long startMs = 0;

String serialBuffer;

float clampAmplitudeUs(float value) {
  if (value < 0.0f) {
    return 0.0f;
  }
  if (value > 500.0f) {
    return 500.0f;
  }
  return value;
}

float parseFloatToken(char *token, float fallback) {
  if (token == nullptr) {
    return fallback;
  }
  return static_cast<float>(atof(token));
}

unsigned long parseUnsignedToken(char *token, unsigned long fallback) {
  if (token == nullptr) {
    return fallback;
  }
  long value = atol(token);
  if (value < 0) {
    return fallback;
  }
  return static_cast<unsigned long>(value);
}

void writeCentered() {
  servo.writeMicroseconds(SERVO_CENTER_US);
}

void printStatus() {
  int fb = analogRead(FB_PIN);
  float fbVolts = fb * 3.3f / 4095.0f;
  Serial.print("OK STATUS amp_us=");
  Serial.print(amplitudeUs, 1);
  Serial.print(" wag_hz=");
  Serial.print(wagFrequencyHz, 3);
  Serial.print(" env_hz=");
  Serial.print(envelopeFrequencyHz, 3);
  Serial.print(" timeout_ms=");
  Serial.print(watchdogTimeoutMs);
  Serial.print(" fb_v=");
  Serial.println(fbVolts, 3);
}

void handleCommand(String message) {
  message.trim();
  if (message.isEmpty()) {
    return;
  }

  char buffer[128];
  message.toCharArray(buffer, sizeof(buffer));
  char *token = strtok(buffer, " ");
  if (token == nullptr) {
    return;
  }

  if (strcmp(token, "STOP") == 0) {
    amplitudeUs = 0.0f;
    lastCommandMs = millis();
    writeCentered();
    return;
  }

  if (strcmp(token, "PING") == 0) {
    Serial.println("OK PONG");
    return;
  }

  if (strcmp(token, "STATUS") == 0) {
    printStatus();
    return;
  }

  if (strcmp(token, "CMD") != 0) {
    Serial.print("ERR unknown_command ");
    Serial.println(message);
    return;
  }

  char *ampToken = strtok(nullptr, " ");
  char *wagToken = strtok(nullptr, " ");
  char *envToken = strtok(nullptr, " ");
  char *timeoutToken = strtok(nullptr, " ");

  amplitudeUs = clampAmplitudeUs(parseFloatToken(ampToken, amplitudeUs));
  wagFrequencyHz = max(0.0f, parseFloatToken(wagToken, wagFrequencyHz));
  envelopeFrequencyHz = max(0.0f, parseFloatToken(envToken, envelopeFrequencyHz));
  watchdogTimeoutMs = parseUnsignedToken(timeoutToken, watchdogTimeoutMs);
  lastCommandMs = millis();
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char incoming = static_cast<char>(Serial.read());
    if (incoming == '\n' || incoming == '\r') {
      if (!serialBuffer.isEmpty()) {
        handleCommand(serialBuffer);
        serialBuffer = "";
      }
      continue;
    }
    serialBuffer += incoming;
  }
}

void updateServo() {
  unsigned long now = millis();
  if (now - lastUpdateMs < UPDATE_MS) {
    return;
  }
  lastUpdateMs = now;

  bool timedOut = watchdogTimeoutMs > 0 && (now - lastCommandMs) > watchdogTimeoutMs;
  if (timedOut || amplitudeUs <= 0.0f || wagFrequencyHz <= 0.0f) {
    writeCentered();
    return;
  }

  float t = (now - startMs) / 1000.0f;
  float env = 1.0f;
  if (envelopeFrequencyHz > 0.0f) {
    env = 0.6f + 0.4f * sinf(2.0f * PI * envelopeFrequencyHz * t);
  }
  float wag = sinf(2.0f * PI * wagFrequencyHz * t);
  int pulseUs = SERVO_CENTER_US + static_cast<int>(lroundf(amplitudeUs * env * wag));
  pulseUs = constrain(pulseUs, SERVO_MIN_US, SERVO_MAX_US);
  servo.writeMicroseconds(pulseUs);
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);

  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  analogReadResolution(12);

  startMs = millis();
  lastCommandMs = millis();
  writeCentered();

  Serial.println("Tail controller ready");
  Serial.println("Commands:");
  Serial.println("  CMD <amp_us> <wag_hz> <env_hz> <timeout_ms>");
  Serial.println("  STOP");
  Serial.println("  PING");
  Serial.println("  STATUS");
}

void loop() {
  readSerialCommands();
  updateServo();
}
