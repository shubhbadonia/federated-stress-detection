/*
  ESP32 Stress Detection Prototype
  --------------------------------
  Uses:
    - MAX30102: pulse/PPG (used as an approximation for ECG-derived HR/HRV)
    - MPU6050: acceleration
    - Optional GSR/EDA analog sensor
    - Optional temperature sensor; otherwise simulated

  IMPORTANT:
    The trained RF was built from WESAD 5-second windows and expects:
      ECG 4 + EDA 5 + ACC 5 + TEMP 5 = 19 features.
    The ESP32 does NOT have the WESAD chest ECG signal, so MAX30102 PPG
    is used to estimate RR intervals. This is an engineering approximation,
    not an exact reproduction of the training sensor modality.

  Libraries:
    SparkFun MAX3010x Sensor Library
    Adafruit MPU6050
    Adafruit Unified Sensor
    Wire
    WiFi
    HTTPClient
*/

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "rf_model.h"

// ---------------- Wi-Fi ----------------
const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";

// Replace with your web API endpoint.
const char* API_URL = "http://YOUR_SERVER_IP:8000/api/stress";

// ---------------- Pins ----------------
#define GSR_PIN 34

MAX30105 max30102;
Adafruit_MPU6050 mpu;

// ---------------- 5-second window ----------------
const uint32_t WINDOW_MS = 5000;

// Acceleration samples collected over the window.
const int MAX_ACC_SAMPLES = 500;
float accMag[MAX_ACC_SAMPLES];
int accCount = 0;

// EDA samples.
const int MAX_EDA_SAMPLES = 500;
float edaSamples[MAX_EDA_SAMPLES];
int edaCount = 0;

// RR intervals in milliseconds from PPG peaks.
const int MAX_RR = 50;
float rrMs[MAX_RR];
int rrCount = 0;

unsigned long windowStart = 0;
unsigned long lastBeat = 0;

// ---------------- Utility ----------------
float meanOf(const float* x, int n) {
  if (n <= 0) return 0.0f;
  float s = 0;
  for (int i=0;i<n;i++) s += x[i];
  return s/n;
}

float stdOf(const float* x, int n) {
  if (n <= 1) return 0.0f;
  float m = meanOf(x,n);
  float s = 0;
  for (int i=0;i<n;i++) {
    float d=x[i]-m;
    s += d*d;
  }
  return sqrtf(s/n);
}

float minOf(const float* x, int n) {
  if (n <= 0) return 0.0f;
  float v=x[0];
  for(int i=1;i<n;i++) if(x[i]<v) v=x[i];
  return v;
}

float maxOf(const float* x, int n) {
  if (n <= 0) return 0.0f;
  float v=x[0];
  for(int i=1;i<n;i++) if(x[i]>v) v=x[i];
  return v;
}

float rmsOf(const float* x, int n) {
  if (n <= 0) return 0.0f;
  float s=0;
  for(int i=0;i<n;i++) s += x[i]*x[i];
  return sqrtf(s/n);
}

// A simple peak count for EDA. The threshold is intentionally conservative.
// Calibrate this to your actual GSR module.
int countEdaPeaks(const float* x, int n) {
  if (n < 3) return 0;
  float m=meanOf(x,n);
  float sd=stdOf(x,n);
  float threshold=m + 0.5f*sd;
  int peaks=0;
  for(int i=1;i<n-1;i++) {
    if(x[i] > x[i-1] && x[i] >= x[i+1] && x[i] > threshold) peaks++;
  }
  return peaks;
}

// ---------------- Sensor collection ----------------
void collectSensorsFor5Seconds() {
  accCount = 0;
  edaCount = 0;
  rrCount = 0;
  windowStart = millis();

  while (millis() - windowStart < WINDOW_MS) {

    // ----- MAX30102 / PPG -----
    long irValue = max30102.getIR();

    if (checkForBeat(irValue)) {
      unsigned long now = millis();

      if (lastBeat != 0) {
        float interval = (float)(now - lastBeat);

        // Physiological RR limits: ~40-180 BPM
        if (interval >= 333 && interval <= 1500 && rrCount < MAX_RR) {
          rrMs[rrCount++] = interval;
        }
      }
      lastBeat = now;
    }

    // ----- MPU6050 -----
    sensors_event_t a, g, tempEvent;
    mpu.getEvent(&a, &g, &tempEvent);

    float amag = sqrtf(
      a.acceleration.x*a.acceleration.x +
      a.acceleration.y*a.acceleration.y +
      a.acceleration.z*a.acceleration.z
    );

    if (accCount < MAX_ACC_SAMPLES)
      accMag[accCount++] = amag;

    // ----- GSR / EDA -----
    int rawGsr = analogRead(GSR_PIN);

    // Normalized 0..1 representation.
    // This is a placeholder scaling; calibrate against your actual module.
    float eda = (float)rawGsr / 4095.0f;

    if (edaCount < MAX_EDA_SAMPLES)
      edaSamples[edaCount++] = eda;

    delay(10);
  }
}

// ---------------- Feature construction ----------------
void buildFeatures(float f[19]) {

  // ===== ECG-derived approximation =====
  // The trained model expects ECG RR statistics.
  // We approximate these from MAX30102 PPG beat intervals.

  if (rrCount > 0) {
    f[0] = meanOf(rrMs, rrCount);     // mean RR
    f[1] = stdOf(rrMs, rrCount);      // std RR / HRV
    f[2] = 60000.0f / f[0];           // heart rate
    f[3] = stdOf(rrMs, rrCount);      // HRV
  } else {
    // No reliable pulse detected.
    // Use a neutral resting estimate rather than random values.
    f[0] = 800.0f;
    f[1] = 30.0f;
    f[2] = 75.0f;
    f[3] = 30.0f;
  }

  // ===== EDA =====
  f[4] = meanOf(edaSamples, edaCount);
  f[5] = stdOf(edaSamples, edaCount);
  f[6] = maxOf(edaSamples, edaCount);
  f[7] = minOf(edaSamples, edaCount);
  f[8] = (float)countEdaPeaks(edaSamples, edaCount);

  // ===== ACC =====
  f[9]  = meanOf(accMag, accCount);
  f[10] = stdOf(accMag, accCount);
  f[11] = maxOf(accMag, accCount);
  f[12] = minOf(accMag, accCount);
  f[13] = rmsOf(accMag, accCount);

  // ===== Temperature =====
  // Your current hardware does not provide the WESAD temperature stream.
  // Simulate a slowly changing skin/body-temperature-like value.
  //
  // This is deliberately deterministic and low-noise so the prototype
  // remains stable. Replace with a real temperature sensor later.
  static float simulatedTemp = 33.0f;
  simulatedTemp += 0.01f * sinf(millis()/60000.0f);
  f[14] = simulatedTemp;
  f[15] = 0.15f;
  f[16] = simulatedTemp + 0.3f;
  f[17] = simulatedTemp - 0.3f;
  f[18] = 0.0f; // temperature slope approximation
}

// ---------------- Web API ----------------
void sendPrediction(int prediction, float probabilities[5], float features[19]) {

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi not connected.");
    return;
  }

  HTTPClient http;
  http.begin(API_URL);
  http.addHeader("Content-Type", "application/json");

  String json = "{";
  json += "\"prediction\":" + String(prediction) + ",";
  json += "\"probabilities\":[";

  for(int i=0;i<5;i++) {
    if(i) json += ",";
    json += String(probabilities[i], 5);
  }

  json += "],\"features\":[";

  for(int i=0;i<19;i++) {
    if(i) json += ",";
    json += String(features[i], 5);
  }

  json += "]}";

  int code = http.POST(json);

  Serial.print("HTTP status: ");
  Serial.println(code);

  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin();

  // MPU6050
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
    while (true) delay(1000);
  }

  // MAX30102
  if (!max30102.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found!");
    while (true) delay(1000);
  }

  max30102.setup();
  max30102.setPulseAmplitudeRed(0x0A);
  max30102.setPulseAmplitudeIR(0x0A);

  // GSR
  analogReadResolution(12);

  // Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void loop() {

  Serial.println("\nCollecting 5-second window...");
  collectSensorsFor5Seconds();

  float features[19];
  buildFeatures(features);

  float probabilities[5];
  int prediction = rf_predict(features, probabilities);

  Serial.println("----- STRESS RESULT -----");
  Serial.print("Prediction: ");
  Serial.println(prediction);

  Serial.print("Probabilities: ");
  for(int i=0;i<5;i++) {
    Serial.print(probabilities[i], 3);
    Serial.print(" ");
  }
  Serial.println();

  Serial.println("Features:");
  for(int i=0;i<19;i++) {
    Serial.print(i);
    Serial.print(": ");
    Serial.println(features[i], 5);
  }

  sendPrediction(prediction, probabilities, features);

  delay(1000);
}
