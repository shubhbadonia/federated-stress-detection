# ESP32 Stress Detection — trained Random Forest

This package contains an embedded inference prototype built from the trained
Random Forest model in `../models/random_forest_model.joblib`.

## Model

- RandomForestClassifier
- 200 trees
- max_depth = 10
- 19 input features
- classes: 0, 1, 2, 3, 4

The training artifact itself is not deployed to the ESP32. The extracted
decision-tree arrays live in `rf_model.cpp` / `rf_model.h`.

## Files

- `esp32_stress_rf.ino` — ESP32 application
- `rf_model.h` — model declarations
- `rf_model.cpp` — exported 200-tree forest + inference
- `README.md` — this guide

## Arduino IDE libraries

Install:

1. SparkFun MAX3010x Sensor Library
2. Adafruit MPU6050
3. Adafruit Unified Sensor

Built-in:
- Wire
- WiFi
- HTTPClient

## Wiring

MAX30102:
- VIN -> 3.3V
- GND -> GND
- SDA -> ESP32 SDA
- SCL -> ESP32 SCL

MPU6050:
- VIN -> 3.3V
- GND -> GND
- SDA -> ESP32 SDA
- SCL -> ESP32 SCL

GSR:
- analog output -> GPIO 34
- GND -> GND
- VCC -> appropriate supply for your module

## Web API

The sketch POSTs JSON to:

`http://YOUR_SERVER_IP:8000/api/stress`

Example:

```json
{
  "prediction": 1,
  "probabilities": [0.02, 0.91, 0.01, 0.03, 0.03],
  "features": [ ... 19 values ... ]
}
```

Change `API_URL` in the sketch.

## Model validation

Before deploying, compare the ESP32 inference against Python on the SAME 19-feature
vectors. Export several `X_test` rows and make sure the C++ prediction equals
`rf.predict()` for each row. This is the strongest test that the tree conversion
was done correctly.

