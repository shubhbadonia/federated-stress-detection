# Java Equivalent

This folder contains a small Java reference implementation of the ESP32 prototype's
feature construction and JSON payload handling.

It mirrors the 19-feature layout used by the embedded sketch, but it does not
replace the generated C++ Random Forest export used on the ESP32.

## Files

- `src/main/java/com/sgsits/stress/StressDetectionJavaEquivalent.java` — feature extraction helpers and HTTP payload support

## Notes

- Uses only the Java standard library.
- Intended as a JVM-side reference, not as a hardware port.
- The actual embedded model remains in `esp32/rf_model.cpp` and `esp32/rf_model.h`.
