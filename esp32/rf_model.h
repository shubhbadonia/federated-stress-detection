#pragma once
#include <Arduino.h>

// Auto-generated from the trained scikit-learn RandomForestClassifier.
// 200 trees, max_depth=10, 19 features, classes 0..4.
// Do not reorder FEATURE_NAMES.

#define RF_N_TREES 200
#define RF_N_FEATURES 19
#define RF_N_CLASSES 5
#define RF_N_NODES 30388

extern const int32_t rf_left[RF_N_NODES];
extern const int32_t rf_right[RF_N_NODES];
extern const int8_t rf_feature[RF_N_NODES];
extern const float rf_threshold[RF_N_NODES];
extern const uint16_t rf_counts[RF_N_NODES][RF_N_CLASSES];
extern const int32_t rf_roots[RF_N_TREES];

int rf_predict(const float features[RF_N_FEATURES], float probabilities[RF_N_CLASSES]);
