# Verify C++ export against the Python model.
# Run this on a computer with numpy + scikit-learn + the original .pkl.
# The C++ implementation itself should be tested on ESP32 with exported X_test rows.

import pickle
import numpy as np

with open("random_forest_model.pkl", "rb") as f:
    rf = pickle.load(f)

print("trees:", len(rf.estimators_))
print("features:", rf.n_features_in_)
print("classes:", rf.classes_)
print("max_depths:", max(t.tree_.max_depth for t in rf.estimators_))
print("total nodes:", sum(t.tree_.node_count for t in rf.estimators_))
