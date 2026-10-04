package com.sgsits.stress;

import java.util.Arrays;

public class EmbeddedStressApp {
    private static final int FEATURE_COUNT = 19;
    private static final int CLASS_COUNT = 5;

    public static void main(String[] args) {
        EmbeddedStressApp app = new EmbeddedStressApp();
        double[] rrMs = {800.0, 790.0, 815.0, 805.0, 810.0};
        double[] eda = {0.12, 0.14, 0.13, 0.20, 0.18};
        double[] accMag = {1.02, 1.00, 1.04, 0.99, 1.01};

        double[] features = StressDetectionJavaEquivalent.buildFeatures(rrMs, eda, accMag, 33.1, 0.0);
        float[] probabilities = new float[CLASS_COUNT];

        RfModel model = new RfModel();
        int prediction = model.predict(toFloatArray(features), probabilities);

        System.out.println("Prediction: " + prediction);
        System.out.println("Probabilities: " + Arrays.toString(probabilities));
        System.out.println("Features: " + Arrays.toString(features));
    }

    private static float[] toFloatArray(double[] values) {
        float[] result = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = (float) values[i];
        }
        return result;
    }
}
