package com.sgsits.stress;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Arrays;

public final class StressDetectionJavaEquivalent {
    private static final int FEATURE_COUNT = 19;
    private static final int CLASS_COUNT = 5;
    private static final double RESTING_RR_MS = 800.0;
    private static final double RESTING_RR_STD_MS = 30.0;
    private static final double RESTING_HEART_RATE_BPM = 75.0;
    private static final double RESTING_HRV_MS = 30.0;
    private static final double DEFAULT_TEMP_C = 33.0;
    private static final double DEFAULT_TEMP_STD = 0.15;
    private static final double DEFAULT_TEMP_DELTA = 0.3;

    private static final String[] FEATURE_NAMES = {
            "ECG_mean_rr",
            "ECG_std_rr",
            "ECG_heart_rate",
            "ECG_hrv",
            "EDA_mean",
            "EDA_std",
            "EDA_max",
            "EDA_min",
            "EDA_num_peaks",
            "ACC_mean",
            "ACC_std",
            "ACC_max",
            "ACC_min",
            "ACC_rms",
            "TEMP_mean",
            "TEMP_std",
            "TEMP_max",
            "TEMP_min",
            "TEMP_slope"
    };

    private StressDetectionJavaEquivalent() {
    }

    public static double[] buildFeatures(double[] rrMs, double[] edaSamples, double[] accMag) {
        return buildFeatures(rrMs, edaSamples, accMag, DEFAULT_TEMP_C, 0.0);
    }

    public static double[] buildFeatures(
            double[] rrMs,
            double[] edaSamples,
            double[] accMag,
            double tempCelsius,
            double tempSlope
    ) {
        double[] features = new double[FEATURE_COUNT];

        if (rrMs != null && rrMs.length > 0) {
            features[0] = mean(rrMs);
            features[1] = stdDev(rrMs);
            features[2] = 60000.0 / features[0];
            features[3] = stdDev(rrMs);
        } else {
            features[0] = RESTING_RR_MS;
            features[1] = RESTING_RR_STD_MS;
            features[2] = RESTING_HEART_RATE_BPM;
            features[3] = RESTING_HRV_MS;
        }

        features[4] = mean(edaSamples);
        features[5] = stdDev(edaSamples);
        features[6] = max(edaSamples);
        features[7] = min(edaSamples);
        features[8] = countPeaks(edaSamples);

        features[9] = mean(accMag);
        features[10] = stdDev(accMag);
        features[11] = max(accMag);
        features[12] = min(accMag);
        features[13] = rms(accMag);

        features[14] = tempCelsius;
        features[15] = DEFAULT_TEMP_STD;
        features[16] = tempCelsius + DEFAULT_TEMP_DELTA;
        features[17] = tempCelsius - DEFAULT_TEMP_DELTA;
        features[18] = tempSlope;

        return features;
    }

    public static String buildPayload(int prediction, double[] probabilities, double[] features) {
        if (probabilities == null || probabilities.length != CLASS_COUNT) {
            throw new IllegalArgumentException("probabilities must contain exactly " + CLASS_COUNT + " values");
        }
        if (features == null || features.length != FEATURE_COUNT) {
            throw new IllegalArgumentException("features must contain exactly " + FEATURE_COUNT + " values");
        }

        StringBuilder json = new StringBuilder();
        json.append('{');
        json.append("\"prediction\":").append(prediction).append(',');
        json.append("\"probabilities\":[");
        for (int i = 0; i < probabilities.length; i++) {
            if (i > 0) {
                json.append(',');
            }
            json.append(formatDouble(probabilities[i]));
        }
        json.append("],\"features\":[");
        for (int i = 0; i < features.length; i++) {
            if (i > 0) {
                json.append(',');
            }
            json.append(formatDouble(features[i]));
        }
        json.append("]}");
        return json.toString();
    }

    public static int postPrediction(String endpoint, int prediction, double[] probabilities, double[] features)
            throws IOException, InterruptedException {
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                .timeout(Duration.ofSeconds(10))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(
                        buildPayload(prediction, probabilities, features), StandardCharsets.UTF_8))
                .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        return response.statusCode();
    }

    public static void printFeatureVector(double[] features) {
        for (int i = 0; i < features.length; i++) {
            System.out.println(i + ": " + formatDouble(features[i]));
        }
    }

    public static void main(String[] args) {
        double[] rrMs = {810.0, 795.0, 802.0, 788.0};
        double[] eda = {0.12, 0.14, 0.13, 0.20, 0.18};
        double[] accMag = {1.02, 1.00, 1.04, 0.99, 1.01};

        double[] features = buildFeatures(rrMs, eda, accMag, 33.1, 0.0);
        System.out.println("Feature vector: " + Arrays.toString(features));
        printFeatureVector(features);
    }

    private static double mean(double[] values) {
        if (values == null || values.length == 0) {
            return 0.0;
        }
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private static double stdDev(double[] values) {
        if (values == null || values.length <= 1) {
            return 0.0;
        }
        double mean = mean(values);
        double sum = 0.0;
        for (double value : values) {
            double delta = value - mean;
            sum += delta * delta;
        }
        return Math.sqrt(sum / values.length);
    }

    private static double min(double[] values) {
        if (values == null || values.length == 0) {
            return 0.0;
        }
        double result = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] < result) {
                result = values[i];
            }
        }
        return result;
    }

    private static double max(double[] values) {
        if (values == null || values.length == 0) {
            return 0.0;
        }
        double result = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > result) {
                result = values[i];
            }
        }
        return result;
    }

    private static double rms(double[] values) {
        if (values == null || values.length == 0) {
            return 0.0;
        }
        double sum = 0.0;
        for (double value : values) {
            sum += value * value;
        }
        return Math.sqrt(sum / values.length);
    }

    private static int countPeaks(double[] values) {
        if (values == null || values.length < 3) {
            return 0;
        }
        double mean = mean(values);
        double threshold = mean + 0.5 * stdDev(values);
        int peaks = 0;
        for (int i = 1; i < values.length - 1; i++) {
            if (values[i] > values[i - 1] && values[i] >= values[i + 1] && values[i] > threshold) {
                peaks++;
            }
        }
        return peaks;
    }

    private static String formatDouble(double value) {
        return String.format(java.util.Locale.US, "%.5f", value);
    }
}
