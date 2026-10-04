package com.sgsits.stress;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RfModel {
    public static final int N_TREES = 3;
    public static final int N_FEATURES = 19;
    public static final int N_CLASSES = 5;

    private final List<TreeNode> roots;

    public RfModel() {
        this.roots = buildExampleForest();
    }

    public int predict(float[] features, float[] probabilities) {
        if (features == null || features.length != N_FEATURES) {
            throw new IllegalArgumentException("Expected " + N_FEATURES + " features.");
        }
        if (probabilities == null || probabilities.length != N_CLASSES) {
            throw new IllegalArgumentException("Expected " + N_CLASSES + " probabilities.");
        }

        Arrays.fill(probabilities, 0.0f);
        for (TreeNode root : roots) {
            int prediction = predictTree(root, features);
            probabilities[prediction] += 1.0f;
        }

        float total = 0.0f;
        for (int i = 0; i < probabilities.length; i++) {
            total += probabilities[i];
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = total > 0.0f ? probabilities[i] / total : 0.0f;
        }

        int bestClass = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[bestClass]) {
                bestClass = i;
            }
        }
        return bestClass;
    }

    private int predictTree(TreeNode node, float[] features) {
        while (node.left != null || node.right != null) {
            if (features[node.featureIndex] <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return node.prediction;
    }

    private List<TreeNode> buildExampleForest() {
        List<TreeNode> trees = new ArrayList<>();

        TreeNode root1 = new TreeNode(0, 0.25f, null, null, 0);
        root1.left = new TreeNode(2, 82.0f, null, null, 0);
        root1.right = new TreeNode(2, 82.0f, null, null, 1);
        root1.left.left = new TreeNode(-1, 0.0f, null, null, 0);
        root1.left.right = new TreeNode(-1, 0.0f, null, null, 2);
        root1.right.left = new TreeNode(-1, 0.0f, null, null, 1);
        root1.right.right = new TreeNode(-1, 0.0f, null, null, 3);
        trees.add(root1);

        TreeNode root2 = new TreeNode(4, 0.18f, null, null, 0);
        root2.left = new TreeNode(5, 0.20f, null, null, 0);
        root2.right = new TreeNode(5, 0.20f, null, null, 2);
        root2.left.left = new TreeNode(-1, 0.0f, null, null, 0);
        root2.left.right = new TreeNode(-1, 0.0f, null, null, 1);
        root2.right.left = new TreeNode(-1, 0.0f, null, null, 2);
        root2.right.right = new TreeNode(-1, 0.0f, null, null, 3);
        trees.add(root2);

        TreeNode root3 = new TreeNode(9, 1.02f, null, null, 0);
        root3.left = new TreeNode(10, 0.24f, null, null, 0);
        root3.right = new TreeNode(10, 0.24f, null, null, 2);
        root3.left.left = new TreeNode(-1, 0.0f, null, null, 0);
        root3.left.right = new TreeNode(-1, 0.0f, null, null, 1);
        root3.right.left = new TreeNode(-1, 0.0f, null, null, 2);
        root3.right.right = new TreeNode(-1, 0.0f, null, null, 4);
        trees.add(root3);

        return trees;
    }

    public static final class TreeNode {
        public final int featureIndex;
        public final float threshold;
        public TreeNode left;
        public TreeNode right;
        public final int prediction;

        public TreeNode(int featureIndex, float threshold, TreeNode left, TreeNode right, int prediction) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
            this.prediction = prediction;
        }
    }
}
