#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <set>
#include <iostream>
#include <random>

// Define a struct for the decision tree nodes
struct DecisionNode {
    int feature_index = -1;  // Index of the feature used for splitting at this node
    double threshold = 0.0;  // Threshold value for the split at this node
    DecisionNode* left = nullptr;  // Pointer to left child node
    DecisionNode* right = nullptr;  // Pointer to right child node
    double value = 0.0;  // Value of the node if it is a leaf node (no children)

    // Constructor for a leaf node
    DecisionNode(double val) : value(val) {}

    // Constructor for a decision node
    DecisionNode(int index, double thresh, DecisionNode* l, DecisionNode* r)
        : feature_index(index), threshold(thresh), left(l), right(r) {}

    // Destructor to clean up dynamically allocated children
    ~DecisionNode() {
        delete left;
        delete right;
    }
};

// Function to calculate the mean squared error (MSE) for a set of targets and associated weights
double calculate_mse(const std::vector<double>& targets, const std::vector<double>& weights) {
    double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    double weighted_mean = std::inner_product(targets.begin(), targets.end(), weights.begin(), 0.0) / sum_weights;
    double mse = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        mse += weights[i] * std::pow(targets[i] - weighted_mean, 2);
    }
    return mse;
}

// Function to find unique values in a sorted manner for a given feature
std::vector<double> unique_sorted(const std::vector<double>& data) {
    std::set<double> unique_data(data.begin(), data.end());
    return std::vector<double>(unique_data.begin(), unique_data.end());
}

// Function to determine the best feature and threshold to split the dataset
std::pair<int, double> best_split(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                                  const std::vector<double>& weights, int min_samples_split) {
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_mse = std::numeric_limits<double>::max();

    // Loop over each feature
    for (int feature = 0; feature < X[0].size(); ++feature) {
        std::vector<double> feature_values;
        for (const auto& sample : X) {
            feature_values.push_back(sample[feature]);
        }

        auto thresholds = unique_sorted(feature_values);
        for (double threshold : thresholds) {
            std::vector<double> left_y, right_y, left_weights, right_weights;

            // Divide data into left and right according to the threshold
            for (size_t i = 0; i < X.size(); ++i) {
                if (X[i][feature] < threshold) {
                    left_y.push_back(y[i]);
                    left_weights.push_back(weights[i]);
                } else {
                    right_y.push_back(y[i]);
                    right_weights.push_back(weights[i]);
                }
            }

            // Only consider valid splits
            if (left_y.size() < min_samples_split || right_y.size() < min_samples_split) {
                continue;
            }

            // Calculate MSE for both sides and their total
            double mse_left = calculate_mse(left_y, left_weights);
            double mse_right = calculate_mse(right_y, right_weights);
            double mse_total = mse_left + mse_right;

            // Update best split if found a new minimum MSE
            if (mse_total < best_mse) {
                best_mse = mse_total;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }
    return {best_feature, best_threshold};
}

// Function to calculate weighted average of a vector
double weighted_average(const std::vector<double>& values, const std::vector<double>& weights) {
    double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    double weighted_sum = std::inner_product(values.begin(), values.end(), weights.begin(), 0.0);
    return sum_weights == 0 ? 0 : weighted_sum / sum_weights;
}

// Recursive function to build the decision tree
DecisionNode* build_tree(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                         const std::vector<double>& weights, int min_samples_split, int max_depth, int depth = 0) {
    if (y.size() < 2 * min_samples_split || std::set<double>(y.begin(), y.end()).size() == 1 || depth >= max_depth) {
        double leaf_value = weighted_average(y, weights);
        return new DecisionNode(leaf_value);
    }

    auto [feature, threshold] = best_split(X, y, weights, min_samples_split);
    if (feature == -1) { // No valid split found
        double leaf_value = weighted_average(y, weights);
        return new DecisionNode(leaf_value);
    }

    std::vector<std::vector<double>> left_X, right_X;
    std::vector<double> left_y, right_y, left_weights, right_weights;

    // Split data into left and right subsets
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][feature] < threshold) {
            left_X.push_back(X[i]);
            left_y.push_back(y[i]);
            left_weights.push_back(weights[i]);
        } else {
            right_X.push_back(X[i]);
            right_y.push_back(y[i]);
            right_weights.push_back(weights[i]);
        }
    }

    // Recursively build left and right subtrees
    DecisionNode* left = build_tree(left_X, left_y, left_weights, min_samples_split, max_depth, depth + 1);
    DecisionNode* right = build_tree(right_X, right_y, right_weights, min_samples_split, max_depth, depth + 1);

    return new DecisionNode(feature, threshold, left, right);
}

// Function to predict the output for a single data point by traversing the tree
double predict_single(const DecisionNode* tree, const std::vector<double>& x) {
    while (tree->value == 0.0) { // Traverse until a leaf node is reached
        tree = x[tree->feature_index] < tree->threshold ? tree->left : tree->right;
    }
    return tree->value;
}

// Function to predict outputs for a dataset
std::vector<double> predict(const DecisionNode* tree, const std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;
    for (const auto& x : X) {
        predictions.push_back(predict_single(tree, x));
    }
    return predictions;
}

int main() {
    // Example usage of the tree
    std::vector<std::vector<double>> X_train = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<double> y_train = {1, 3, 5};
    std::vector<std::vector<double>> X_test = {{2, 3}, {4, 5}};

    std::vector<double> weights(y_train.size(), 1.0); // Uniform weights for simplicity

    int min_samples_split = 1;
    int max_depth = 10;

    // Build the decision tree using the training data
    DecisionNode* tree = build_tree(X_train, y_train, weights, min_samples_split, max_depth);
    // Predict using the built tree
    auto predictions = predict(tree, X_test);

    std::cout << "Predictions: ";
    for (auto pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    delete tree;  // Clean up dynamically allocated memory
    return 0;
}
