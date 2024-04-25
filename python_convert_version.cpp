#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <set>
#include <iostream>
#include <random>

struct DecisionNode {
    int feature_index = -1;
    double threshold = 0.0;
    DecisionNode* left = nullptr;
    DecisionNode* right = nullptr;
    double value = 0.0;

    DecisionNode(double val) : value(val) {}
    DecisionNode(int index, double thresh, DecisionNode* l, DecisionNode* r)
        : feature_index(index), threshold(thresh), left(l), right(r) {}
    ~DecisionNode() {
        delete left;
        delete right;
    }
};

double calculate_mse(const std::vector<double>& targets, const std::vector<double>& weights) {
    double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    double weighted_mean = std::inner_product(targets.begin(), targets.end(), weights.begin(), 0.0) / sum_weights;
    double mse = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        mse += weights[i] * std::pow(targets[i] - weighted_mean, 2);
    }
    return mse;
}

// Helper to find unique values and thresholds in a sorted manner
std::vector<double> unique_sorted(const std::vector<double>& data) {
    std::set<double> unique_data(data.begin(), data.end());
    return std::vector<double>(unique_data.begin(), unique_data.end());
}

std::pair<int, double> best_split(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                                  const std::vector<double>& weights, int min_samples_split) {
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_mse = std::numeric_limits<double>::max();

    for (int feature = 0; feature < X[0].size(); ++feature) {
        // Extract all values for the current feature
        std::vector<double> feature_values;
        for (const auto& sample : X) {
            feature_values.push_back(sample[feature]);
        }

        auto thresholds = unique_sorted(feature_values);
        for (double threshold : thresholds) {
            std::vector<double> left_y, right_y, left_weights, right_weights;

            for (size_t i = 0; i < X.size(); ++i) {
                if (X[i][feature] < threshold) {
                    left_y.push_back(y[i]);
                    left_weights.push_back(weights[i]);
                } else {
                    right_y.push_back(y[i]);
                    right_weights.push_back(weights[i]);
                }
            }

            if (left_y.size() < min_samples_split || right_y.size() < min_samples_split) {
                continue;
            }

            double mse_left = calculate_mse(left_y, left_weights);
            double mse_right = calculate_mse(right_y, right_weights);
            double mse_total = mse_left + mse_right;

            if (mse_total < best_mse) {
                best_mse = mse_total;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }
    return {best_feature, best_threshold};
}

double weighted_average(const std::vector<double>& values, const std::vector<double>& weights) {
    double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    double weighted_sum = std::inner_product(values.begin(), values.end(), weights.begin(), 0.0);
    if (sum_weights == 0) return 0; // To handle the case where all weights might be zero
    return weighted_sum / sum_weights;
}

DecisionNode* build_tree(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                         const std::vector<double>& weights, int min_samples_split, int max_depth, int depth = 0) {
    if (y.size() < 2 * min_samples_split || std::set<double>(y.begin(), y.end()).size() == 1 || depth >= max_depth) {
        double leaf_value = weighted_average(y, weights);
        return new DecisionNode(leaf_value);
    }

    auto [feature, threshold] = best_split(X, y, weights, min_samples_split);
    if (feature == -1) {
        double leaf_value = weighted_average(y, weights);
        return new DecisionNode(leaf_value);
    }

    std::vector<std::vector<double>> left_X, right_X;
    std::vector<double> left_y, right_y, left_weights, right_weights;

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

    DecisionNode* left = build_tree(left_X, left_y, left_weights, min_samples_split, max_depth, depth + 1);
    DecisionNode* right = build_tree(right_X, right_y, right_weights, min_samples_split, max_depth, depth + 1);

    return new DecisionNode(feature, threshold, left, right);
}

double predict_single(const DecisionNode* tree, const std::vector<double>& x) {
    while (tree->value == 0.0) {
        if (x[tree->feature_index] < tree->threshold) {
            tree = tree->left;
        } else {
            tree = tree->right;
        }
    }
    return tree->value;
}

std::vector<double> predict(const DecisionNode* tree, const std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;
    for (const auto& x : X) {
        predictions.push_back(predict_single(tree, x));
    }
    return predictions;
}

int main() {
    // Example usage
    std::vector<std::vector<double>> X_train = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<double> y_train = {1, 3, 5};
    std::vector<std::vector<double>> X_test = {{2, 3}, {4, 5}};

    std::vector<double> weights(y_train.size(), 1.0); // Uniform weights for simplicity

    int min_samples_split = 1;
    int max_depth = 10;

    DecisionNode* tree = build_tree(X_train, y_train, weights, min_samples_split, max_depth);
    auto predictions = predict(tree, X_test);

    std::cout << "Predictions: ";
    for (auto pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    delete tree;  // Clean up dynamically allocated memory
    return 0;
}
