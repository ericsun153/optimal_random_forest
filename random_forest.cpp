#include <vector>
#include <algorithm>
#include <numeric>

class Tree {
private:
    struct Node {
        // Data and structure for the node
    };

    Node* root;

public:
    Tree(const std::vector<double>& X, const std::vector<double>& Y) {
        // Build the tree from data
    }

    ~Tree() {
        // Properly delete nodes to avoid memory leaks
    }

    double predict(const std::vector<double>& x) const {
        // Traverse the tree to predict the value for new data point x
        return 0.0;
    }
};

class RandomForest {
private:
    std::vector<Tree> trees;

public:
    RandomForest(size_t B, double alpha, double w, size_t k) {
        // Create B trees with the specified parameters
    }
 
    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
        // Train each tree with a subset of data
    }

    double predict(const std::vector<double>& x) const {
        std::vector<double> predictions(trees.size());
        for (size_t i = 0; i < trees.size(); ++i) {
            predictions[i] = trees[i].predict(x);
        }
        
        // Take the average of predictions from all trees
        double sum = std::accumulate(predictions.begin(), predictions.end(), 0.0);
        return sum / predictions.size();
    }
};

// ... Additional code to handle input/output, data normalization, etc.

int main() {
    // Example usage
    size_t B = 10; // Number of trees
    double alpha = 0.5;
    double w = 0.1;
    size_t k = 5;
    RandomForest forest(B, alpha, w, k);

    // Example data
    std::vector<std::vector<double>> X_train; // Training features
    std::vector<double> Y_train; // Training labels

    // Train the forest
    forest.train(X_train, Y_train);

    // Predict a new data point
    std::vector<double> x_new; // New data point
    double prediction = forest.predict(x_new);
    
    // Output the prediction
    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
}
