// import all necessary extension
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <iostream>
#include <limits>
#include <stdexcept>
using std::string;
using namespace std;

struct DataPoint {
    std::vector<double> features;
    double response;
};

class Node {
public:
    std::vector<DataPoint> data;
	Node * left;
	Node * right;
    int splitFeature; // The index of the feature used to split this node.
    double splitValue; // The value of the feature
    double meanResponse; // Used as the prediction for new data points.

public:    
    Node(std::vector<DataPoint> data)
        : data(data), left(nullptr), right(nullptr), splitFeature(-1), splitValue(0), meanResponse(0) {
        // Calculate the mean response when the node is created, provided there is data.
        if (!data.empty()) {
            meanResponse = getResponseMean();
        }
    }

    // Destructor.
    ~Node() {
        delete left; 
        delete right;
    }
    
    bool isLeaf() const {
        return (left == nullptr && right == nullptr);
    }

    double getResponseMean() const {
        // If no data left, don't make decision
        if (data.empty()) {
            return 0; // Avoid division by zero if there's no data.
        }
        
        double sum = 0.0;
        for (const auto& point : data) {
            sum += point.response;
        }
        return sum / data.size();
    }

    void partition(int featureIndex, double value) {
        splitFeature = featureIndex;
        splitValue = value;

        // Temporary storage for splitted data.
        std::vector<DataPoint> leftData;
        std::vector<DataPoint> rightData;

        // Partition the data into left and right sets based on the split criterion.
        for (const auto& point : data) {
            if (point.features[featureIndex] < value) {
                leftData.push_back(point);
            }
            else {
                rightData.push_back(point);
            }
        }

        // Create new nodes for the left and right children.
        left = new Node(leftData);
        right = new Node(rightData);

        data.clear();
    }
};

class DecisionTree {
private:
    Node* root;

    DecisionTree() : root(nullptr) {}

    Node* buildTree(std::vector<DataPoint>& data, size_t minSize) {
        if (data.size() <= minSize) {
            return new Node(data);
        }

        double bestMSE = std::numeric_limits<double>::max();
        int bestFeature = -1;
        double bestValue = 0;
        std::vector<DataPoint> bestLeft, bestRight;

        // Iterate over all features.
        for (size_t featureIndex = 0; featureIndex < data[0].features.size(); ++featureIndex) {
            double value;
            std::vector<DataPoint> leftSplit, rightSplit;

            double mse = trySplit(data, featureIndex, value, leftSplit, rightSplit);

            // If found a smaller mse, found a better split
            if (mse < bestMSE) {
                bestMSE = mse;
                bestFeature = featureIndex;
                bestValue = value;
                bestLeft = leftSplit;
                bestRight = rightSplit;
            }
        }

        // If no feature improved the split, this becomes a leaf node.
        if (bestFeature == -1) {
            return new Node(data);
        }

        // Create a new node with the best split and recursively build its left and right subtrees.
        Node* node = new Node(data);
        node->splitFeature = bestFeature;
        node->splitValue = bestValue;
        node->left = buildTree(bestLeft, minSize);
        node->right = buildTree(bestRight, minSize);
        return node;
    }

    double trySplit(const std::vector<DataPoint>& data, size_t featureIndex, double& value,
                    std::vector<DataPoint>& leftSplit, std::vector<DataPoint>& rightSplit) {
        // Placeholder for logic to determine the best split point and calculate MSE.
        return std::numeric_limits<double>::max();
    }

    double predict(const std::vector<double>& features) {
        // Implement logic to traverse the tree and make a prediction
        return 0.0;
    }
};

class RandomForest {
private:
    std::vector<DecisionTree> trees;

public:
    void buildForest(std::vector<DataPoint>& data, int numTrees, int minSamplesSplit) {
        // Implement logic to build the forest of trees
        // This includes the bootstrap sampling of the dataset
    }

    double predict(const std::vector<double>& features) {
        // Aggregate predictions from all trees
        return 0.0;
    }
};

// Example usage:
int main() {
    std::vector<DataPoint> data; // Assume this is populated with your dataset

    int B = 10; // Number of trees
    int minSamplesSplit = 5; // Minimum number of samples to split a node
    RandomForest forest;
    forest.buildForest(data, B, minSamplesSplit);

    std::vector<double> newPointFeatures; // Features of the new data point to predict
    double prediction = forest.predict(newPointFeatures);
    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
}
