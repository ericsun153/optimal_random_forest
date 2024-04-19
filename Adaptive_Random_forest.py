import numpy as np

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index # Index of the feature used for splitting
        self.threshold = threshold # Threshold value for the splitting criterion
        self.left = left # Left child node
        self.right = right # Right child node
        self.value = value  # Output value if the node is a leaf

# Calculate the Mean Squared Error for a given split
def calculate_mse(targets, weights):
    # Calculate the weighted average of the targets
    weighted_mean = np.sum(targets * weights) / np.sum(weights)
    # Calculate the weighted MSE
    mse = np.sum(weights * (targets - weighted_mean) ** 2)
    return mse

# Determine the best feature and threshold to split the dataset
def best_split(X, y, weights, min_samples_split, split_record):
    feature_count = X.shape[1]
    best_feature, best_threshold, best_mse = None, None, float("inf")
    best_split_left, best_split_right = None, None
    
    # Check for the least frequently split feature
    least_splits = min(split_record)
    candidate_features = [i for i, splits in enumerate(split_record) if splits == least_splits]

    # Iterate over candidate features
    for feature in candidate_features:
        thresholds = np.unique(X[:, feature])
        # Iterate over each possible threshold for the current feature
        for threshold in thresholds:
            left_mask = X[:, feature] < threshold
            right_mask = ~left_mask

            if np.sum(left_mask) < min_samples_split or np.sum(right_mask) < min_samples_split:
                continue  # Skip if either side is too small

            # Calculate MSE for the current split
            mse_left = calculate_mse(y[left_mask], weights[left_mask])
            mse_right = calculate_mse(y[right_mask], weights[right_mask])
            mse_total = mse_left + mse_right

            # Update the best split if the current split is better
            if mse_total < best_mse:
                best_feature = feature
                best_threshold = threshold
                best_mse = mse_total
                best_split_left = left_mask
                best_split_right = right_mask

    return best_feature, best_threshold, best_split_left, best_split_right

# Build the decision tree recursively
def build_tree(X, y, weights, min_samples_split, max_samples_leaf, split_record, depth=0):
    # Stopping conditions: few samples or all samples belong to one category
    if len(y) < 2 * min_samples_split or len(set(y)) == 1:
        leaf_value = np.average(y, weights=weights, returned=True)
        return DecisionNode(value=leaf_value)
    # Avoid creating very deep trees
    if depth > max_samples_leaf:
        return DecisionNode(value=np.average(y, weights=weights, returned=True))

    # Find the best split
    feature, threshold, left_mask, right_mask = best_split(X, y, weights, min_samples_split, split_record)

    if feature is None:  # If no valid split was found
        return DecisionNode(value=np.average(y, weights=weights, returned=True))

    # Split the dataset and build left and right subtrees
    left_subtree = build_tree(X[left_mask], y[left_mask], weights[left_mask], min_samples_split, max_samples_leaf, split_record, depth+1)
    right_subtree = build_tree(X[right_mask], y[right_mask], weights[right_mask], min_samples_split, max_samples_leaf, split_record, depth+1)

    # Record the split
    split_record[feature] += 1

    # Return the current node with its left and right children
    return DecisionNode(feature_index=feature, threshold=threshold, left=left_subtree, right=right_subtree)

# Predict the output for a single data point by traversing the tree
def predict_single(tree, x):
    # Traverse the tree until a leaf node is reached
    while tree.value is None:
        # Go to the left or right child depending on the feature value and threshold
        if x[tree.feature_index] < tree.threshold:
            tree = tree.left
        else:
            tree = tree.right
    return tree.value

# Predict outputs for a dataset
def predict(tree, X):
    # Apply the single point prediction function to each data point in the dataset
    return np.array([predict_single(tree, x) for x in X])

class AdaptiveSplitBalancingForest:
    def __init__(self, B=100, alpha=0.5, omega=0.5, k=1):
        # Initialize the forest with a specified number of trees (B), 
        # the weight for the complement set (alpha), the proportion of data
        # used for the main set (omega), and the minimum samples in each leaf node (k).
        self.B = B # Number of trees
        self.alpha = alpha  # Weight for the complement set
        self.omega = omega # Proportion for the main set
        self.k = k # Minimum samples per leaf
        self.trees = [] # List to hold the individual trees

    def fit(self, X, y):
        N, _ = X.shape  # Total number of samples in the data
        luvN = int(self.omega * N) # Number of samples in the main set

        # Loop to create and train each tree in the forest
        for _ in range(self.B):
            indices = np.random.permutation(N) # Randomly permute the indices
            T_luv = indices[:luvN] # Indices for the main set
            T_luv_complement = indices[luvN:] # Indices for the complement set

            # Create weights for samples
            weights = np.ones(N)
            weights[T_luv_complement] *= self.alpha

            # Create a record to track splits per feature
            split_record = np.zeros(X.shape[1])

            # Build a single decision tree on the main set with adjusted weights
            tree = build_tree(X[T_luv], y[T_luv], weights[T_luv], self.k, 2*self.k - 1, split_record)
            self.trees.append(tree)

    def predict(self, X):
        # Aggregate predictions from all the trees by averaging them
        tree_predictions = np.array([predict(tree, X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0) # Average predictions across trees

# Example usage:
# asbf = AdaptiveSplitBalancingForest(B=10, alpha=0.5, omega=1, k=5)
# asbf.fit(X_train, y_train)
# predictions = asbf.predict(X_test)