import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        # Stop if max depth is reached or if all samples have the same label
        if (self.max_depth is not None and depth >= self.max_depth) or len(set(y)) == 1:
            return self._leaf(y)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return self._leaf(y)

        # Split the data
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_subtree, "right": right_subtree}

    def _leaf(self, y):
        # Leaf node with the most common class label
        return np.argmax(np.bincount(y))

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        # Calculate information gain of a potential split
        parent_entropy = self._entropy(y)

        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0

        n = len(y)
        n_left, n_right = np.sum(left_indices), np.sum(right_indices)
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])

        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        return parent_entropy - child_entropy

    def _entropy(self, y):
        # Calculate the entropy of a label distribution
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _predict_sample(self, x, tree):
        # Predict class label for a single sample
        if not isinstance(tree, dict):
            return tree
        if x[tree["feature"]] < tree["threshold"]:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])
