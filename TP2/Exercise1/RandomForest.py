from collections import Counter
from ID3 import DecisionTree
import numpy as np

class RandomForest:
    def __init__(self, n_trees = 10, min_samples_split = 2, max_depth = 10, n_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split = self.min_samples_split, max_depth = self.max_depth, n_features = self.n_features)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace = True)
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(tree_pred) for tree_pred in tree_preds])
        return predictions
    
    def get_Total_nodes(self):
        return sum([tree.get_Total_nodes() for tree in self.trees])
    
    def get_mean_nodes_per_tree(self):
        return self.get_Total_nodes() / self.n_trees

