import numpy as np
import pandas as pd
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None, mode='gini'):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                                max_depth=self.max_depth, mode=mode)
            self.trees.append(tree)

    def fit(self, X, y):
        # split data into n_trees subsets
        n_samples, n_features = X.shape
        # add index column to X
        X_with_idx = np.hstack((X, np.arange(n_samples).reshape(-1, 1)))
        n_samples_per_tree = n_samples // self.n_trees
        for tree in self.trees:
            idxs = np.random.choice(n_samples, size=n_samples_per_tree, replace=True)
            X_subset_with_id = X_with_idx[idxs]
            X_subset = X_subset_with_id[:, :-1]
            y_subset = y[idxs]
            tree.fit(X_subset, y_subset)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [max(set(row), key=list(row).count) for row in tree_preds]
        return np.array(y_pred)