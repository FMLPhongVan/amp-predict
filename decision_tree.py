import numpy as np

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecisionTree():
    def __init__(self, max_depth=5, min_samples_split=2, mode='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.mode = mode

    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self._grow_tree(dataset)

    def predict(self, dataset):
        return [self._traverse_tree(x, self.root) for x in dataset]

    def print_tree(self):
        self._print_tree(self.root)

    def _print_tree(self, tree, indent=' '):
        if tree.value is not None:
            print(tree.value)
        else:
            print("X{} <= {}".format(tree.feature, tree.threshold))
            print(indent + "T->", end='')
            self._print_tree(tree.left, indent + indent)
            print(indent + "F->", end='')
            self._print_tree(tree.right, indent + indent)

    def print_tree_to_file(self, filename):
        with open(filename, 'w') as f:
            self._print_tree_to_file(self.root, f)

    def _print_tree_to_file(self, tree, f, indent=' '):
        if tree.value is not None:
            f.write(str(tree.value))
        else:
            f.write("X{} <= {}\n".format(tree.feature, tree.threshold))
            f.write(indent + "T->")
            self._print_tree_to_file(tree.left, f, indent + ' ')
            f.write("\n")
            f.write(indent + "F->")
            self._print_tree_to_file(tree.right, f, indent + ' ')

    def _traverse_tree(self, X, tree):
        if tree.value is not None:
            return tree.value
        feature_value = X[tree.feature]
        if feature_value <= tree.threshold:
            return self._traverse_tree(X, tree.left)
        else:
            return self._traverse_tree(X, tree.right)

    def _grow_tree(self, dataset, depth = 0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_features == 1:
            return Node(value=self._most_common_label(Y))

        if num_samples >= self.min_samples_split and depth < self.max_depth:
            feature_id, threshold, dataset_left, dataset_right = self._get_best_split(dataset, num_samples, num_features)

            if dataset_left is not None and dataset_right is not None:
                left = self._grow_tree(dataset_left, depth + 1)
                right = self._grow_tree(dataset_right, depth + 1)
                return Node(feature_id, threshold, left, right)

        leaf_value = self._most_common_label(Y)
        return Node(value=leaf_value)

    def _most_common_label(self, y):
        y = list(y)
        return max(y, key=y.count)

    def _get_best_split(self, dataset, num_samples, num_features):
        best_split_scores = 1
        if self.mode == 'entropy':
            best_split_scores = -float('inf')

        best_feature_id = None
        best_threshold = None
        best_left = None
        best_right = None

        for feature_id in range(num_features):
            feature_values = dataset[:, feature_id]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                dataset_left, dataset_right = self._split(dataset, feature_id, threshold)
                if len(dataset_left) == 0 or len(dataset_right) == 0: continue
                parent, left, right = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                info_gain = self._information_gain(parent, left, right)

                if self._better(info_gain, best_split_scores):
                    best_split_scores = info_gain
                    best_feature_id = feature_id
                    best_threshold = threshold
                    best_left = dataset_left
                    best_right = dataset_right

        return best_feature_id, best_threshold, best_left, best_right

    def _split(self, dataset, feature_id, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_id] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_id] > threshold])
        return dataset_left, dataset_right

    def _information_gain(self, parent, left, right):
        if self.mode == 'entropy':
            return self._entropy(parent) - (len(left) / len(parent)) * self._entropy(left) - (len(right) / len(parent)) * self._entropy(right)
        else:
            return (len(left) / len(parent)) * self._gini_index(left) + (len(right) / len(parent)) * self._gini_index(right)
    def _better(self, a, b):
        if self.mode == 'entropy':
            return a > b
        else:
            return a < b

    def _gini_index(self, y):
        classes = np.unique(y)
        gini = 0
        for label in classes:
            p = len(y[y == label]) / len(y)
            gini += p * p
        return 1 - gini

    def _entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for label in classes:
            p = len(y[y == label]) / len(y)
            entropy += -p * np.log2(p)
        return entropy

