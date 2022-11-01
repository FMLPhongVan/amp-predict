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
    def __init__(self, max_depth=5, min_samples_split=2, min_impurity=1e-7, mode='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
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

    def _grow_tree(self, dataset, depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and depth<=self.max_depth:
            # find the best split
            feature_index, threshold, left, right, info_gain = self._best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if (info_gain > 0 and self.mode == 'entropy') or (info_gain >= 0 and self.mode == 'gini'):
                # recur left
                left_subtree = self._grow_tree(left, depth + 1)
                # recur right
                right_subtree = self._grow_tree(right, depth + 1)
                # return decision node
                return Node(feature_index, threshold, left_subtree, right_subtree, info_gain)

        # compute leaf node
        leaf_value = self._most_common_label(Y)
        # return leaf node
        return Node(value=leaf_value)

    def _best_split(self, dataset, num_samples, num_features):
        max_info_gain = -float('inf')
        #if self.mode == 'gini': max_info_gain = -max_info_gain
        ans_feature_idx = None
        ans_threshold = None
        ans_dataset_left = None
        ans_dataset_right = None
        ans_info_gain = max_info_gain
        for feature_idx in range(num_features):
            feature_values = dataset[:, feature_idx]
            unique_values = np.unique(feature_values)
            for threshold in unique_values:
                dataset_left, dataset_right = self._split(dataset, feature_idx, threshold)
                if len(dataset_left) == 0 or len(dataset_right) == 0:
                    continue
                y_left = dataset_left[:, -1]
                y_right = dataset_right[:, -1]
                info_gain = self._information_gain(dataset[:, -1], y_left, y_right, self.mode)

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    ans_feature_idx = feature_idx
                    ans_threshold = threshold
                    ans_dataset_left = dataset_left
                    ans_dataset_right = dataset_right
                    ans_info_gain = max_info_gain
        return ans_feature_idx, ans_threshold, ans_dataset_left, ans_dataset_right, ans_info_gain

    def _split(self, dataset, feature_idx, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_idx] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_idx] > threshold])
        return dataset_left, dataset_right

    def _information_gain(self, y, y_left, y_right, criterion = 'entropy'):
        if criterion == 'entropy':
            return self._entropy(y) - (len(y_left) / len(y)) * self._entropy(y_left) - (len(y_right) / len(y)) * self._entropy(y_right)
        elif criterion == 'gini':
            return 1 - (len(y_left) / len(y)) * self._gini_index(y_left) - (len(y_right) / len(y)) * self._gini_index(y_right)





    def _most_common_label(self, y):
        y = list(y)
        return max(y, key=y.count)

    def _entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p = len(y[y == cls]) / len(y)
            entropy += -p * np.log2(p)
        return entropy

    def _gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p = len(y[y == cls]) / len(y)
            gini += p**2
        return 1 - gini
