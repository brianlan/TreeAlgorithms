from abc import abstractmethod

import numpy as np
from scipy import stats

from utilities import information_gain


class TreeNode(object):
    def __init__(self, attr=None, value=None, classes=None):
        self.attr = attr
        self.value = value
        self.branches = None
        self.predicted_class = None

        if classes is None:
            self.unique_class, self.count_per_class = np.array([]), np.array([])
        else:
            self.unique_class, self.count_per_class = np.unique(classes, return_counts=True)

    def __repr__(self):
        if self.attr is None:
            return 'nil'
        else:
            stats = ', '.join(['{}: {}'.format(c, n) for c, n in zip(self.unique_class, self.count_per_class)])
            pred = '' if self.predicted_class is None else '{}<-'.format(self.predicted_class)
            return '{}{}={} ({})'.format(pred, self.attr, self.value, stats)

    def __str__(self):
        return self.__repr__()


class Algorithm:
    @abstractmethod
    def eval(self, parent, X, y):
        pass

    @abstractmethod
    def find_best_split(self, X, y):
        pass


class ID3(Algorithm):
    def __init__(self, min_sample_split=10):
        self.min_sample_split = min_sample_split

    def eval(self, parent, X, y):
        parent.predicted_class = stats.mode(y).mode[0]

        if X.size == 0 or X.shape[0] <= self.min_sample_split or len(np.unique(y)) == 1:
            return

        best_attr = self.find_best_split(X, y)
        best_attr_array = X[:, best_attr]
        distinct_values = np.unique(best_attr_array)
        parent.branches = []
        for v in distinct_values:
            # sub_X = X[best_attr_array == v].drop(best_attr, axis=1)
            sub_X = X[best_attr_array == v]
            sub_classes = y[best_attr_array == v]
            sub_tree = TreeNode(attr=best_attr, value=v, classes=sub_classes)
            self.eval(sub_tree, sub_X, sub_classes)
            parent.branches.append(sub_tree)

    def find_best_split(self, X, y):
        best_info_gain = -99999
        best_attr = None
        for col_idx in range(X.shape[1]):
            info_gain = information_gain(X[:, col_idx], y)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attr = col_idx

        return best_attr


class C45(Algorithm):
    def __init__(self, min_sample_split=10):
        self.min_sample_split = min_sample_split

    def eval(self, parent, X, y):
        parent.predicted_class = stats.mode(y).mode[0]

        if X.size == 0 or X.shape[0] <= self.min_sample_split or len(np.unique(y)) == 1:
            return

        best_attr = self.find_best_split(X, y)
        best_attr_array = X[:, best_attr]
        distinct_values = np.unique(best_attr_array)
        parent.branches = []
        for v in distinct_values:
            # sub_X = X[best_attr_array == v].drop(best_attr, axis=1)
            sub_X = X[best_attr_array == v]
            sub_classes = y[best_attr_array == v]
            sub_tree = TreeNode(attr=best_attr, value=v, classes=sub_classes)
            self.eval(sub_tree, sub_X, sub_classes)
            parent.branches.append(sub_tree)

    def find_best_split(self, X, y):
        best_info_gain = -99999
        best_attr = None
        for col_idx in range(X.shape[1]):
            info_gain = information_gain(X[:, col_idx], y)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attr = col_idx

        return best_attr


class DecisionTree(object):
    def __init__(self, X, y, algorithm=None, min_sample_split=10):
        self.X = X
        self.y = y
        self.tree = TreeNode(attr='<start>', value='<start>', classes=self.y)
        self.algorithm = ID3(min_sample_split) if algorithm is None else algorithm

    def _get_prediction(self, node, x):
        if node.branches:
            for branch in node.branches:
                if x[branch.attr] == branch.value:
                    return self._get_prediction(branch, x)

        return node.predicted_class

    def train(self):
        self.algorithm.eval(self.tree, self.X, self.y)

    def predict(self, X):
        pred = []
        for row_idx in range(X.shape[0]):
            pred.append(self._get_prediction(self.tree, X[row_idx, :]))

        return np.array(pred)
