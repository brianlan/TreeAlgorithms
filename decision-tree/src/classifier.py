import pandas as pd
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


class DecisionTree(object):
    def __init__(self, X, y, min_sample_split=10):
        self.X = X
        self.y = y
        self.min_sample_split = min_sample_split
        self.tree = TreeNode(attr='<start>', value='<start>', classes=self.y)

    def _id3(self, parent, examples, classes):
        if examples.size == 0 or examples.shape[0] <= self.min_sample_split or len(np.unique(classes)) == 1:
            parent.predicted_class = stats.mode(classes).mode[0]
            return

        best_info_gain = -99999
        best_attr = None
        for col_idx in range(examples.shape[1]):
            info_gain = information_gain(examples[:, col_idx], classes)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attr = col_idx

        best_attr_array = examples[:, best_attr]
        distinct_values = np.unique(best_attr_array)
        parent.branches = []
        for v in distinct_values:
            # sub_examples = examples[best_attr_array == v].drop(best_attr, axis=1)
            sub_examples = examples[best_attr_array == v]
            sub_classes = classes[best_attr_array == v]
            sub_tree = TreeNode(attr=best_attr, value=v, classes=sub_classes)
            self._id3(sub_tree, sub_examples, sub_classes)
            parent.branches.append(sub_tree)

    def _get_class(self, node, x):
        if node.predicted_class is not None:
            return node.predicted_class

        for branch in node.branches:
            if x[branch.attr] == branch.value:
                return self._get_class(branch, x)

    def train(self):
        self._id3(self.tree, self.X, self.y)

    def predict(self, X):
        pred = []
        for row_idx in range(X.shape[0]):
            pred.append(self._get_class(self.tree, X[row_idx, :]))

        return np.array(pred)
