from abc import abstractmethod
from collections import defaultdict

import numpy as np
from scipy import stats

from utilities import categorical_information_gain, numerical_information_gain, calc_accuracy


class OccurrenceDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def increment_by(self, another_dict):
        for k, v in another_dict.items():
            self[k] += v


class TreeNode(object):
    def __init__(self, attr=None, ref_value=None, path_func=None):
        self.attr = attr
        self.ref_value = ref_value
        self.path_func = path_func
        self.branches = None
        self.predicted_class = None

    def __repr__(self):
        if self.attr is None:
            return 'nil'
        else:
            pred = '' if self.predicted_class is None else '{}<-'.format(self.predicted_class)
            return '{}({}, {})'.format(pred, self.attr, self.ref_value)

    def __str__(self):
        return self.__repr__()

    def get_feature_occurrence(self):
        occurrence = OccurrenceDict(int)

        if self.branches:
            occurrence[self.branches[0].attr] += 1
            for child in self.branches:
                if child.attr is not None:
                    occurrence.increment_by(child.get_feature_occurrence())

        return occurrence


class Algorithm:
    @abstractmethod
    def grow(self, parent, X, y):
        pass

    @abstractmethod
    def find_best_split(self, X, y):
        pass


class ID3(Algorithm):
    def __init__(self, min_sample_split=10):
        self.min_sample_split = min_sample_split

    def grow(self, parent, X, y):
        parent.predicted_class = stats.mode(y).mode[0]

        if X.size == 0 or X.shape[0] <= self.min_sample_split or len(np.unique(y)) == 1:
            return

        best_attr = self.find_best_split(X, y)
        best_attr_array = X[:, best_attr]
        distinct_values = np.unique(best_attr_array)
        parent.branches = []
        for v in distinct_values:
            sub_X = X[best_attr_array == v]
            sub_classes = y[best_attr_array == v]
            sub_tree = TreeNode(attr=best_attr, ref_value='='+str(v), path_func=lambda x, v=v: x == v)
            self.grow(sub_tree, sub_X, sub_classes)
            parent.branches.append(sub_tree)

    def find_best_split(self, X, y):
        best_info_gain = -99999
        best_attr = None
        for col_idx in range(X.shape[1]):
            info_gain = categorical_information_gain(X[:, col_idx], y)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attr = col_idx

        return best_attr


class C45(Algorithm):
    def __init__(self, min_sample_split=10, random_feature_set_size=None):
        self.min_sample_split = min_sample_split
        self.random_feature_set_size = random_feature_set_size

    def _grow_branch(self, parent, X, y, best_attr, ref_value, path_func):
        tree = TreeNode(attr=best_attr, ref_value=ref_value, path_func=path_func)
        sub_idxes = path_func(X[:, best_attr])
        sub_y = y[sub_idxes]
        sub_X = X[sub_idxes, :]
        self.grow(tree, sub_X, sub_y)
        parent.branches.append(tree)

    def grow(self, parent, X, y):
        parent.predicted_class = stats.mode(y).mode[0]

        if X.size == 0 or X.shape[0] <= self.min_sample_split or len(np.unique(y)) == 1:
            return

        best_attr, best_threshold = self.find_best_split(X, y)

        parent.branches = []
        num_samples_lt = np.sum(X[:, best_attr] < best_threshold)
        num_samples_gte = np.sum(X[:, best_attr] >= best_threshold)

        if num_samples_lt > 0:
            self._grow_branch(parent, X, y, best_attr, '<'+str(best_threshold), lambda x, th=best_threshold: x < th)

        if num_samples_gte > 0:
            self._grow_branch(parent, X, y, best_attr, '>='+str(best_threshold), lambda x, th=best_threshold: x >= th)

    def find_best_split(self, X, y):
        if self.random_feature_set_size:
            features_to_use = np.random.choice(X.shape[1], self.random_feature_set_size)
        else:
            features_to_use = range(X.shape[1])

        best_info_gain = -99999
        best_threshold = None
        best_attr = None
        for col_idx in features_to_use:
            unique_values = np.unique(X[:, col_idx])
            for threshold in unique_values:
                info_gain = numerical_information_gain(X[:, col_idx], y, threshold)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_attr = col_idx
                    best_threshold = threshold

        return best_attr, best_threshold


class DecisionTree(object):
    def __init__(self, algorithm=None):
        self.root = TreeNode(attr='<start>', ref_value='<start>', path_func=lambda x: False)
        self.algorithm = ID3() if algorithm is None else algorithm

    def _get_prediction(self, node, x):
        if node.branches:
            for branch in node.branches:
                if branch.path_func(x[branch.attr]):
                    return self._get_prediction(branch, x)

        return node.predicted_class

    def train(self, X, y):
        self.algorithm.grow(self.root, X, y)

    def predict(self, X):
        pred = []
        for row_idx in range(X.shape[0]):
            pred.append(self._get_prediction(self.root, X[row_idx, :]))

        return np.array(pred)

    def get_feature_occurrence(self):
        return self.root.get_feature_occurrence()


class RandomForest:
    def __init__(self, num_trees=4, predict_method='vote', num_bootstrap_samples=None, random_feature_set_size=None):
        self.num_trees = num_trees
        self.predict_method = predict_method
        self.random_feature_set_size = random_feature_set_size
        self.num_bootstrap_samples = num_bootstrap_samples
        self.forest = []

    def train(self, X, y, verbose=False):
        acc_bootstrap_list = []
        acc_out_of_bag_list = []

        for i in range(self.num_trees):
            num_bootstrap_samples = int(X.shape[0] / 5) if self.num_bootstrap_samples is None else self.num_bootstrap_samples
            indices = np.random.choice(X.shape[0],  num_bootstrap_samples, replace=True)

            # create bootstrap data
            X_bootstrap, y_bootstrap = X[indices, :], y[indices]

            # create out-of-bag data
            unique_chosen_indices = np.unique(indices)
            out_of_bag_mask = np.ones(X.shape[0], dtype=bool)
            out_of_bag_mask[unique_chosen_indices] = False
            X_out_of_bag = X[out_of_bag_mask, :]
            y_out_of_bag = y[out_of_bag_mask]

            # train random forest
            tree = DecisionTree(algorithm=C45(min_sample_split=1,
                                              random_feature_set_size=self.random_feature_set_size))
            tree.train(X_bootstrap, y_bootstrap)
            self.forest.append(tree)

            # calculate accuracy and print
            acc_bootstrap = calc_accuracy(tree.predict(X_bootstrap), y_bootstrap)
            acc_out_of_bag = calc_accuracy(tree.predict(X_out_of_bag), y_out_of_bag)

            if verbose:
                print('Tree {}. Bootstrap accuracy: {:.3f}, out-of-bag accuracy: {:.3f}'.format(i,
                                                                                                acc_bootstrap,
                                                                                                acc_out_of_bag))
            acc_bootstrap_list.append(acc_bootstrap)
            acc_out_of_bag_list.append(acc_out_of_bag)

        return acc_bootstrap_list, acc_out_of_bag_list

    def predict(self, X):
        pred = []

        recommendations = np.array([t.predict(X) for t in self.forest])

        if self.predict_method == 'avg':
            for i in range(recommendations.shape[1]):
                pred.append(recommendations[:, i].mean())
        else:
            for i in range(recommendations.shape[1]):
                pred.append(stats.mode(recommendations[:, i]).mode[0])

        return np.array(pred)

    def get_feature_occurrence(self):
        final_feature_occurrence = OccurrenceDict(int)
        for tree in self.forest:
            final_feature_occurrence.increment_by(tree.get_feature_occurrence())

        return final_feature_occurrence
