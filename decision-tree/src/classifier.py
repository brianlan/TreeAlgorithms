import pandas as pd
from scipy import stats

from utilities import information_gain


class TreeNode(object):
    def __init__(self, attr=None, value=None, classes=None):
        self.attr = attr
        self.value = value
        self.count_per_class = pd.Series([], name='class') if classes is None else classes['class'].value_counts()
        self.children = None
        self.predicted_class = None

    def __repr__(self):
        if self.attr == None:
            return 'nil'
        else:
            stats = ', '.join(['{}: {}'.format(c, n) for c, n in self.count_per_class.iteritems()])
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

    def id3(self, parent, examples, classes):
        if examples.empty or len(examples) < self.min_sample_split or len(classes['class'].unique()) == 1:
            parent.predicted_class = stats.mode(classes).mode[0, 0]
            return

        best_info_gain = -99999
        best_attr = None
        for attr in examples.columns:
            info_gain = information_gain(examples[attr], classes['class'])
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attr = attr

        distinct_values = examples[best_attr].unique()
        parent.children = []
        for v in distinct_values:
            sub_examples = examples[examples[best_attr] == v].drop(best_attr, axis=1)
            sub_classes = classes[examples[best_attr] == v]
            sub_tree = TreeNode(attr=best_attr, value=v, classes=sub_classes)
            self.id3(sub_tree, sub_examples, sub_classes)
            parent.children.append(sub_tree)

    def train(self):
        self.id3(self.tree, self.X, self.y)

    def predict(self, X):
        pass