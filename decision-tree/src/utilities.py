from math import log

import numpy as np


def calc_accuracy(pred, ground_truth):
    return np.sum(ground_truth == pred) / float(len(ground_truth))


def entropy(x):
    """x should be of type pd.Series"""
    etrpy = 0.0
    num_examples = len(x)
    for c in x.unique():
        x_c = x[x == c]
        p = len(x_c) / float(num_examples)
        etrpy += -p * log(p, 2)

    return etrpy


def entropy_fast(x):
    num_examples = float(len(x))
    bin_count = np.bincount(x)
    cnt = bin_count[bin_count > 0]
    p = cnt / num_examples
    logp = np.log2(p)
    tmp = -p * logp
    return tmp.sum()


def information_gain(values, classes):
    """values should be of type pd.Series"""
    num_examples = float(len(values))
    gain = entropy_fast(classes)

    bin_count = np.bincount(values)
    unique_values, cnt = np.nonzero(bin_count)[0], bin_count[bin_count > 0]
    for v, c in zip(unique_values, cnt):
        gain -= c / num_examples * entropy_fast(classes[values == v])

    return gain
