from math import log


def calc_accuracy(pred, ground_truth):
    pass


def entropy(x):
    """x should be of type pd.Series"""
    etrpy = 0.0
    num_examples = len(x)
    for c in x.unique():
        x_c = x[x == c]
        p = len(x_c) / float(num_examples)
        etrpy += -p * log(p, 2)

    return etrpy

def information_gain(values, classes):
    """values should be of type pd.Series"""
    num_examples = float(len(values))
    gain = entropy(classes)
    for v in values.unique():
        classes_v = classes[values == v]
        gain -= len(classes_v) / num_examples * entropy(classes_v)

    return gain