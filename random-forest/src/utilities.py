import numpy as np


def calc_accuracy(pred, ground_truth):
    return np.sum(ground_truth == pred) / float(len(ground_truth))


def find_mis_classified_samples(pred, ground_truth):
    for i in range(len(ground_truth)):
        if pred[i] != ground_truth[i]:
            print('[{}] ground_truth: {}, pred: {}'.format(i, ground_truth[i], pred[i]))


def entropy(x):
    if len(x) == 0:
        return 0

    num_examples = float(len(x))
    bin_count = np.bincount(x)
    cnt = bin_count[bin_count > 0]
    p = cnt / num_examples
    logp = np.log2(p)
    tmp = -p * logp
    return tmp.sum()


def categorical_information_gain(values, classes):
    num_examples = float(len(values))
    gain = entropy(classes)

    bin_count = np.bincount(values)
    unique_values, cnt = np.nonzero(bin_count)[0], bin_count[bin_count > 0]
    for v, c in zip(unique_values, cnt):
        gain -= c / num_examples * entropy(classes[values == v])

    return gain


def numerical_information_gain(values, classes, threshold):
    num_examples = float(len(values))
    gain = entropy(classes)

    classes_lt = classes[values < threshold]
    classes_gte = classes[values >= threshold]

    gain -= len(classes_lt) / num_examples * entropy(classes_lt)
    gain -= len(classes_gte) / num_examples * entropy(classes_gte)

    return gain
