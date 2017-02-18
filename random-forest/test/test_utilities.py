import numpy as np
from pytest import approx

from utilities import numerical_information_gain


def test_numerical_information_gain1():
    values = np.array([1, 0, 0, 0.5, 1, 3, 3, 3, 3])
    classes = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])
    threshold = 2
    information_gain = numerical_information_gain(values, classes, threshold)
    assert information_gain == approx(0.091091)


def test_numerical_information_gain2():
    values = np.array([1, 0, 0, 0.5, 1, 3, 3, 3, 3])
    classes = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])
    threshold = 4
    information_gain = numerical_information_gain(values, classes, threshold)
    assert information_gain == 0.0
