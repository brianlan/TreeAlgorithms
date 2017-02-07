import pandas as pd
import numpy as np
from pytest import approx

from utilities import entropy, information_gain, entropy_fast, calc_accuracy


def test_entropy():
    values = pd.Series([0, 0, 1])
    assert entropy(values) == approx(0.9182958340544896)


def test_entropy_fast():
    values = pd.Series([0, 0, 1])
    assert entropy_fast(values) == approx(0.9182958340544896)


def test_information_gain():
    values = pd.Series([0, 0, 1, 0])
    classes = pd.Series([2, 2, 3, 3])
    assert information_gain(values, classes) == approx(0.31127812445913283)


def test_calc_accurary():
    pred = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    ground_truth = np.array([1, 1, 2, 2, 2, 2, 2, 2, 1, 1])
    acc = calc_accuracy(pred, ground_truth)
    assert acc == 0.5
