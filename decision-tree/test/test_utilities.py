import pandas as pd
from pytest import approx

from utilities import entropy, information_gain


def test_entropy():
    values = pd.Series(['s', 's', 'f'])
    assert entropy(values) == approx(0.9182958340544896)


def test_information_gain():
    values = pd.Series(['s', 's', 'f', 's'])
    classes = pd.Series(['A', 'A', 'B', 'B'])
    assert information_gain(values, classes) == approx(0.31127812445913283)