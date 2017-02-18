import numpy as np

from classifier import C45
from classifier import DecisionTree


def test_train_decision_tree_on_small_data():
    train_X = np.array([
        [1, 3],
        [0, 2],
        [0, 1],
        [0.5, 0.5],
        [1, 0],
        [3, 3],
        [3, 2],
        [3, 1],
        [3, 0]
    ])
    train_y = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])

    model = DecisionTree(train_X, train_y, algorithm=C45(min_sample_split=1))
    model.train()

    assert len(model.root.branches) == 2

    assert len(model.root.branches[0].branches) == 2
    assert len(model.root.branches[1].branches) == 2

    assert model.root.branches[0].branches[0].predicted_class == 0
    assert model.root.branches[0].branches[1].predicted_class == 1
    assert model.root.branches[1].branches[0].predicted_class == 1
    assert model.root.branches[1].branches[1].predicted_class == 0
