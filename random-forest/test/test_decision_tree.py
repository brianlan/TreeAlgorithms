import numpy as np

from classifier import C45
from classifier import DecisionTree
from classifier import TreeNode


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

    model = DecisionTree(algorithm=C45(min_sample_split=1))
    model.train(train_X, train_y)

    assert len(model.root.branches) == 2

    assert len(model.root.branches[0].branches) == 2
    assert len(model.root.branches[1].branches) == 2

    assert model.root.branches[0].branches[0].predicted_class == 0
    assert model.root.branches[0].branches[1].predicted_class == 1
    assert model.root.branches[1].branches[0].predicted_class == 1
    assert model.root.branches[1].branches[1].predicted_class == 0


def test_get_feature_occurrence():
    root = TreeNode(attr='<start>', ref_value='<start>', path_func=lambda x: False)
    lvl_1_1 = TreeNode(attr=0, ref_value=1, path_func=lambda x: x < 1)
    lvl_1_2 = TreeNode(attr=0, ref_value=1, path_func=lambda x: x >= 1)
    root.branches = [lvl_1_1, lvl_1_2]
    lvl_1_1.branches = [TreeNode(attr=1, ref_value=3, path_func=lambda x: x < 3),
                        TreeNode(attr=1, ref_value=3, path_func=lambda x: x >= 3)]

    lvl_2_2 = TreeNode(attr=1, ref_value=2, path_func=lambda x: x >= 2)
    lvl_2_2.branches = [TreeNode(attr=0, ref_value=1.5, path_func=lambda x: x < 1.5),
                        TreeNode(attr=0, ref_value=1.5, path_func=lambda x: x >= 1.5)]

    lvl_1_2.branches = [TreeNode(attr=1, ref_value=2, path_func=lambda x: x < 2), lvl_2_2]

    feature_occurrence = root.get_feature_occurrence()

    assert feature_occurrence[0] == 2
    assert feature_occurrence[1] == 2
