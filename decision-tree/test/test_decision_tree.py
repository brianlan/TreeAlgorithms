import pandas as pd

from classifier import DecisionTree, TreeNode


def test_train_decision_tree_on_small_data():
    train_X = pd.DataFrame({
        'grade': ['steep', 'steep', 'flat', 'steep'],
        'bumpiness': ['bumpy', 'smooth', 'bumpy', 'smooth'],
        'speed_limit': ['yes', 'yes', 'no', 'no']
    })

    train_y = pd.DataFrame({
        'class': ['slow', 'slow', 'fast', 'fast']
    })

    model = DecisionTree(train_X.values, train_y.values, min_sample_split=1)
    model.train()
    assert len(model.tree.branches) == 2
    assert '<start>=<start>' in str(model.tree)
    assert model.tree.predicted_class is None
    assert model.tree.branches[0].branches is None
    assert model.tree.branches[1].branches is None
    assert model.tree.branches[0].attr == model.tree.branches[1].attr == 2

    if model.tree.branches[0].value == 'yes':
        assert model.tree.branches[0].predicted_class == 'slow'
    else:
        assert model.tree.branches[0].predicted_class == 'fast'

    if model.tree.branches[1].value == 'yes':
        assert model.tree.branches[1].predicted_class == 'slow'
    else:
        assert model.tree.branches[1].predicted_class == 'fast'


# def test_train_decision_tree_on_small_data2():
#     train_X = pd.DataFrame({
#         'grade': ['steep', 'steep', 'flat', 'steep'],
#         'bumpiness': ['bumpy', 'smooth', 'bumpy', 'smooth'],
#     })
#
#     train_y = pd.DataFrame({
#         'class': ['slow', 'slow', 'fast', 'fast']
#     })
#
#     model = DecisionTree(train_X.values, train_y.values, min_sample_split=1)
#     model.train()
#     assert len(model.tree.branches) == 2
#     assert '<start>=<start>' in str(model.tree)
#     assert model.tree.predicted_class is None
#     assert model.tree.branches[0].attr == model.tree.branches[1].attr == 0
#
#     if model.tree.branches[0].value == 'flat':
#         assert model.tree.branches[0].predicted_class == 'fast'
#
#     if model.tree.branches[1].value == 'flat':
#         assert model.tree.branches[1].predicted_class == 'fast'


def test_predict_using_decision_tree():
    model = DecisionTree(None, None, min_sample_split=1)
    model.tree = TreeNode(attr='<start>', value='<start>')

    left_branch = TreeNode(attr=2, value='yes')
    left_branch.predicted_class = 'slow'

    right_branch = TreeNode(attr=2, value='no')
    right_branch.predicted_class = 'fast'

    model.tree.branches = [left_branch, right_branch]

    test_X = pd.DataFrame({
        'grade': ['steep', 'flat'],
        'bumpiness': ['smooth', 'smooth'],
        'speed_limit': ['yes', 'no']
    })
    pred = model.predict(test_X.values)
    assert pred['class'].values.tolist() == ['slow', 'fast']
