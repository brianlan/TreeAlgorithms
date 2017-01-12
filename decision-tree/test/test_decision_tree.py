import pandas as pd

from classifier import DecisionTree


def test_train_decision_tree_on_small_data():
    train_X = pd.DataFrame({
        'grade': ['steep', 'steep', 'flat', 'steep'],
        'bumpiness': ['bumpy', 'smooth', 'bumpy', 'smooth'],
        'speed_limit': ['yes', 'yes', 'no', 'no']
    })

    train_y = pd.DataFrame({
        'class': ['slow', 'slow', 'fast', 'fast']
    })

    model = DecisionTree(train_X, train_y, min_sample_split=1)
    model.train()
    assert len(model.tree.children) == 2
    assert '<start>=<start>' in str(model.tree)
    assert model.tree.predicted_class is None
    assert model.tree.children[0].children is None
    assert model.tree.children[1].children is None
    assert model.tree.children[0].attr == model.tree.children[1].attr == 'speed_limit'

    if model.tree.children[0].value == 'yes':
        assert model.tree.children[0].predicted_class == 'slow'
    else:
        assert model.tree.children[0].predicted_class == 'fast'

    if model.tree.children[1].value == 'yes':
        assert model.tree.children[1].predicted_class == 'slow'
    else:
        assert model.tree.children[1].predicted_class == 'fast'


def test_train_decision_tree_on_small_data2():
    train_X = pd.DataFrame({
        'grade': ['steep', 'steep', 'flat', 'steep'],
        'bumpiness': ['bumpy', 'smooth', 'bumpy', 'smooth'],
    })

    train_y = pd.DataFrame({
        'class': ['slow', 'slow', 'fast', 'fast']
    })

    model = DecisionTree(train_X, train_y, min_sample_split=1)
    model.train()
    assert len(model.tree.children) == 2
    assert '<start>=<start>' in str(model.tree)
    assert model.tree.predicted_class is None
    assert model.tree.children[0].attr == model.tree.children[1].attr == 'grade'

    if model.tree.children[0].value == 'flat':
        assert model.tree.children[0].predicted_class == 'fast'

    if model.tree.children[1].value == 'flat':
        assert model.tree.children[1].predicted_class == 'fast'
