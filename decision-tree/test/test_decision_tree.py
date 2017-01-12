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
    pass