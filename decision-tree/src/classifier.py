class DecisionTree(object):
    def __init__(self, X, y, min_sample_split=10):
        self.X = X
        self.y = y
        self.min_sample_split = min_sample_split

    def predict(self, X, y):
        pass