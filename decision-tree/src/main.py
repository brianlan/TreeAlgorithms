import pandas as pd
import numpy as np

from classifier import DecisionTree
from utilities import calc_accuracy


######################################################
#               Load data
######################################################
attributes = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']
train_df = pd.read_csv('../dataset/train.txt')
test_df = pd.read_csv('../dataset/test.txt')
train_X, train_y = train_df[attributes], train_df[['class']]
test_X, test_y = test_df[attributes], test_df[['class']]

######################################################
#               Train model
######################################################
model = DecisionTree(train_X, train_y, min_sample_split=10)
model.train()
pred_y = model.predict(test_X)
acc = calc_accuracy(pred_y, test_y)