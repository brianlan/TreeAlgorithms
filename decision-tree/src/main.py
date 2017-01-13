import time

import pandas as pd

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
t1 = time.time()
model = DecisionTree(train_X, train_y, min_sample_split=5)
model.train()
print('Training phase took {:.2f} seconds.'.format(time.time() - t1))

t2 = time.time()
pred_y = model.predict(test_X)
acc = calc_accuracy(pred_y, test_y)
print('Prediction phase took {:.2f} seconds, and the accuracy is: {:.3f}'.format(time.time() - t2, acc))
