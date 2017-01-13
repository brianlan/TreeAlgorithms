import time

import pandas as pd
import numpy as np
from sklearn import tree, preprocessing

from utilities import calc_accuracy


######################################################
#                    Load data
######################################################
attributes = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']
train_df = pd.read_csv('../dataset/train.txt')
test_df = pd.read_csv('../dataset/test.txt')
train_X, train_y = train_df[attributes].values, train_df[['class']].values.tolist()
test_X, test_y = test_df[attributes].values, test_df[['class']]

######################################################
#   Transform data - from categorical to numerical
######################################################
t0 = time.time()
le_feature = preprocessing.LabelEncoder()
le_label = preprocessing.LabelEncoder()
le_feature.fit(train_X.reshape(np.prod(train_X.shape)))
le_label.fit(train_y)
train_X_numerical = le_feature.transform(train_X.reshape(np.prod(train_X.shape))).reshape(train_X.shape)
train_y_numerical = le_label.transform(train_y)
test_X_numerical = le_feature.transform(test_X.reshape(np.prod(test_X.shape))).reshape(test_X.shape)
test_y_numerical = le_label.transform(test_y)
print('Transforming data took {:.1f} ms'.format((time.time() - t0) * 1000))

######################################################
#                    Train model
######################################################
t1 = time.time()
clf = tree.DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(train_X_numerical, train_y_numerical)
print('Training phase took {:.1f} ms.'.format((time.time() - t1) * 1000))

t2 = time.time()
pred_y_numerical = clf.predict(test_X_numerical)
pred_y = le_label.inverse_transform(pred_y_numerical)
acc = calc_accuracy(pd.DataFrame({'class': pred_y}), test_y)
print('Prediction phase took {:.1f} ms, and the accuracy is: {:.3f}'.format((time.time() - t2) * 1000, acc))
