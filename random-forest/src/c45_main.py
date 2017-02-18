import time

import pandas as pd

from classifier import C45, DecisionTree


######################################################
#               Load data
######################################################
train_raw = pd.read_csv('../dataset/train.csv')
test_raw = pd.read_csv('../dataset/test.csv')

X_train, y_train = train_raw.values[:, 1:], train_raw.values[:, 0].astype(int)
X_test = test_raw.values[:, 1:]

######################################################
#               Train model
######################################################
t1 = time.time()
model = DecisionTree(algorithm=C45(min_sample_split=10))
model.train(X_train, y_train)
print('Training phase took {:.1f} ms.'.format((time.time() - t1) * 1000))

######################################################
#               Predicting Test Samples
######################################################
t2 = time.time()
pred_y = model.predict(X_test)
print('Prediction phase took {:.1f} ms'.format((time.time() - t2) * 1000))

submission = pd.DataFrame({'MoleculeId': range(1, len(pred_y)+1), 'PredictedProbability': pred_y})
submission.to_csv('../dataset/submission.csv', index=False)
