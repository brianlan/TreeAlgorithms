import time

import numpy as np
import pandas as pd

from classifier import RandomForest


######################################################
#               Load data
######################################################
train_raw = pd.read_csv('../dataset/train.csv')
test_raw = pd.read_csv('../dataset/test.csv')

X_train, y_train = train_raw.values[:, 1:], train_raw.values[:, 0].astype(int)
X_test = test_raw.values

######################################################
#               Super Parameter Selection
######################################################
# avg_accuracy = {}
# best_avg_accuracy = -1
# best_n_bs_samples = best_n_features = None
#
# for n_bs_samples in [100, 300, 500, 1000, 1500, 2000, 3000]:
#     for n_features in [50, 100, 200, 500, 800, 1000, 1500]:
#         model = RandomForest(num_trees=10, predict_method='avg', num_bootstrap_samples=n_bs_samples,
#                              random_feature_set_size=n_features)
#         acc_bootstrap_list, acc_out_of_bag_list = model.train(X_train, y_train)
#         cur_acc = np.mean(acc_out_of_bag_list)
#         avg_accuracy[(n_bs_samples, n_features)] = cur_acc
#
#         if cur_acc > best_avg_accuracy:
#             best_avg_accuracy = cur_acc
#             best_n_bs_samples = n_bs_samples
#             best_n_features = n_features
#
#         print('n_bs_examples: {}, n_features: {}, avg_accuracy: {}'.format(n_bs_samples, n_features, cur_acc))

######################################################
#               Train model
######################################################
t1 = time.time()
num_trees = 256
num_bootstrap_samples = 800
random_feature_set_size = 400

model = RandomForest(num_trees=num_trees, predict_method='avg', num_bootstrap_samples=num_bootstrap_samples,
                     random_feature_set_size=random_feature_set_size)
# model = RandomForest(num_trees=64, predict_method='avg', num_bootstrap_samples=1000,
#                      random_feature_set_size=500)
model.train(X_train, y_train, verbose=True)
print('Training phase took {:.1f} ms.'.format((time.time() - t1) * 1000))

######################################################
#               Predicting Test Samples
######################################################
t2 = time.time()
pred_y = model.predict(X_test)
print('Prediction phase took {:.1f} ms'.format((time.time() - t2) * 1000))

submission = pd.DataFrame({'MoleculeId': range(1, len(pred_y)+1), 'PredictedProbability': pred_y})
submission.to_csv('../dataset/submission_{}_{}_{}.csv'.format(num_trees, num_bootstrap_samples, random_feature_set_size), index=False)
