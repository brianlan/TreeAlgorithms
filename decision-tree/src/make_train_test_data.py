import pandas as pd
import numpy as np

NUM_TRAIN = 10000
col_names = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']

all = pd.read_csv('../dataset/nursery.data.txt', header=None)
np.random.seed(1024)
rand_perm = np.random.permutation(len(all))
train, test = all.loc[rand_perm[:NUM_TRAIN]], all.loc[rand_perm[NUM_TRAIN:]]
train.columns = test.columns = col_names
train.to_csv('../dataset/train.txt', index=False)
test.to_csv('../dataset/test.txt', index=False)
