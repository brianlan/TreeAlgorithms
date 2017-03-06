# Performance
## Dataset
[Predicting a Biological Response](https://www.kaggle.com/c/bioresponse#description)

## Parameter Settings
``` python
num_trees = 1024
random_feature_set_size = 45
```

## Accuracy
It's measured using Log Loss. This code could achieve **0.41085** and **0.45216** on private and public test dataset respectively.

## Time
**Machine Sepcs**
* Macbook Pro
* Processor 2.8 GHz Intel Core i7
* 16 GB 1600 MHz DDR3

It takes roughly **5 seconds** to train a single tree. No parallelization is implemented.