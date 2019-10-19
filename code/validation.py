import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score,f1_score,precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import datetime
import pylab
from collections import defaultdict

import gradboost

from master import crossvalidation, splitvalidation

train = np.load('preprocessed.npy')

# Gradient boost validation
for n_estimators in [50]:
        for max_depth in [6]:
                gradboost.n_estimators=n_estimators
                gradboost.max_depth=max_depth
                print(gradboost.n_estimators, gradboost.max_depth)
                
                model_ls = [gradboost.gradboost]
                scaler_ls = [None]
                splitvalidation(models=model_ls, data=train, scalers=scaler_ls)

