#%% Imports
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score,f1_score,precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import datetime
import pylab
from collections import defaultdict

import scalers
import lda
import qda
import nb
import logisticregression
import nnet
import randomforest

from master import crossvalidation, splitvalidation
train = np.load('./data/train.npy')

#%% Logistic regression validation

for C in [1e-18,1e-12,1e-6,1,1e6]:
        logisticregression.C = C
        model_ls = [logisticregression.logisticregression]
        scaler_ls = [None]
        crossvalidation(models=model_ls, data=train, scalers=scaler_ls)
        
        model_ls = [logisticregression.logisticregression]
        scaler_ls = [scalers.rank_gauss]
        splitvalidation(models=model_ls, data=train, scalers=scaler_ls)

#%% LDA/QDA validation

model_ls = [lda.lda]
scaler_ls = [None]
crossvalidation(models=model_ls, data=train, scalers=scaler_ls)

model_ls = [qda.qda]
scaler_ls = [None]
crossvalidation(models=model_ls, data=train, scalers=scaler_ls)

model_ls = [lda.lda]
scaler_ls = [scalers.rank_gauss]
splitvalidation(models=model_ls, data=train, scalers=scaler_ls)

model_ls = [qda.qda]
scaler_ls = [scalers.rank_gauss]
splitvalidation(models=model_ls, data=train, scalers=scaler_ls)

#%% Gaussian Naive Bayes validation

model_ls = [nb.gaussnb]
scaler_ls = [None]
crossvalidation(models=model_ls, data=train, scalers=scaler_ls)

model_ls = [nb.gaussnb]
scaler_ls = [scalers.rank_gauss]
splitvalidation(models=model_ls, data=train, scalers=scaler_ls)

#%% Random forest validation

train = np.load('./data/train.npy')
X = train[:,1:]
t = train[:,0]

for n_estimators in [20,100,500]:
        for max_depth in [1,2,5,None]:
                randomforest.n_estimators=n_estimators
                randomforest.max_depth=max_depth
                
                model_ls = [randomforest.random_forest]
                scaler_ls = [None]
                splitvalidation(models=model_ls, data=train, scalers=scaler_ls)

for n_estimators in [20,100]:
        for max_depth in [1,2,5,None]:
                randomforest.n_estimators=n_estimators
                randomforest.max_depth=max_depth                
                model_ls = [randomforest.random_forest]
                scaler_ls = [scalers.rank_gauss]
                splitvalidation(models=model_ls, data=train, scalers=scaler_ls)

randomforest.n_estimators=1000
randomforest.max_depth=1
model_ls = [randomforest.random_forest]
scaler_ls = [None]
splitvalidation(models=model_ls, data=train, scalers=scaler_ls)

randomforest.n_estimators=5000
randomforest.max_depth=1
model_ls = [randomforest.random_forest]
scaler_ls = [None]
splitvalidation(models=model_ls, data=train, scalers=scaler_ls)

#%% NNet validation

train = np.load('./data/train.npy')
X = train[:,1:]
t = train[:,0]

for n_layers in [3,4,5]:
                for dropout_rate in [0,0.3,0.6,0.9]:
                        nnet.n_layers = n_layers
                        nnet.dropout_rate = dropout_rate
                        model_ls = [nnet.net]
                        scaler_ls = [scalers.standardizer]
                        splitvalidation(models=model_ls, data=train, scalers=scaler_ls)

for n_layers in [3,4,5]:
                for dropout_rate in [0,0.3,0.6,0.9]:
                        nnet.n_layers = n_layers
                        nnet.dropout_rate = dropout_rate
                        model_ls = [nnet.net]
                        scaler_ls = [scalers.rank_gauss]
                        splitvalidation(models=model_ls, data=train, scalers=scaler_ls)
