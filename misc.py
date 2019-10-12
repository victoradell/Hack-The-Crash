#%% Imports
import pandas as pd
from sklearn import preprocessing
import numpy as np
import pylab 
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import scalers
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.linear_model import LinearRegression
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

#%% Rank gauss example
for i in [80, 193]:
    measurements = train[:, i]
    mean = np.mean(measurements)
    sd = np.std(measurements)
    pylab.suptitle(f"var{i:d}")
    pylab.subplot(2,2,1)
    pylab.hist(measurements, density=True, bins='auto')
    pylab.title(f"Hist mean={mean:.2f} sd={sd:.2f}")
    pylab.subplot(2,2,2)
    stats.probplot(measurements, dist="norm", plot=pylab)
    pylab.title(f"Normal qqplot")
    pylab.subplot(2,2,3)
    measurements = np.expand_dims(measurements, axis=1)
    z = scalers.rank_gauss().fit(measurements).transform(measurements)
    z = np.squeeze(z)
    pylab.hist(z, density=True, bins='auto')
    meanz = np.mean(z)
    sdz = np.std(z)
    pylab.title(f"Hist rkg mean={meanz:.2f} sd={sdz:.2f}")
    pylab.subplot(2,2,4)
    stats.probplot(z, dist="norm", plot=pylab)
    pylab.title(f"Normal qqplot rkg")
    pylab.tight_layout(pad=2,h_pad=0.5)
    pylab.savefig(f"preprocessing/var{i:d}_rkgauss.png", dpi=80)
    pylab.clf()

#%% ROC auc plots generation
train = np.load('./data/train.npy')
X = train[:,1:]
t = train[:,0]
models = [logisticregression.logisticregression, nb.gaussnb, lda.lda]
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.1, random_state=42)
probas_arr = np.empty((t_test.shape[0], len(models)))
fpr = defaultdict(list)
tpr = defaultdict(list)
roc_auc = defaultdict(list)
for i in range(len(models)):
        model = models[i]()
        probas_arr[:,i] = model.fit(X_train,t_train,X_test,t_test).predict_proba(X_test)[:,1]
        fpr[i], tpr[i], _ = roc_curve(t_test, probas_arr[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

pylab.figure()
lw = 2
for i in range(len(models)):
    pylab.plot(fpr[i], tpr[i], color=['red','green','blue'][i], lw=lw, label=f'{models[i]().name} area = {roc_auc[i]:.6f}')
pylab.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
pylab.xlim([0.0, 1.0])
pylab.ylim([0.0, 1.05])
pylab.xlabel('False Positive Rate')
pylab.ylabel('True Positive Rate')
pylab.title('Receiver operating characteristic example')
pylab.legend(loc="lower right")
pylab.savefig(f'showcasing/sampleROCAUCgraph.png', dpi=80)
pylab.show()