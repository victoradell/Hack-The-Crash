import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score,f1_score,precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import datetime
import pylab
from collections import defaultdict
import os

import scalers
from lda import lda
from qda import qda
from nb import gaussnb, gaussnb_weighted
from logisticregression import logisticregression
from nnet import net
from randomforest import random_forest


scores_file_addr = 'scores.txt'
scoresv_file_addr = 'scoresv.txt'

def crossvalidation(models, data, n_splits=10, n_repeats=1, name=None, scalers=None):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    X = data[:,1:]
    t = data[:,0]
    roc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for train,test in cv.split(X,t):
        probas_arr = np.empty((t[test].shape[0], len(models)))   
        for i in range(len(models)):
            if scalers != None and scalers[i] != None:
                model = models[i](scalers[i]())
                if i==0 and name == None:
                    name = model.name
            else:
                model = models[i]()
                if i==0 and name == None:
                    name = model.name
            probas_arr[:,i] = model.fit(X[train],t[train]).predict_proba(X[test])[:,1]
        probas = np.mean(probas_arr, axis = 1)
        np.save(os.path.join('prediction',f'{name}'), probas)
        np.savetxt(os.path.join('prediction',f'{name}'), probas, delimiter='\n')
        class_pred = np.where(probas > 0.5, 1, 0)
        #print('class_pred number of positives:',sum(class_pred))
        roc_scores.append(roc_auc_score(t[test], probas))
        f1_scores.append(f1_score(t[test], class_pred))
        precision_scores.append(precision_score(t[test], class_pred))
        recall_scores.append(recall_score(t[test], class_pred))
    scores_file = open(scores_file_addr, "a+")
    scores_file.write(name + " " + "{:.6f}".format(np.nanmean(roc_scores)) + " " + datetime.datetime.now().strftime("%H:%M %d-%m") + " " + str(n_splits) + 'x' + str(n_repeats) + '\n')
    scores_file.close()
    scoresv_file = open(scoresv_file_addr, "a+")
    scoresv_file.write(name + " " + datetime.datetime.now().strftime("%H:%M %d-%m") + '\n')
    tab = "    "
    scoresv_file.write(tab + "roc_score: " + "{:.6f}".format(np.nanmean(roc_scores)) + tab + "{:.6f}".format(np.nanstd(roc_scores)) + '\n')
    scoresv_file.write(tab + "f1_score:  " + "{:.6f}".format(np.nanmean(f1_scores)) + tab + "{:.6f}".format(np.nanstd(f1_scores)) + '\n')
    scoresv_file.write(tab + "pre_score: " + "{:.6f}".format(np.nanmean(precision_scores)) + tab + "{:.6f}".format(np.nanstd(precision_scores)) + '\n')
    scoresv_file.write(tab + "rec_score: " + "{:.6f}".format(np.nanmean(recall_scores)) + tab + "{:.6f}".format(np.nanstd(recall_scores)) + '\n')
    scoresv_file.write('\n')
    scoresv_file.close()
    print('Finished crossval for: ' + name + '\n')

def splitvalidation(models, data, test_size=0.1, scalers=None, name=None):
    X = data[:,1:]
    t = data[:,0]
    X_train, X_test, t_train, t_test = train_test_split(X, t, stratify=t, test_size=test_size, random_state=42)
    probas_arr = np.empty((t_test.shape[0], len(models)))
    for i in range(len(models)):
        if scalers != None and scalers[i] != None:
            model = models[i](scalers[i]())
            if i==0 and name == None:
                name = model.name
        else:
            model = models[i]()
            if i==0 and name == None:
                name = model.name
        probas_arr[:,i] = model.fit(X_train,t_train).predict_proba(X_test)[:,1]
    probas = np.mean(probas_arr, axis = 1)
    np.save(os.path.join('prediction',f'{name}'), probas)
    np.savetxt(os.path.join('prediction',f'{name}'), probas, delimiter='\n')
    class_pred = np.where(probas > 0.5, 1, 0)
    roc = roc_auc_score(t_test, probas)
    f1 = f1_score(t_test, class_pred)
    prec = f1_score(t_test, class_pred)
    rec = recall_score(t_test, class_pred)
    scores_file = open(scores_file_addr, "a+")
    scores_file.write(name + " " + "{:.6f}".format(roc) + " " + datetime.datetime.now().strftime("%H:%M %d-%m") + " " + 'splitvalidation: ' + str(test_size) + '\n')
    scores_file.close()
    scoresv_file = open(scoresv_file_addr, "a+")
    scoresv_file.write(name + " " + datetime.datetime.now().strftime("%H:%M %d-%m") + '\n')
    tab = "    "
    scoresv_file.write(tab + "roc_score: " + "{:.6f}".format(roc) + '\n')
    scoresv_file.write(tab + "f1_score:  " + "{:.6f}".format(f1) + '\n')
    scoresv_file.write(tab + "pre_score: " + "{:.6f}".format(prec) + '\n')
    scoresv_file.write(tab + "rec_score: " + "{:.6f}".format(rec) + '\n')
    scoresv_file.write('\n')
    scoresv_file.close()
    print('Finished splitvalidation for: ' + name + '\n')