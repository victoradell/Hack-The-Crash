import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score,f1_score,precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import datetime
import pylab
from collections import defaultdict
import os

import scalers


scores_file_addr = 'scores.txt'
scoresv_file_addr = 'scoresv.txt'

# Train test split validation function
def splitvalidation(models, data, test_size=0.2, scalers=None, name=None):
    X = data[:,:-1]
    t = data[:,-1]
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size, random_state=42)
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
        winners = np.load('test.npy')
        wprobas = model.predict_proba(winners)[:,-1]
        wclasses = np.where(wprobas > 0.5, 1, 0)
        np.savetxt('submission0.5.csv', wclasses, delimiter=',')
    probas = np.mean(probas_arr, axis = 1)
    np.save(os.path.join('prediction',f'{name}'), probas)
    np.savetxt(os.path.join('prediction',f'{name}'), probas, delimiter='\n')
    print(np.mean(probas[t_test==0]), np.mean(probas[t_test==1]), (np.mean(probas[t_test==1]) + np.mean(probas[t_test==0]))/2)
    thresh = 0.25
    class_pred = np.where(probas > thresh, 1, 0)
    roc = roc_auc_score(t_test, probas)
    f1 = f1_score(t_test, class_pred)
    prec = precision_score(t_test, class_pred)
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