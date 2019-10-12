#%%
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

#%% Data import

train = "./data/train.csv"
test = "./data/test.csv"
db = pd.read_csv(train)
db_test = pd.read_csv(test)

train = db.iloc[:, 1:202].to_numpy(dtype = 'float64') ## Training data matrix
np.save(os.path.join('data','train'), train)

test = db_test.iloc[:, 1:201].to_numpy(dtype = 'float64') ## Test data matrix
np.save(os.path.join('data','test'), train[:,1:])

#%% Previsualization
for i in [66,80,193,78]:
    measurements = train[:, i]
    mean = np.mean(measurements)
    sd = np.std(measurements)
    pylab.suptitle(f"var{i:d} mean={mean:.2f} sd={sd:.2f}")
    pylab.subplot(1,2,1)
    pylab.hist(measurements, density=True, bins='auto')
    pylab.title(f"Histogram var{i:d}")
    pylab.subplot(1,2,2)
    stats.probplot(measurements, dist="norm", plot=pylab)
    pylab.title(f"Normal qqplot var{i:d}")
    pylab.savefig(f"previsual/var{i:d}.png", dpi=80)
    pylab.clf()

#%% By class histogram and qq-plot
for i in range(201):
    measurements0 = np.squeeze(train[np.where(train[:,0]==1), i])
    measurements1 = np.squeeze(train[np.where(train[:,0]==0), i])
    mean0 = np.mean(measurements0)
    sd0 = np.std(measurements0)
    mean1 = np.mean(measurements1)
    sd1 = np.std(measurements1)
    pylab.suptitle(f"var{i:d} mean0={mean0:.2f} sd0={sd0:.2f} mean1={mean1:.2f} sd1={sd1:.2f}")
    pylab.subplot(2,2,1)
    pylab.hist(measurements0, density=True, bins='auto')
    pylab.title(f"Histogram var{i:d} target=0")
    pylab.subplot(2,2,2)
    stats.probplot(measurements0, dist="norm", plot=pylab)
    pylab.title(f"Normal qqplot var{i:d} target=0")
    pylab.subplot(2,2,3)
    pylab.hist(measurements1, density=True, bins='auto')
    pylab.title(f"Histogram var{i:d} target=1")
    pylab.subplot(2,2,4)
    stats.probplot(measurements1, dist="norm", plot=pylab)
    pylab.title(f"Normal qqplot var{i:d} target=1")
    pylab.tight_layout(pad=2,h_pad=0.5)
    pylab.savefig(f"previsual/var{i:d}_byclass.png", dpi=80)
    pylab.clf()

#%% Linear separability
x = np.zeros(200)
for i in range(1,201):
    measurements0 = np.squeeze(train[np.where(train[:,0]==1), i])
    measurements1 = np.squeeze(train[np.where(train[:,0]==0), i])
    mean0 = np.mean(measurements0)
    mean1 = np.mean(measurements1)
    sd0 = np.std(measurements0)
    sd1 = np.std(measurements1)
    r = (measurements0.shape[0]-1)/(measurements0.shape[0]+measurements1.shape[0]-2)
    sd = r*sd0 + (1-r)*sd1
    x[i-1] = abs(mean0-mean1)/sd
pylab.title(f"Histogram of relative mean separation")
pylab.hist(x, bins='auto')
pylab.savefig(f"previsual/linsep_histogram.png", dpi=80)
pylab.show()
print(np.max(x), np.where(x==np.max(x)))
print(np.median(x))
print(np.min(x), np.where(x==np.min(x)))

#%% PCA and screeplot of train
X = np.delete(train,0,axis=1)
X = preprocessing.StandardScaler().fit(X).transform(X)
pca = sklearnPCA().fit(X)
pylab.suptitle(f"PCA scree plot train")
pylab.plot(np.cumsum(pca.explained_variance_ratio_))
pylab.xlabel('number of components')
pylab.ylabel('cumulative explained variance')
pylab.savefig(f"featurES/screeplot.png", dpi=80)
pylab.show()

print(np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)))
print(np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.8)))

#%% PCA and screeplot of train+test
X = np.concatenate((np.delete(train,0,axis=1),test),axis=0)
X = preprocessing.StandardScaler().fit(X).transform(X)
pca = sklearnPCA().fit(X)
pylab.suptitle(f"PCA scree plot train+test")
pylab.plot(np.cumsum(pca.explained_variance_ratio_))
pylab.xlabel('number of components')
pylab.ylabel('cumulative explained variance')
pylab.savefig(f"featurES/screeplot_both.png", dpi=80)
pylab.show()

print(np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)))
print(np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.8)))

#%% VIF procedure attempt
X = np.delete(train,0,axis=1)
sc = np.zeros(X.shape[1])
for i in range(X.shape[1]):
    T = np.delete(X,i,axis=1)
    y = X[:,i]
    sc[i] = LinearRegression().fit(T,y).score(T,y)
    sc[i] = 1/(1-sc[i])
    if (i % 20 == 0):
        print(i)
pylab.suptitle(f"VIF boxplot mean={np.mean(sc):.5f} max={np.max(sc):.5f} min={np.min(sc):.5f}")
pylab.boxplot(sc, labels=['Training set'], showmeans=True)
pylab.ylabel('VIF')
pylab.savefig(f"featurES/vif_boxplot.png", dpi=80)
print('Max VIF:', np.max(sc))
print('Mean VIF:', np.mean(sc))
print('Min VIF:', np.min(sc))