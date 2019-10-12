import numpy as np
from bisect import bisect
from scipy.special import erfinv

class standardizer():
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.name = 'standardizer'
    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    def transform(self, X):
        X = (X - self.mean)/self.std
        return X

class rank_gauss():
    def __init__(self):
        self.N = 0
        self.ord = np.empty((0,0))
        self.name = 'rankgauss'
    def fit(self,X):
        self.ord = np.empty(X.shape)
        self.N = X.shape[0]+2
        for i in range(X.shape[1]):
            self.ord[:,i] = np.sort(X[:,i])
        return self
    def transform(self, X):
        Y = np.empty(X.shape)
        for c in range(X.shape[1]):
            index = np.searchsorted(self.ord[:,c], X[:,c])
            Y[:,c] = np.sqrt(2)*erfinv(2*(((index+1)/self.N) - 0.5))
        return Y
