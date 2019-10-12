import numpy as np
import scalers
from sklearn.naive_bayes import GaussianNB

class gaussnb():
    def __init__(self, scaler=scalers.standardizer()):
        self.name = f'gaussnb scaler={scaler.name}'
        self.t1 = scaler
        self.estimator = GaussianNB()
    def fit(self,X,t):
        X = self.t1.fit(X).transform(X)
        self.estimator.fit(X, t)
        return self
    def predict_proba(self,X):
        X = self.t1.transform(X)
        return self.estimator.predict_proba(X)

class gaussnb_weighted():
    def __init__(self, scaler=scalers.standardizer()):
        self.name = 'gaussnb_std_wt'
        self.t1 = scaler
        self.estimator = GaussianNB()
    def fit(self,X,t,X_test=None,t_test=None):
        prior = np.sum(t)/t.shape[0]
        sample_weight = np.array([1, (1-prior)/(prior)])[np.array(t,dtype='int')]
        X = self.t1.fit(X).transform(X)
        self.estimator.fit(X, t, sample_weight=sample_weight)
        return self
    def predict_proba(self,X):
        X = self.t1.transform(X)
        return self.estimator.predict_proba(X)