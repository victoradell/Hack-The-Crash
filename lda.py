import numpy as np
import scalers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class lda():
    def __init__(self, scaler=scalers.standardizer()):
        self.name = f'lda scaler={scaler.name}' 
        self.t1 = scaler
        self.estimator = LinearDiscriminantAnalysis()
    def fit(self,X,t):
        X = self.t1.fit(X).transform(X)
        self.estimator.fit(X, t)
        return self
    def predict_proba(self,X):
        X = self.t1.transform(X)
        return self.estimator.predict_proba(X)

    