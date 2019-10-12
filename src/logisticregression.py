import numpy as np
import scalers
from sklearn.linear_model import LogisticRegression

C = 1.0
class logisticregression():
    def __init__(self, scaler = scalers.standardizer()):
        self.name = f'logisticregression log10(C)={np.log10(C):.2f} scaler={scaler.name}'
        self.t1 = scaler
        self.estimator = LogisticRegression(solver='liblinear', C=C)
    def fit(self,X,t):
        X = self.t1.fit(X).transform(X)
        self.estimator.fit(X, t)
        return self
    def predict_proba(self,X):
        X = self.t1.transform(X)
        return self.estimator.predict_proba(X)
    def get_coefs(self):
        return [self.t1.mean, self.t1.standardizer, self.estimator._coef]