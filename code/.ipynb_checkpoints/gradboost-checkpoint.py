import numpy as np
import scalers
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE

n_estimators=500
max_depth=None

# Gradient boosting wrapper class
class gradboost():
    def __init__(self, scaler=scalers.standardizer()):
        if max_depth != None:
            self.name = f'gradboost n_est={n_estimators:d} max_depth={max_depth:d} scaler={scaler.name}'
        else:
            self.name = f'gradboost n_est={n_estimators:d} scaler={scaler.name}'
        self.t1 = scaler
        self.estimator = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    def fit(self,X,t):
        # X, t = SMOTE().fit_sample(X, t) tested upsampling in train, didn't improve results
        X = self.t1.fit(X).transform(X)
        self.estimator.fit(X, t)
        return self
    def predict_proba(self,X):
        X = self.t1.transform(X)
        return self.estimator.predict_proba(X)