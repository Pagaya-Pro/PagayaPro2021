import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

class CO_predictor:

    def __init__(self):
        self._fit = False
        self._processed = False
        self._corr_lst = ['total_inquiries', 'all803', 'all804', 'all801', 'inquiries_last6_months', 'aut720', 'rep001', 'g099s', 'hr06s', 're102s', 'rev302']
        self._drop_lst = ['borrower_city', 'issue_date', 'loan_amnt']

    def _train_preprocess(self, X, y):
        res = self._general_preprocess(X)
        res['CO'] = y
        self._occupation_series = res.groupby('occupation').CO.mean()
        res.occupation = self._occupation_series[res.occupation.values].values
        res.drop('CO', axis=1, inplace=True)
        self._processed = True
        return res, y

    def _general_preprocess(self, X):
        res = X.copy()
        enc = OrdinalEncoder()
        res['ordinal_date'] = enc.fit_transform(np.array(res.issue_date).reshape(-1, 1))
        res['log_loan_amnt'] = np.log(res['loan_amnt'])
        res.occupation.fillna('unknown', inplace=True)
        if self._processed and not self._fit:
            res.occupation = self._occupation_series[res.occupation.values].values
        res.drop(self._corr_lst, axis=1, inplace=True)
        res.drop(self._drop_lst, axis=1, inplace=True)
        return res

    def fit(self, X, y):
        self._fit = True
        X_train, y_train = self._train_preprocess(X, y)
        pipe_weight = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(class_weight={0: 1, 1: 5}))])
        pipe_weight.fit(X_train, y_train)
        self.model = pipe_weight
        self._fit = False

    def predict(self, X):
        data = self._general_preprocess(X)
        return self.model.predict(data)

    def predict_proba(self, X):
        data = self._general_preprocess(X)
        return self.model.predict_proba(data)
