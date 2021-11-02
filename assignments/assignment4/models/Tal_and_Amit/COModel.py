import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, ClassifierMixin


class COModel(BaseEstimator, ClassifierMixin):

    def __init__(self, threshold=0.5):
        self.pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
        self.threshold = 1 - threshold

    def fit(self, X, y):
        self.pipe.fit(self.data_preparation(X, True, y), y)

    def predict(self, X):
        return self.predict_proba(X)[:, 1] > self.threshold

    def predict_proba(self, X):
        return self.pipe.predict_proba(self.data_preparation(X))

    def data_preparation(self, X, is_train=False, y=None):
        if 'co_mob' in X:
            X.drop('co_mob', axis=1, inplace=True)

        if 'borrower_city' in X:
            X.drop('borrower_city', axis=1, inplace=True)

        if 'issue_date' in X:
            enc = OrdinalEncoder()
            X['ordinal_date'] = enc.fit_transform(X.issue_date.values.reshape(-1, 1))
            X.drop('issue_date', axis=1, inplace=True)

        X.loc[:, 'occupation'] = X.occupation.fillna('unknown')

        if is_train:
            X['CO'] = y
            self.occ_series = X.groupby('occupation').CO.mean()
            X.drop('CO', axis=1, inplace=True)

        if X.occupation.dtype != np.float64:
            X.loc[:, 'occupation'] = self.occ_series[X['occupation'].values].values

        columns = ['total_inquiries', 'all803', 'all804', 'all801', 'inquiries_last6_months', 'aut720', 'rep001',
                   'g099s',
                   'hr06s', 're102s', 'rev302']
        if 'total_inquiries' in X:
            X.drop(columns, axis=1, inplace=True)

        if 'loan_amnt' in X:
            X['log_loan_amnt'] = np.log(X['loan_amnt'])
            X.drop('loan_amnt', axis=1, inplace=True)

        if 'co_amount' in X:
            X.drop('co_amount', axis=1, inplace=True)

        return X