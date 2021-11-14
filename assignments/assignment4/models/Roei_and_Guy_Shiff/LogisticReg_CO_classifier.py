import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

warnings.filterwarnings('ignore')


class co_classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = Pipeline([("standard_scaler", StandardScaler()), ("logistic_regression", LogisticRegression())])
        self.occupation_series = None

    def preprocess(self, X, y=None):
        train_prep = np.any(y != None)
        correlated_cols_to_drop = ['all803', 'all804', 'all801', 'hr12s', 'g094s', 'aut720', 'monthly_pmt',
                                   're102s', 'rev302', 'hr09s', 'rep001', 'rep901', 's004s', 'iln720', 'ale724',
                                   'all807', 'bac302', 'mt47s', 'rep071',
                                   'iln724', 'inquiries_last6_months', 'in36s', 'g960s', 'borrower_city', 'loan_amnt',
                                   'issue_date']
        X_data = X.copy()
        if 'co_amount' in X_data.columns:
            X_data.drop(columns='co_amount', inplace=True)
        X_data['log_loan_amnt'] = np.log(X_data['loan_amnt'])

        le = LabelEncoder()
        X_data['issue_date_encode'] = le.fit_transform(X_data.issue_date)
        X_data.drop(columns=correlated_cols_to_drop, inplace=True)

        X_data['occupation'].fillna(value='Other', inplace=True)

        if train_prep:
            y_data = y.copy()
            y_data.columns = ['co']
            self.occupation_series = pd.concat([X_data, y_data], axis=1).groupby('occupation').co.mean()

        X_data.occupation = self.occupation_series[X_data.occupation.values].values

        return X_data

    def fit(self, X, y):
        X = self.preprocess(X, y)
        self.model.fit(X, y)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.model.predict_proba(X)

    def predict(self, X):
        pred_prob = self.predict_proba(X)
        return pred_prob[:, -1] > self.threshold
