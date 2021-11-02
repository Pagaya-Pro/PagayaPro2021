import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')


class LogisticChargeOff(BaseEstimator, ClassifierMixin):

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.df_occ = None
        self.model = Pipeline([("standard_scaler", StandardScaler()), ("logistic_regression", LogisticRegression())])
        self.curr_columns = []

    def preprocess(self, X, y=None):
        X = X.copy()
        X.drop('borrower_city', axis=1, inplace=True)
        if 'co_amount' in X.columns:
            X.drop('co_amount', axis=1, inplace=True)

        # issue_date
        le = preprocessing.LabelEncoder()
        le.fit(X['issue_date'])
        X['issue_date_encode'] = le.transform(X['issue_date'])
        X.drop('issue_date', axis=1, inplace=True)

        # occupation
        X['occupation'].fillna('Unknown', inplace=True)
        X.loc[X['occupation'].str.contains('Student') == True, 'occupation'] = 'Student'
        X.loc[X['occupation'].str.contains('Nurse') == True, 'occupation'] = 'Nurse'
        if self.df_occ is None:
            self.df_occ = pd.concat([X, y], axis=1).groupby('occupation')['co_mob'].mean().to_frame()
            self.df_occ.columns = ["occupation_mean"]
        X = X.join(self.df_occ, on='occupation', how='left')
        X.drop('occupation', axis=1, inplace=True)

        # corr
        self.remove_corr_columns(X, y)

        # outliers
        X['log_loan_amnt'] = np.log(X['loan_amnt'])
        X.drop('loan_amnt', axis=1, inplace=True)
        return X

    def remove_corr_columns(self,X,y):

        if y is None:
            X.drop(self.curr_columns, axis=1, inplace=True)
            return 0

        have_corr = True
        while have_corr:
            x_corr = X.corr()
            corr_df = x_corr[(x_corr > 0.75) & (x_corr < 1)]
            corr_sum = corr_df.notna().sum(axis=1).to_frame().reset_index()
            max_corr = corr_sum[(corr_sum['index'] != 'credit_score') & (corr_sum['index'] != 'int_rate')].sort_values(by=0,ascending=False).iloc[0]
            if max_corr[0] > 0:
                self.curr_columns.append(max_corr['index'])
                X.drop(max_corr['index'], axis=1, inplace=True)
            else:
                have_corr = False

    def fit(self, X, y):
        X = self.preprocess(X, y)
        self.model.fit(X, y)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.model.predict_proba(X)

    def predict(self, X):
        pred_prob = self.predict_proba(X)
        return pred_prob[:,0] < self.threshold





