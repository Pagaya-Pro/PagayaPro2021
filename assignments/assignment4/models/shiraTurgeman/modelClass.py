import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pagayapro.paths.data_paths import ASSIGNMENT4_DATA
import os
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve
from scipy.stats import spearmanr
import numpy_financial as npf


class CO_clf():

    def __init__(self, T_w=5, F_w=1):
        self.T_w = T_w
        self.F_w = F_w
        self.enc = OrdinalEncoder()
        self.class_weight = {True: self.T_w, False: self.F_w}
        self.clf = Pipeline([('scaler', StandardScaler()), (
        'LogisticRegression', LogisticRegression(random_state=0, class_weight=self.class_weight))])

        pass

    def _preproccess(self, X):
        X = X.drop(['borrower_city'], axis=1)

        X[["issue_date"]] = self.enc.transform(X[["issue_date"]])

        X['occupation'] = self.occupation_map.loc[X['occupation'].values].values

        X = X.drop(
            columns=['inquiries_last6_months', 'all807', 'total_inquiries', 'g094s', 'ale720', 'monthly_pmt', 'hr09s',
                     'rep071', 'co_amount'])

        X.loan_amnt = np.log(X.loan_amnt)

        return X

    def fit(self, X, y):
        X = X.copy()
        y = y.copy()
        X.loc[X.occupation.isnull(), 'occupation'] = 'Other'
        temp = pd.DataFrame([X.occupation, y], index=['occupation', 'co']).T.groupby('occupation')
        self.occupation_map = temp.sum() / temp.count()

        self.enc.fit(X[["issue_date"]])

        X = self._preproccess(X)

        class_weight = {True: self.T_w, False: self.F_w}

        self.clf.fit(X, y)

    def predict(self, X):
        X = X.copy()
        X.loc[X.occupation.isnull(), 'occupation'] = 'Other'
        X = self._preproccess(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = X.copy()
        X.loc[X.occupation.isnull(), 'occupation'] = 'Other'
        X = self._preproccess(X)
        return self.clf.predict_proba(X)




