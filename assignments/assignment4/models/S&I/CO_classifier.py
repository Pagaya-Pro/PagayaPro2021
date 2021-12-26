import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

class CO_classifier():
    def __init__(self):
        self.occupation_encoder = None
        self.to_drop = None
        self.LR_scaled = None
        self.issue_date_enc = None
        self.model_trained = False

    def preprocess_data(self, X_orig, y= None, is_fit = True):
        X = X_orig.copy()

        ## occupation
        X['occupation'].fillna('Other', inplace = True)
        if is_fit:
            occupation_encoder = pd.DataFrame([X.occupation, y], index=['occupation', 'co']).T.groupby('occupation')
            self.occupation_encoder = occupation_encoder.sum() / occupation_encoder.count()
        X.occupation = X.occupation.apply(lambda x: self.occupation_encoder.loc[x] if x in self.occupation_encoder.index else self.occupation_encoder['Other'])
#         print(X.occupation)
        ## borrower_city
        X.drop('borrower_city', inplace=True, axis=1)

        ## issue_date
        if is_fit:
            self.issue_date_enc = OrdinalEncoder()
        X['issue_date'] = self.issue_date_enc.fit_transform(X.issue_date.to_frame())

        ## loan_amnt
        X.loan_amnt = np.log(X.loan_amnt)

        # Multicollinearity
        if is_fit:
            corr_matrix = X_train.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            self.to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        X.drop(self.to_drop, axis=1, inplace=True)
        return X


    def fit(self, X, y):
        # preprocessing
        X = self.preprocess_data(X, y, is_fit = True)

        # Model train
        self.LR_scaled = make_pipeline(StandardScaler(), LogisticRegression())
        self.LR_scaled.fit(X, y)
        self.model_trained = True


    def predict(self, X):
        if not self.model_trained:
            print("The model is not fit yet!")
            return
        # preprocessing
        res = self.preprocess_data(X, is_fit = False)
        # Model predict
        return self.LR_scaled.predict(res)


    def predict_proba(self, X):
        if not self.model_trained:
            print("The model is not fit yet!")
            return
        # preprocessing
        X = preprocess_data(self, X, is_fit=False)
        # Model predict
        return self.LR_scaled.predict_proba(X)