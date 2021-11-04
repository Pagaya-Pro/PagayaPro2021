import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

class CO_classifier():
    def __init__(self):
        self.features_to_drop = ['borrower_city','co_amount','all801','all804','all803','all807','total_inquiries','aut720','ale720','iln724','iln720','in36s','monthly_pmt','bc102s','re102s','bac302','hr09s','mt47s','mt36s','rev703','rep001','g099s','hr24s','rep901']

    def fit(self, X, y):
        #drop irrelevant features
        X_c = self.drop_features(X.copy())
        #encode occupation
        self.train_mean_CO, self.occupation_encoder = self.calc_occupation_encoder(X, y)
        self.encode_occupation(X_c)
        #Save first date and convert to days
        self.first_date = X_c['issue_date'].min()
        self.parse_issue_date(X_c)
        #change loan amount to log loan amount
        self.loan_amnt_to_log(X_c)

        self.lr = Pipeline((('Scaler', StandardScaler()),
                            ('LR', LogisticRegression())))
        #Give CO loans more weight
        weights = np.ones(len(X_c))
        weights[y == 1] = 5
        self.lr.fit(X_c, y, LR__sample_weight=weights)


    def predict(self, X):
        # drop irrelevant features
        X_c = self.drop_features(X.copy())
        # encode occupation
        self.encode_occupation(X_c)
        # Convert to days from first train date
        self.parse_issue_date(X_c)
        # change loan amount to log loan amount
        self.loan_amnt_to_log(X_c)
        return self.lr.predict(X_c)

    def predict_proba(self, X):
        # drop irrelevant features
        X_c = self.drop_features(X.copy())
        # encode occupation
        self.encode_occupation(X_c)
        # Convert to days from first train date
        self.parse_issue_date(X_c)
        # change loan amount to log loan amount
        self.loan_amnt_to_log(X_c)

        return self.lr.predict_proba(X_c)

    def drop_features(self,X):
        #drop all the features we saw was problematic - high variance or high multicollinearity
        X_c = X.copy()
        return X_c.drop(columns=self.features_to_drop)

    def calc_occupation_encoder(self, X, y):
    #Create an encoder with the predicted CO rates per occupation
        X_with_CO = X.copy()
        X_with_CO['CO'] = y
        train_mean_CO = y.mean()
        encoder = X_with_CO.groupby('occupation', dropna=False)['CO'].mean()
        return train_mean_CO, encoder

    def occupation_decoder(self,occupation):
        if occupation is None:
            return self.occupation_encoder[np.nan]
        if occupation in self.occupation_encoder.index:
            return self.occupation_encoder[occupation]
        return self.train_mean_CO

    def encode_occupation(self, X):
    #Replaces occupations in occupation_series according to occupation_encoder
        X['occupation'] = X['occupation'].apply(self.occupation_decoder)

    def parse_issue_date(self, X):
        X['issue_date'] = (X['issue_date'] - self.first_date).dt.days

    def loan_amnt_to_log(self,X):
        #change loan amount feature to log loan amount
        X['log_loan_amount'] = np.log(X['loan_amnt'])
        X = X.drop(columns='loan_amnt')