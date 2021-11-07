import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class Model:
    def __init__(self):
        """
        Defining class variables
        """
        self.le = LabelEncoder()
        self.model = Pipeline((('StandardScaler',StandardScaler()),('LogisticRegression', LogisticRegression())))

    def find_correlated_features(self, X: pd.DataFrame, threshold=0.75):
        """
        Finds the correlated features that we want to remove.
        :param X: Train df
        :param threshold: threshold of correlation
        """
        self.dropped_features = []
        X = X.copy()
        while True:
            corr_mat = X.corr()
            high_corr_num = (np.abs(corr_mat) > threshold).sum()
            high_corr_num['loan_amnt'] = 0

            if high_corr_num.max() > 1:
                feature_to_remove = high_corr_num.idxmax()
                self.dropped_features.append(feature_to_remove)
                X = X.drop(feature_to_remove, axis=1)
            else:
                break

    def handle_categorical(self, X: pd.DataFrame):
        """
        Preprocessing X matrix according to the logic we presented in the notebook.
        :param X: DataFrame
        :return: X preprocessed
        """
        X = X.drop(['co_mob', 'co_amount', 'borrower_city'], axis=1, errors='ignore')

        if 'issue_date' in X.columns:
            X['issue_date'] = self.le.transform(X['issue_date'])

        if 'occupation' in X.columns:
            X['occupation'] = X['occupation'].apply(lambda x: self.occupation_co_mean[x] if x in self.occupation_co_mean else self.occupation_co_mean[np.nan])

        return X

    def fit(self, X, y):
        """
        Fits the model according to train set
        :param X: Training set
        :param y: labels
        """
        X = X.copy()
        # build date transformer
        self.le.fit(X['issue_date'])

        # build occupation dict
        X['label'] = y
        self.occupation_co_mean = X.groupby('occupation',dropna=False)['label'].mean()
        X = X.drop('label', axis=1)

        # handle categorical features
        X = self.handle_categorical(X)

        # build correlated dictionary & dropping them
        self.find_correlated_features(X)
        X = X.drop(self.dropped_features, axis=1, errors='ignore')

        # log loan amount
        X.loan_amnt = np.log(X.loan_amnt)

        self.model.fit(X, y)

    def prepare_x_for_predict(self, X):
        """
        Function that preprocess new sample set (to be as the training)
        :param X: Samples set
        :return: Sample set in the right format
        """
        # handle categorical features
        X = self.handle_categorical(X)

        # build correlated dictionary & dropping them
        X = X.drop(self.dropped_features, axis=1, errors='ignore')

        # log loan amount
        X.loan_amnt = np.log(X.loan_amnt)
        return X

    def predict(self, X):
        """
        Makes prediction on X
        :param X: samples set
        :return: labels prediction
        """
        return self.model.predict(self.prepare_x_for_predict(X))

    def predict_proba(self, X):
        """
        Makes probability prediction on X
        :param X: samples set
        :return: probability to be on each label
        """
        return self.model.predict_proba(self.prepare_x_for_predict(X))
