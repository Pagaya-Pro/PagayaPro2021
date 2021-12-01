import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


class PipeLogisticRegression():

    def __init__(self, th=0.5):
        """
        This class preprocesses, fits and predicts charge-offs on loans borrowers asked Pagaya to take.
        This class uses StandardScaler to normalize and LogisticRegression for prediction (sklearn).
        :param th: threshold - sets the probability for nonCO from which loans will be given.
        """
        self.le = preprocessing.LabelEncoder()
        self.th = th
        self.lr = Pipeline(
            [('scaler', StandardScaler()), ('lr', LogisticRegression(C=0.1, penalty='l1', solver='liblinear'))])

    def fit(self, X, y):
        """
        This function calls to the preprocess and trains the model.
        :param X: data on loans
        :param y: target variable
        """
        y = y.fillna(0)
        y[y != 0] = 1
        X = pd.concat([X, y], axis=1)
        self.le.fit(X.issue_date)

        self.set_mean_of_co_mob_for_each_occupation(X)
        X, y = self.preprocess(X, y)
        X.drop(columns=y.name, inplace=True)
        self.columns_to_drop = self.remove_corr_columns(X)
        self.lr.fit(X, y)

    def predict(self, X_test):
        """

        :param X_test: data of loans to predict.
        :return: Series of predictions for each loan.(0/1)
        """
        X_test = self.preprocess(X_test)[0]
        X_test.drop(columns=self.columns_to_drop, inplace=True)
        return (self.predict_proba(X_test)[:, 1] > self.th).astype('int')

    def predict_proba(self, X):
        """

        :param X: data of loans to predict
        :return: Series of the probability to charge off for each loan.
        """
        return self.lr.predict_proba(X)

    def set_mean_of_co_mob_for_each_occupation(self, X):
        """
        This function defines for each occupation the percentage of charge off and put the values in self.mean.
        :param X: The data of loans.
        """
        X.occupation.fillna('None', inplace=True)
        X.occupation = X.occupation.str.lower()

        X.loc[X.occupation.str.contains('student'), 'occupation'] = 'student'
        X.loc[X.occupation.str.contains('profession that is not part of this list'), 'occupation'] = 'other'
        X.loc[X.occupation.str.contains('nurse - registered nurse \(rn\)'), 'occupation'] = 'nurse (rn)'
        X.loc[X.occupation.str.contains(
            'secretary/administrative assistant'), 'occupation'] = 'administrative assistant'
        X.loc[X.occupation.str.contains('judge'), 'occupation'] = 'professional'

        self.means = X.groupby('occupation').mean().iloc[:, -1]

    def preprocess(self, X, y=None):
        """
        This class cleans, drops, and prepares data for fit/prediction.
        :param X: The data
        :param y: target variable
        :return: X,y after the preprocessing.
        """
        X.occupation.fillna('None', inplace=True)
        X.occupation = X.occupation.str.lower()

        X.loc[X.occupation.str.contains('student'), 'occupation'] = 'student'
        X.loc[X.occupation.str.contains('profession that is not part of this list'), 'occupation'] = 'other'
        X.loc[X.occupation.str.contains('nurse - registered nurse \(rn\)'), 'occupation'] = 'nurse (rn)'
        X.loc[X.occupation.str.contains(
            'secretary/administrative assistant'), 'occupation'] = 'administrative assistant'
        X.loc[X.occupation.str.contains('judge'), 'occupation'] = 'professional'
        X['occupation'] = X.apply(lambda x: self.means.loc[x['occupation']], axis=1)
        X['loan_amnt'] = np.log(X.loan_amnt)
        X.drop(columns=['borrower_city'], inplace=True)
        if not isinstance(y, type(None)):
            y = y.fillna(0)

        X.drop(columns=['co_amount'], inplace=True)
        X['issue_date'] = self.le.transform(X.issue_date)

        return X, y

    def remove_corr_columns(self, X):
        """
        This function finds the correlation between the rows and puts rows with high correlation with
        another row (which is not in the list) in the list.
        :param X: The data.
        :return: which columns we need to drop.
        """
        columns_to_drop = []
        while (X.corr().abs() > 0.80).sum().sort_values(ascending=False).values[0] > 1:
            col_to_drop = \
                (X.corr().abs() > 0.80).sum().sort_values(ascending=False).index[0]
            columns_to_drop.append(col_to_drop)
            X.drop(columns=col_to_drop, inplace=True)

        return columns_to_drop
