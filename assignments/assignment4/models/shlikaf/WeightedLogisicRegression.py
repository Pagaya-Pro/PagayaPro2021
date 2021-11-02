import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from pagayapro.paths.data_paths import ASSIGNMENT4_DATA
import os
import numpy_financial as npf
from sklearn import preprocessing

class WeightedLogisticregression():
    CASHFLOWS = pd.read_parquet(os.path.join(ASSIGNMENT4_DATA, "prosper_cashflows.parquet"))
    THRESHOLD = 0.45
    WEIGHTS = {1:5, 0:1}
    CORR_THR = 0.4

    def __init__(self):
        self.X = None
        self.y = None
        self.weighted_pipe = None

    def fit(self, X, y):
        self.X = X, self.y = self.__apply_transformation_to_data(X, y)
        self.weighted_pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight=WeightedLogisticregression.WEIGHTS))

    def predict(self, X):
        pipe_proba = self.weighted_pipe.predict_proba(X)
        return (pipe_proba <= WeightedLogisticregression.THRESHOLD).astype('int')

    def predict_proba(self, X):
        return self.weighted_pipe.predict_proba(X)

    def __apply_transformation_to_data(self, X, y=None):
        if y != None:
            trans_y = (~y.isna()).astype('int')  # 1 is CO, 0 no CO

        processed_X = self.__preprocess_X(X)
        filtered_X = self.__filter_features(processed_X)

        return filtered_X, trans_y

    def __preprocess_X(self, X):
        processed_X = X.copy()

        processed_X.drop('co_mob', axis=1, inplace=True)
        processed_X.drop('borrower_city', axis=1, inplace=True)

        le = preprocessing.LabelEncoder()
        processed_X['issue_date'] = le.fit_transform(processed_X.issue_date)

        le = preprocessing.LabelEncoder()
        processed_X['occupation'] = le.fit_transform(processed_X.occupation)

        processed_X['loan_amnt'] = np.log(processed_X['loan_amnt'])

        return processed_X

    def __filter_features(self, X):
        filtered_X = X.copy()

        for drop in range(X.shape[1]):
            mat = np.abs(filtered_X.corr().values)
            second_largest_idx = mat.argsort()[:, -2]
            second_largest_corr = []

            for i in range(mat.shape[0]):
                second_largest_corr.append(mat[i, second_largest_idx[i]])
            second_largest_corr = np.array(second_largest_corr)

            if second_largest_corr.max()< WeightedLogisticregression.CORR_THR:
                break

            column_to_drops = filtered_X.columns[np.argpartition(second_largest_corr, -2)[-2:]]
            column1_corr = np.abs(filtered_X[column_to_drops[0]].corr(self.y))
            column2_corr = np.abs(filtered_X[column_to_drops[1]].corr(self.y))

            if column1_corr > column2_corr:
                column_to_drop = column_to_drops[1]
            else:
                column_to_drop = column_to_drops[0]

            filtered_X.drop(column_to_drop, axis=1, inplace=True)

        return filtered_X


    @staticmethod
    def get_portfolio_irr(indices):
        """
        cashflows- a dataframe of cashflows
        indices- pd.Series of T/F or a list of indices

        output: the yearly irr of the dataframe cashflows.loc[indices]
        """
        return (((npf.irr(WeightedLogisticregression.CASHFLOWS.loc[indices].sum()) + 1) ** 12) - 1) * 100
