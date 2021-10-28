from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class PreProcessor_NM(TransformerMixin):

    def __init__(self, co_linearity_threshold=0.35, features_to_remove=None):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.co_linearity_threshold = co_linearity_threshold
        self.features_to_remove = ['monthly_pmt']
        self.features_not_to_remove = ['loan_amnt']
        self.target_encode_dict = {}
        self.X_train = None
        self.y_train = None


    def fit_transform(self,X,y=None):

        self.X_train = X
        self.y_train = y

        # Drop un interesting column
        self.X_train.drop(columns='borrower_city',inplace=True)
        # Encode the dates using label encoder
        self.X_train['issue_date'] = LabelEncoder().fit_transform(self.X_train.issue_date)
        # Target encode the occupation column
        if type(y) == None:
            #If there is no y we cant target encode
            self.X_train.drop(columns=['occupation'], inplace=True)
        else:
            for occupation in self.X_train['occupation'].unique():
                self.target_encode_dict[occupation] = self.y_train[self.X_train['occupation'] == occupation].mean()

            self.target_encode_dict[None] = self.y_train[self.X_train['occupation'].isna()].mean()

            self.X_train['occupation'] = self.X_train['occupation'].apply(self.target_encode_occupation)

        # Remove data leakage)
        self.X_train.drop(columns=['co_amount'], inplace=True)

        # Remove correlated features
        self.get_correlated_features_to_remove(X)
        self.X_train.drop(columns=self.features_to_remove, inplace=True)

        #Convert laon amount to it's log value
        self.X_train['loan_amnt'] = np.log(self.X_train['loan_amnt'])

        return self.X_train


    def transform(self,X_test):

        # Drop un interesting column
        X_test.drop(columns=['borrower_city','co_amount'], inplace=True)

        # Encode the dates using label encoder
        X_test['issue_date'] = LabelEncoder().fit_transform(X_test.issue_date)
        X_test['occupation'] = X_test['occupation'].apply(self.target_encode_occupation)
        X_test['loan_amnt'] = np.log(X_test['loan_amnt'])
        X_test.drop(columns=self.features_to_remove, inplace=True)
        return X_test

    def get_correlated_features_to_remove(self,X):

        correlated_features = self.get_correlated_list(X)

        from collections import Counter
        features_to_remove = ['monthly_pmt']
        features_not_to_remove = ['loan_amnt']

        while (len(correlated_features) > 0):
            tupple_to_list = [a[0] for a in correlated_features] + [a[1] for a in correlated_features]
            counts = Counter(tupple_to_list)
            sorted_list = sorted(counts, key=counts.get, reverse=True)
            removed_feature = sorted_list[0]
            self.features_to_remove.append(removed_feature)
            correlated_features = list(
                filter(lambda features: (features[0] != removed_feature) and (features[1] != removed_feature),
                       correlated_features))

        self.features_to_remove = list(set(self.features_to_remove) - set(self.features_not_to_remove))


    def get_correlated_list(self,X):

        max_index = 0
        corr_matrix = X.corr().abs()

        # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
        correlation_sorted = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                              .stack()
                              .sort_values(ascending=False))

        for i in range(len(correlation_sorted)):
            if (correlation_sorted[i] <  self.co_linearity_threshold ):
                max_index = i
                break

        return set([tupple for tupple in correlation_sorted.index[0:i]])

    def target_encode_occupation(self,occupation):
        return self.target_encode_dict[occupation]



