import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")


"""
This class preprocesses data before generating the logistic regression model.
It is called from LogitClass.
"""
class PreprocessClass():
    def __init__(self):
        pass

    def solve_multicolin(self, X):
        """
        Solves multicolinearity in a given matrix based on correlation and VIF.
        :param X: a numeric dataframe
        :return: X without columns with high linear dependency
        """
        X = X.copy()
        cor_mat = X.corr()
        # Drop correlated features
        for i, j in enumerate(range(len(cor_mat))):
            if i == j:
                cor_mat.iloc[i, j] = np.nan
        while True:
            max_correlated = cor_mat[cor_mat > 0.75].count().idxmax()
            if cor_mat[cor_mat > 0.75].count()[max_correlated] >= 2:
                cor_mat = cor_mat.drop(max_correlated, axis=0).drop(max_correlated, axis=1)
            else:
                break
        # Drop
        X = X[cor_mat.columns]
        while True:
            vif_series = pd.Series([variance_inflation_factor(X.values, i)
                                    for i in range(X.shape[1])],
                                   index=X.columns)
            max_VIF = vif_series.idxmax()
            if vif_series.loc[max_VIF] >= 30:
                X.drop(max_VIF, axis=1, inplace=True)
            else:
                break
        return X

    def preprocess_train(self, X, y):
        """
        Fill NaNs, adjust variables types, create new features, reduce multicolinearity
        :param X: The training feature dataframe (features as given in assignment 4).
        :param y: The target training series (co_mob).
        :return:  X, y after being preprocessed.
        """
        X, y = X.copy(), y.copy()
        y.fillna(0, inplace=True)

        X.drop('borrower_city', axis=1, inplace=True)
        X.drop('co_amount', axis=1, inplace=True)
        # Switch loan_amnt with log_loan_amnt
        X['log_loan_amnt'] = np.log(X.loan_amnt)
        X.drop('loan_amnt', axis=1, inplace=True)

        X.issue_date = LabelEncoder().fit_transform(X.issue_date)
        # None to string
        X.occupation.loc[X.occupation.isna()] = 'None'
        concat_X_y = pd.concat([X, y], axis=1)

        # Target to binary
        y = y.astype(bool).astype(int)

        # Mean of features by occupation
        try:
            self.occ_feature_encoded = concat_X_y.groupby('occupation').mean().co_mob
        except:
            self.occ_feature_encoded = concat_X_y.groupby('occupation').mean().iloc[:, -1]

        # Add occupation encoded and drop occupation
        X['occupation_encoded'] = X.apply(lambda x: self.occ_feature_encoded.loc[x['occupation']],
                                          axis=1)
        X.drop('occupation', axis=1, inplace=True)

        # Drop features causing multi-colinearity
        X = self.solve_multicolin(X)
        self.columns = X.columns

        return X, y

    def preprocess_test(self, X):
        """
        Adjust variables types, create new features, reduce multicolinearity
        :param X: a X test dataframe, unprocessed.
        :return: a processed X dataframe.
        """
        x = X.copy()
        x['issue_date'] = LabelEncoder().fit_transform(x['issue_date'])
        # None to string
        x.occupation.fillna('None', inplace=True)

        x['log_loan_amnt'] = np.log(x.loan_amnt)
        x['occupation_encoded'] = x.apply(lambda y: self.occ_feature_encoded.loc[y['occupation']],
                                          axis=1)
        x = x[self.columns]
        return x



"""
This class serves as a prediction model 
"""
class LogitClass():
    def __init__(self):
        """
        Create an instance of preprocess class and standard scaler
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import LogisticRegression

        self.preprocess = PreprocessClass()
        self.log_pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression())


    def fit(self, X, y):
        self.X_train, self.y_train = self.preprocess.preprocess_train(X, y)
        self.log_pipe.fit(self.X_train, self.y_train)


    def predict(self, X):
        return self.log_pipe.predict(self.preprocess.preprocess_test(X))

    def predict_proba(self, X):
        return self.log_pipe.predict_proba(self.preprocess.preprocess_test(X))
