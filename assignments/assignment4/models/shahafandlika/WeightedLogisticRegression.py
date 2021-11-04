import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy_financial as npf
from sklearn import preprocessing

class WeightedLogisticregression():

    def __init__(self, threshold=0.45, weights={1:5, 0:1}, corr_thr=0.4):
        """
        Initiates an instance of WeightedLogisticregression - model that maximizes IRR per given portfolio.
        The model receives loan transaction data X, and labels y that indicate whether a loan will CO, and gives predictions such that the IRR of X is optimal

        :param threshold - the threshold that decides if the label of an instance is 0/1 -
               if the probability is smaller than threshold - the label will be 1
        :param weights - the ratio between CO/non-CO. default: 5/1
        :param corr_thr - this is a hyper parameter for reducing colinearity.
               If two columns have correlation smaller than this value, they are not concidered correlated. default" 0.4
        """
        self.y = None  # only used during the fit, i can move it into the fit
        self.weighted_pipe = None
        self.occupation_map = None
        self.good_columns = None
        self.le = None

    def fit(self, X, y):
        """
        Fits the model to labeled data. The model processes the data and filters columns to deal with multicolinearity.
        :param X: pandas.DataFrame object used for fitting, current version (v0.00001)
        :param y: a vector of [0,1] or [value,Nan] of the same length as X to be used as labels
        :return: a fitted model
        """
        preprocessed_X = self.__apply_transformation_to_data(X, y)
        self.weighted_pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight=self.weights))
        self.weighted_pipe.fit(preprocessed_X, self.y)
        return self

    def predict(self, X):
        """
        Must be called after fit. Predicts the labels of X by the model. The model processes the data and filters columns to deal with multicolinearity.
        Only numeric columns are used to make the predictions in the model
        :param X: pandas.DataFrame object used for fitting, current version (v0.00001)
        :return: predictions of the model (0,1) to every instance in X
        """
        preprocessed_X = self.__apply_transformation_to_data(X, None)
        pipe_proba = self.weighted_pipe.predict_proba(preprocessed_X)[:1]
        return (pipe_proba <= self.threshold).astype('int')

    def predict_proba(self, X):
        """
        predicts the probability of CO and non CO according to a fitted model
        :param X: Data about loans
        :return: probabilities that the loans will CO and not CO
        """
        preprocessed_X = self.__apply_transformation_to_data(X)
        return self.weighted_pipe.predict_proba(preprocessed_X)

    def __apply_transformation_to_data(self, X, y=None):
        """
        Internal function for initiating the model
        Transforms the labels into 0/1 (if necessary)
        Processes X: uses only numeric columns and filters columns to reduce multicolinearity
        :param X: the data
        :param y: labels
        :return: preprocessed X
        """
        if y is not None:
            if y.isna().any() == False:
                self.y = y
            else:
                self.y = (~y.isna()).astype('int')  # 1 is CO, 0 no CO
            # self.y.columns = ['co_mob']
            # # 1) I need to add a סיומת to the y column
            # # so that it will certainly be unique, but this implementation should be good enough for this demo
            # # 2) if i was making this app generic i wouldn't demand "occupation"
            # # i would simply convert all catergorical data this way
            # if 'occupation' in X.columns:
            #     self.occupation_map = pd.concat([X, self.y], axis=1).groupby('occupation')['co_mob'].mean()

        numerical_X = self.__make_numerical(X)
        if self.good_columns is None:
            filtered_X = self.__filter_features(numerical_X)
            self.good_columns = filtered_X.columns
        else:
            filtered_X = numerical_X[self.good_columns]

        return filtered_X

    def __make_numerical(self, X):
        """
        Internal function for preprocessing X
        :param X: data where some of the columns are non-numerical
        :return: data with only numeric columns
        """
        return X.select_dtypes(include='number')

        # # if i want to make this function more like what i did in the assignment i would do this
        # processed_X = X.copy()
        #
        # # processed_X.drop('co_mob', axis=1, inplace=True)
        # # processed_X.drop('borrower_city', axis=1, inplace=True)
        #
        # # this is a non-generic method to handle dates.
        # if self.le is None:
        #     # we should get here if were calling this on data that needs fitting
        #     self.le = preprocessing.LabelEncoder()
        #     self.le.fit(processed_X.issue_date)
        # processed_X['issue_date'] = self.le.transform(processed_X.issue_date)
        #
        # # this is non-generic, for the demo
        # if 'occupation' in X.columns:
        #     processed_X['occupation'] = processed_X.occupation.map(self.occupation_map)
        #
        # # again, can should done genercly with sklearn.preprocessing.PowerTransformer
        # # perhaps it's better to not preprocess X... but then we aren't doing anything of value...
        # processed_X['loan_amnt'] = np.log(processed_X['loan_amnt'])
        #
        # # this whole meth
        # processed_X = processed_X.select_dtypes(include='number')
        #
        # return processed_X

    def __filter_features(self, X):
        """
        Internal function for preprocessing X
        Removes features that are too highly correlated with each other
        :param X: the training data
        :return: X with columns that their correlation is smaller then self.corr_thr
        """
        filtered_X = X.copy()

        while filtered_X.shape[1] > 1:  # there is a break inside the loop to end it.

            """
            calculates the correlation matrix for each feature. 
            since the correlation of each feature with itself is always maximal, and thus the largest,
            we need to look at the second most correlated feature to find how correlated a feature is 
            with another feature
            """
            mat = np.abs(filtered_X.corr().values)
            second_largest_idx = mat.argsort()[:, -2]
            second_largest_corr = []

            for i in range(mat.shape[0]):
                second_largest_corr.append(mat[i, second_largest_idx[i]])
            # for each feature, this shows how correlated the feature is with another
            second_largest_corr = np.array(second_largest_corr)

            # stopping condition: if all correlations are smaller than the threshold we break
            if np.max(second_largest_corr) < self.corr_thr:
                break

            # if we got here, it means we have atleast two highly correlated features
            # since the two features are correlated with each other, we want to determine witch one to drop
            columns_to_drop = filtered_X.columns[np.argpartition(second_largest_corr, -2)[-2:]]
            column1_corr = np.abs(filtered_X[columns_to_drop[0]].corr(self.y))
            column2_corr = np.abs(filtered_X[columns_to_drop[1]].corr(self.y))

            # we decide to drop the feature that is less correlated with our target variable
            if column1_corr > column2_corr:
                columns_to_drop = columns_to_drop[1]
            else:
                columns_to_drop = columns_to_drop[0]

            # here we drop the column with the highest internal correlation and the lowest correlation with the target
            filtered_X.drop(columns_to_drop, axis=1, inplace=True)

        return filtered_X


