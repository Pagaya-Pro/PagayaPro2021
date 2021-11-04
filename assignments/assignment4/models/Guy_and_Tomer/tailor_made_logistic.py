import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

class tailor_made_logistic(BaseEstimator, ClassifierMixin):
    def __init__(self, class_weight=None):
        """
        Initialize the model. We allow for configuring the class weight.
        Parameters:
        class_weight (dict or ‘balanced’, default=None): 
        Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
        """
        self.class_weight = class_weight

    def set_occupation_feature(self, X: pd.DataFrame, y: np.array):
        """
        Create a series with all occupation classes in the training data as index,
        and the charge off rate per class as values.
        Add a wildcard case for unsees occupation classes with overall co rate.
        
        Parameters:
        X (pd.DataFrame): Data of features.
        y (np.array): Target values.
        """
        data = X.copy()
        
        data['co'] = y.values
        
        occupation_feature = data.groupby('occupation').mean()['co']
        occupation_feature.loc[r'.*'] = data['co'].mean()
        
        self.occupation_feature = occupation_feature
        
    def __preprocess(self, X: pd.DataFrame, y: np.array=None) -> pd.DataFrame:
        """
        Preprocess the features.
        When y is not None, we assume this is the training data and call set_occupation_feature.
        We assume the input data includes the features co_amount and co_mob.
        If they don't exist (or have a different name), an error will be raised.
        Parameters:
        X (pd.DataFrame): Data of features.
        y (np.array): Target values.
        Returns:
             Pandas DataFrame: preprocessed data.
        """
        data = X.copy()
        # drop correlated features
        correlated = [
            'g094s',
            'all803',
            'aut720',
            'all807',
            'monthly_pmt',
            'hr06s',
            'all801',
            'rep071',
            'rev302'
        ]
        data.drop(correlated, axis=1, inplace=True)
        # drop borrower_city, co_amount, and co_mob
        data.drop(['borrower_city','issue_date'], axis=1, inplace=True)
        
        # convert occupation column to co means per class
        data['occupation'].fillna('nan', inplace=True)
        if y is not None:
            self.set_occupation_feature(data, y)
        data['occupation'].replace(self.occupation_feature, inplace=True, regex=True)
        
        return data
    
    def fit(self, X: pd.DataFrame, y: np.array):
        """
        Preprocess the data and then use it to fit a logistic regression model.
        We assume the input data includes the features co_amount and co_mob.
        If they don't exist (or have a different name), an error will be raised.
        Parameters:
        X (pd.DataFrame): Data of features.
        y (np.array): Target values.
        Returns:
             Pandas DataFrame: preprocessed data.
        """
        data = self.__preprocess(X, y)
        
        self.clf = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic_reg', LogisticRegression(class_weight=self.class_weight))
        ])
        self.clf.fit(data, y)

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Given a new set of features, generate a prediction vector for
        whether each sample will result in a charge off.
        Parameters:
        X_test (pd.DataFrame): Data to test.
        Returns:
             np.array: Predicted target values.
        """
        data = self.__preprocess(X_test)
        
        return self.clf.predict(data)

    def predict_proba(self, X_test: pd.DataFrame) -> np.array:
        """
        Given a new set of features, generate a probability vector for
        whether each sample will result in a charge off.
        Parameters:
        X_test (pd.DataFrame): Data to test.
        Returns:
             np.array: Predicted probabilities for each of the target values.
        """
        data = self.__preprocess(X_test)
        
        return self.clf.predict_proba(data)