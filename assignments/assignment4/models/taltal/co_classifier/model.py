from preprocessor import Cleandata
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class Taltalim:
    def fit(self, X, y):
        self.pipe = Pipeline([('preprocessor', Cleandata()),
                              ('scaler', StandardScaler()),
                              ('logr', LogisticRegression(max_iter=1000))])
        self.pipe.fit(X, y)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] > threshold).astype('int')

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)



if __name__ == '__main__':
    import os
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    data = pd.read_csv('data.csv')
    X, y = data.drop(['co_mob', 'co_amount'], axis=1), (~data.co_mob.isna()).astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    cls = Taltalim()
    cls.fit(X_train, y_train)
    print(metrics.accuracy_score(y_test, cls.predict(X_test)))












