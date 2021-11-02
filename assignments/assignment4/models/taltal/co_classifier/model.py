# from model import preprocessor
from preprocessor import Cleandata
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
# import co_classifier

class Taltalim (BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        self.c = Cleandata()
        X = self.c.fit_transform(X, y)
        self.s = StandardScaler()
        X = self.s.fit_transform(X, y)
        self.l = LogisticRegression(max_iter=1000)
        self.l.fit(X, y)

        # self.pipe = Pipeline([('preprocessor', Cleandata()), ('scaler', StandardScaler()), ('logr', LogisticRegression(max_iter=1000))])
        # self.pipe.fit(X, y)

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X)[:, 1] > threshold

    def predict_proba(self, X):
        X = self.c.transform(X)
        X = self.s.transform(X)
        return self.l.predict_proba(X)



if __name__ == '__main__':
    import os
    # from pagayapro.paths.data_paths import ASSIGNMENT4_DATA
    # data = pd.read_parquet(os.path.join(ASSIGNMENT4_DATA, "prosper_data.parquet"))
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    data = pd.read_csv('data.csv')
    # y = (~y.isna()).astype('int')
    X, y = data.drop(['co_mob', 'co_amount'], axis=1), (~data.co_mob.isna()).astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # y_test = (~y_test.isna()).astype('int')
    cls = Taltalim()
    cls.fit(X_train, y_train)
    print(np.isnan(cls.predict_proba(X_test)).sum())
    print(y_test)
    # cls.predict(X_test).sum()
    print(metrics.accuracy_score(y_test, cls.predict(X_test).astype('int')))
    # X, y = Cleandata().fit_transform(data.drop('co_mob', axis=1), data.co_mob)
    # print(type(X), type(y))
    # s = StandardScaler()
    # x__ = s.fit_transform(X, y)
    # print(x__.shape)
    # c = LogisticRegression(max_iter=1000)
    # c.fit(x__, y)
    print(X)










