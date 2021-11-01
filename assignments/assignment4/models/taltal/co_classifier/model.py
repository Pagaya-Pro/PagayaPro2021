from co_classifier import preprocessor
from co_classifier.preprocessor import Cleandata
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class Taltalim:
    def fit(self, X, y):
        self.pipe = Pipeline([('preprocessor', Cleandata()), ('scaler', StandardScaler()), ('logr', LogisticRegression(max_iter=1000))])
        self.pipe.fit(X, y)

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X)[:, 1] > threshold

    def predict_proba(self, X):
        return self.predict_proba(X)


if __name__ == '__main__':
    import os
    from pagayapro.paths.data_paths import ASSIGNMENT4_DATA
    data = pd.read_parquet(os.path.join(ASSIGNMENT4_DATA, "prosper_data.parquet"))
    cls = Taltalim()
