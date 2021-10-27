import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class Linear_regression_NM(LogisticRegression):

    def __init__(self,threshold=0.35,max_co_linearity=0.8):
        LogisticRegression.__init__(self)
        self.threshold = threshold
        self.max_co_linearity = max_co_linearity


    def fit(self, X, y):
        self.fit(X,y)
        print("ok")

    # predict(self,X,y):
    #     self.fit(X,y)
    #
    # predict_proba(self,X,y):
    #     self.predict_proba(X)