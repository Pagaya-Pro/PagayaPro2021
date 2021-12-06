from sklearn.linear_model import LogisticRegression

class Logistic_regression_NM(LogisticRegression):

    def __init__(self,co_linearity_threshold=0.35,likelihood_threshold = 0.62):
        LogisticRegression.__init__(self)
        self.co_linearity_threshold = co_linearity_threshold
        self.likelihood_threshold = likelihood_threshold


    def fit(self, X, y):
        LogisticRegression.fit(self,X,y)
        print("ok")

    def predict(self,X):
        y_predict = LogisticRegression.predict_proba(self,X)
        return (y_predict[:,1] >= self.likelihood_threshold).astype('int')
