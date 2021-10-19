import pandas as pd
import os
from pagayapro.paths.project_paths import MODELS_PATH
"""
PPCC21Model classifies loans at a high probability of CO. 
Trained on TU2016Q2.

Attributes:
 - self.model - the underlying model instance
 - self.columns - necessary features columns
 
Methods (sklearn-compatible):
 - self.fit() - fit the model to new data
 - self.predict() - predict T/F (will/will not CO)
 - self.predict_proba() - return predicted probability of CO
"""

class PPCC21Model:
    def __init__(self):
        import pickle
        self.model = pickle.load(open(os.path.join(MODELS_PATH,"PPCC21.sav"),
                                      'rb'))
        self.columns = (
            pd.read_parquet(os.path.join(MODELS_PATH,"fcs.parquet")).fcs.values
        )

    def __repr__(self):
        return "Pagaya PPCC21 ChargeOffClassifier"
        
    def fit(self, X, y):
        self.model.fit(X[self.columns], y)
        return self
    
    def predict(self, X):
        return self.model.predict(X[self.columns])

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.columns])
