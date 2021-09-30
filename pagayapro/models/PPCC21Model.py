import pandas as pd
import os
from pagayapro.paths.project_paths import MODELS_PATH

class PPCC21Model:
    def __init__(self):
        import pickle
        self.model = pickle.load(open(os.path.join(MODELS_PATH,"PPCC21.sav"),
                                      'rb'))
        self.columns = (
            pd.read_parquet(os.path.join(MODELS_PATH,"fcs.parquet")).fcs.values
        )
        
    def fit(self, X, y):
        self.model.fit(X[self.columns], y)
        return self
    
    def predict(self, X):
        return self.model.predict(X[self.columns])

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.columns])
