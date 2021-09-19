import pandas as pd

class PPCC21Model:
    def __init__(self):
        import pickle
        import s3fs
        self.model = pickle.load(s3fs.S3FileSystem().open("s3://pagaya-pro-source/models/PPCC21.sav", 'rb'))
        self.columns = (
            pd.read_parquet(
                "s3://pagaya-pro-source/models/"
                "fcs.parquet").fcs.values
        )
        
    def fit(self, X, y):
        self.model.fit(X[self.columns], y)
        return self
    
    def predict(self, X):
        return self.model.predict(X[self.columns])
