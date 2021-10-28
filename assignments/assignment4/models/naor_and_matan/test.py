from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from Linear_regression_NM import Linear_regression_NM
from PreProcessor_NM import PreProcessor_NM

# from pagayapro.paths.data_paths import ASSIGNMENT4_DATA
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os



def main():
    np.random.seed(1)
    ASSIGNMENT4_DATA = '~/Downloads/'
    data = pd.read_parquet(os.path.join(ASSIGNMENT4_DATA,"prosper_data.parquet"))
    from sklearn.model_selection import train_test_split
    y = 1 - data.co_mob.isna()
    data.drop(columns=['co_mob'], inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
    PreProcessor = PreProcessor_NM(co_linearity_threshold=0.4)

    model = Pipeline([('processor', PreProcessor_NM()),('scaler', StandardScaler()), ('lor', LogisticRegression())])

    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)

    print(f"accuracy is {(y_test==y_predict).mean()}")


if __name__ == "__main__":
    main()