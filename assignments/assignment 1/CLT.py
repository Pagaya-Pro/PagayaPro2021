import numpy as np
import pandas as pd

def CLT(S, n):
    num_samples = S.size // 2
    arr = np.random.choice(S, size = (num_samples, n), replace = True)
    return pd.Series(arr.mean(axis = 1))